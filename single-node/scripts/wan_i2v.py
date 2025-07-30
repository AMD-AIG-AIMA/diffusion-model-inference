# Adapted from The Alibaba Wan Team Authors, licensed under the Apache License, Version 2.0
# See NOTICE file for details.
from datetime import datetime
import argparse
import os
import logging
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from PIL import Image

from generate import _validate_args, _init_logging, EXAMPLE_PROMPT

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, str2bool

import json


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan and benchmark E2E inference time."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.",
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.",
    )
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.",
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.",
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.",
    )
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.",
    )
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.",
    )
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.",
    )
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="[image to video] The image to generate the video from.",
    )
    parser.add_argument(
        "--first_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (first frame) to generate the video from.",
    )
    parser.add_argument(
        "--last_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (last frame) to generate the video from.",
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
        help="The solver used to sample.",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps."
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.",
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--benchmark_output_directory",
        type=str,
        default=None,
        help="Benchmark output directory",
    )
    parser.add_argument(
        "--num_benchmark_steps",
        type=int,
        default=1,
        help="Number of benchmark steps (repetitions)",
    )
    parser.add_argument(
        "--skip_warmup",
        action="store_true",
        default=False,
        help="Whether to skip GPU warmup.",
    )

    args = parser.parse_args()

    _validate_args(args)

    return args


def benchmark_i2v(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert (
            args.ulysses_size * args.ring_size == world_size
        ), f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            initialize_model_parallel,
            init_distributed_environment,
        )

        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task or "flf2v" in args.task,
            )
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank,
            )
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}"
            )

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert (
            cfg.num_heads % args.ulysses_size == 0
        ), f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "i2v" not in args.task:
        logging.error(
            "Benchmarking script is only implemented for i2v task - use generate.py for other tasks."
        )
        sys.exit(1)

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None:
        args.image = EXAMPLE_PROMPT[args.task]["image"]
    logging.info(f"Input prompt: {args.prompt}")
    logging.info(f"Input image: {args.image}")

    img = Image.open(args.image).convert("RGB")
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                tar_lang=args.prompt_extend_target_lang,
                image=img,
                seed=args.base_seed,
            )
            if prompt_output.status == False:
                logging.info(f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    logging.info("Creating WanI2V pipeline.")
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    if args.skip_warmup:
        logging.warning("Skipping GPU warmup, hopefully this is what you intended!")
    else:
        logging.info("Running warmup for up to 2 minutes...")
        warmup_start_time = time.time()
        warmup_iteration = 0
        warmup_sample_steps = 5
        while time.time() - warmup_start_time < 120:
            warmup_iteration += 1
            logging.info(
                f"Warmup iteration {warmup_iteration}; using {warmup_sample_steps=} instead of {args.sample_steps=}"
            )
            video = wan_i2v.generate(
                args.prompt,
                img,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=warmup_sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )

    for benchmark_step in range(1, args.num_benchmark_steps + 1):
        logging.info(f"Generating video for benchmark step {benchmark_step} ...")
        start_time = time.perf_counter()
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )
        end_time = time.perf_counter()
        if rank == 0:
            logging.info(
                f"Benchmark step {benchmark_step} elapsed time: {end_time - start_time:.2f} seconds."
            )

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = ".png" if "t2i" in args.task else ".mp4"
            args.save_file = (
                f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}"
                + suffix
            )

        # Dump benchmark args
        d_args = vars(args)
        args_file = os.path.join(args.benchmark_output_directory, "args.json")
        with open(args_file, "w") as fp:
            json.dump(d_args, fp, indent=4)

        # Make sure video is saved to same directory as benchmark
        video_save_file = os.path.join(args.benchmark_output_directory, args.save_file)

        logging.info(f"Saving generated video to {args.save_file}")
        cache_video(
            tensor=video[None],
            save_file=video_save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    benchmark_i2v(args)
