# Adapted from xDiT Team, licensed under the Apache License, Version 2.0
# See NOTICE file for details.
import csv
import functools
import gc
import logging
import os.path as osp
from typing import Any, Dict, Optional, Union

import torch
from diffusers import (
    DiffusionPipeline,
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    export_to_video,
    scale_lora_layers,
    unscale_lora_layers,
)
from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_pipeline_parallel_world_size,
    get_runtime_state,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    get_world_group,
    initialize_runtime_state,
)
from xfuser.model_executor.layers.attention_processor import (
    xFuserHunyuanVideoAttnProcessor2_0,
)

if xFuserHunyuanVideoAttnProcessor2_0 is None:
    raise RuntimeError("xFuserHunyuanVideoAttnProcessor2_0 is not available.")


def create_arg_parser():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument(
        "--batch_size",
        help="The batch size.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=10,
        help="The number of benchmark repetitions.",
    )
    parser.add_argument(
        "--sleep_dur",
        type=int,
        default=30,
        help="The duration to sleep in between different pipe calls in seconds.",
    )
    parser.add_argument(
        "--bench_output",
        type=str,
        default="",
        help="When provided, run benchmarking to record E2E inference time.",
    )
    return xFuserArgs.add_cli_args(parser).parse_args()


def print_model_args(args, engine_config):
    """
    Prints all model arguments for logging.
    """

    def print_args(args):
        for key, val in vars(args).items():
            print(f"{key:>35} = {val}", flush=True)
        print("\n", flush=True)

    print(f"{'-' * 30}PARAMETERS{'-' * 30}")
    print(f"{'filename':>35} = hunyuan_video_usp_example_custom.py")
    print_args(args)
    print_args(engine_config.parallel_config.sp_config)
    print(f"{'-' * 30}PARAMETERS{'-' * 30}")


def clean_cache():
    torch.cuda.empty_cache()
    gc.collect()


def load_model(input_config, engine_config):
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        revision="refs/pr/18",
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        transformer=transformer,
        torch_dtype=torch.float16,
        revision="refs/pr/18",
    )

    initialize_runtime_state(pipe, engine_config)
    get_runtime_state().set_video_input_parameters(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        batch_size=input_config.batch_size,
        num_inference_steps=input_config.num_inference_steps,
        split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
    )

    return pipe


def pipe_reduce_memory_usage(pipe, args, local_rank):
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    if args.enable_tiling:
        pipe.vae.enable_tiling(
            # Make it runnable on GPUs with 48GB memory
            # tile_sample_min_height=128,
            # tile_sample_stride_height=96,
            # tile_sample_min_width=128,
            # tile_sample_stride_width=96,
            # tile_sample_min_num_frames=32,
            # tile_sample_stride_num_frames=24,
        )

    if args.enable_slicing:
        pipe.vae.enable_slicing()


def get_prompts():
    prompts = []

    with open("video_descriptions.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",", quotechar='"')
        prompts = [row[1] for row in reader][1:]

    if not prompts:
        print("WARNING! Zero prompts found.")

    return prompts


def run_pipe(pipe, input_config, warmup=False):
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=1 if warmup else input_config.num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=input_config.max_sequence_length,
        guidance_scale=3.5,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    return output


def latents_to_video(
    output, input_config, engine_args, engine_config, args, output_path
):
    for i, curr_output_frames in enumerate(output.frames):
        parallel_info = (
            f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
            f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
            f"tp{engine_args.tensor_parallel_degree}_"
            f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}_"
            f"frames{input_config.num_frames}_steps{input_config.num_inference_steps}_seqlen{input_config.max_sequence_length}_"
            f"tiling{args.enable_tiling}_slicing{args.enable_slicing}_offload{args.enable_model_cpu_offload}_"
            f"tc{engine_config.runtime_config.use_torch_compile}"
        )

        resolution = f"{input_config.width}x{input_config.height}"
        output_filename = osp.join(
            output_path,
            f"hunyuan_video_{i+1}_bs{input_config.batch_size}_{parallel_info}_{resolution}.mp4",
        )
        export_to_video(curr_output_frames, output_filename, fps=24)
        print(f"output saved to {output_filename}")


def parallelize_transformer(pipe: DiffusionPipeline):
    transformer = pipe.transformer

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logging.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        if batch_size % get_classifier_free_guidance_world_size() != 0:
            raise RuntimeError(
                f"Cannot split dim 0 of hidden_states ({batch_size}) into {get_classifier_free_guidance_world_size()} parts."
            )

        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, timestep, encoder_attention_mask
        )

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1
        )
        hidden_states = hidden_states.flatten(1, 3)

        hidden_states = torch.chunk(
            hidden_states, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(
            hidden_states, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]

        encoder_attention_mask = encoder_attention_mask.to(torch.bool).any(dim=0)
        encoder_hidden_states = encoder_hidden_states[:, encoder_attention_mask, :]

        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size() != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True

        encoder_hidden_states = torch.chunk(
            encoder_hidden_states, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(
                encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2
            )[get_sequence_parallel_rank()]

        freqs_cos, freqs_sin = image_rotary_emb

        def get_rotary_emb_chunk(freqs):
            freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=0)[
                get_sequence_parallel_rank()
            ]
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos)
        freqs_sin = get_rotary_emb_chunk(freqs_sin)
        image_rotary_emb = (freqs_cos, freqs_sin)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):

                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}

            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        None,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        None,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, None, image_rotary_emb
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, None, image_rotary_emb
                )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)
        hidden_states = get_cfg_group().all_gather(hidden_states, dim=0)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            -1,
            p_t,
            p,
            p,
        )

        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

    for block in transformer.transformer_blocks + transformer.single_transformer_blocks:
        block.attn.processor = xFuserHunyuanVideoAttnProcessor2_0()
