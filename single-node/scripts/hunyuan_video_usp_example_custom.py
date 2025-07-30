# Adapted from xDiT Team, licensed under the Apache License, Version 2.0
# See NOTICE file for details.

# Benchmark batch sizes >= 1

import os
import sys
import time

import huvideo_utils as ut
import torch
from accelerate.utils import set_seed
from xfuser import xFuserArgs
from xfuser.core.distributed import (
    get_runtime_state,
    get_world_group,
    is_dp_last_group,
)

if torch.version.hip:
    try:
        from opt_groupnorm import OPTGroupNorm

        torch.nn.GroupNorm = OPTGroupNorm
        print("Using OPTGroupNorm as torch.nn.GroupNorm")
    except ImportError:
        print("Using torch.nn.GroupNorm")


def main():
    args = ut.create_arg_parser()
    engine_args = xFuserArgs.from_cli_args(args)

    engine_config, input_config = engine_args.create_config()

    if not input_config.prompt:
        prompts = ut.get_prompts()
        args.prompt = (
            prompts[0 : args.batch_size] if args.batch_size > 1 else [prompts[0]]
        )
        input_config.prompt = args.prompt
        input_config.batch_size = args.batch_size

    set_seed(input_config.seed)
    local_rank = get_world_group().local_rank
    global_rank = get_world_group().rank_in_group
    node_rank = global_rank // 8
    print(
        f"Node rank: {node_rank}, Global rank: {global_rank}, Local rank: {local_rank}"
    )

    warmup_steps = engine_config.runtime_config.warmup_steps

    if get_world_group().rank == get_world_group().world_size - 1:
        ut.print_model_args(args, engine_config)

    if engine_args.pipefusion_parallel_degree != 1:
        raise RuntimeError("This script does not support PipeFusion.")
    if engine_args.use_parallel_vae:
        raise RuntimeError("parallel VAE not implemented for HunyuanVideo")

    pipe = ut.load_model(input_config, engine_config)
    ut.parallelize_transformer(pipe)
    ut.pipe_reduce_memory_usage(pipe, args, local_rank)

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    if warmup_steps > 0:
        print(f"Rank {global_rank} warming up")
        if engine_config.runtime_config.use_torch_compile:
            print(f"Rank {global_rank} compiling")
            torch._inductor.config.reorder_for_compute_comm_overlap = True
            pipe.transformer = torch.compile(pipe.transformer, mode="default")

        for _ in range(warmup_steps):
            _ = ut.run_pipe(pipe, input_config, warmup=True)

    ut.clean_cache()
    time.sleep(args.sleep_dur)

    if args.bench_output:
        if is_dp_last_group():
            os.makedirs(args.bench_output, exist_ok=True)

        for i in range(1, args.n_repeats + 1):
            torch.cuda.reset_peak_memory_stats()

            pipe_events = {
                "before_pipe": torch.cuda.Event(enable_timing=True),
                "after_pipe": torch.cuda.Event(enable_timing=True),
            }

            pipe_events["before_pipe"].record()
            output = ut.run_pipe(pipe, input_config)

            pipe_events["after_pipe"].record()

            torch.cuda.synchronize(local_rank)

            peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
            pipe_elapsed_time = (
                pipe_events["before_pipe"].elapsed_time(pipe_events["after_pipe"])
                / 1000
            )

            if get_world_group().rank == get_world_group().world_size - 1:
                print(f"Iteration {i}")
                print(f"Pipe epoch time: {pipe_elapsed_time:.2f} sec")
                print(
                    f"Parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
                )

            ut.clean_cache()
            time.sleep(args.sleep_dur)

        if is_dp_last_group() and input_config.output_type != "latent":
            ut.latents_to_video(
                output,
                input_config,
                engine_args,
                engine_config,
                args,
                args.bench_output,
            )
    else:
        get_runtime_state().destroy_distributed_env()
        sys.exit(f"Empty path (bench: {args.bench_output}")

    get_runtime_state().destroy_distributed_env()
    ut.clean_cache()


if __name__ == "__main__":
    main()
