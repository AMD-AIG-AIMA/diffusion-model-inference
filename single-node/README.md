# Running Distributed Diffusion Models Inference on AMD MI300X Accelerators
This repository shows how to run single-node distributed inference on the [Wan 2.1 Image to video model](https://github.com/Wan-Video/Wan2.1) and [HunyuanVideo video generation model](https://github.com/Tencent-Hunyuan/HunyuanVideo) on AMD MI300X accelerators.
## Setup
### Get Docker image
Pull the Docker image:
```
docker pull amdsiloai/pytorch-xdit:rocm6.4.0_ubuntu-22.04_py3.12_pytorch2.7
```

If you want to inspect the Dockerfile used to build the image, take a look at [`./Dockerfile`](./Dockerfile).

### Download models
Follow the instructions on how to install [`huggingface_hub`](https://github.com/huggingface/huggingface_hub).

Download the models (Note: the download size is tens of gigabytes):
```
export HF_HOME=~/.cache/huggingface
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P
huggingface-cli download tencent/HunyuanVideo --revision refs/pr/18
```
If you want, you can set the `HF_HOME` environment variable in the snippet above to download to a different directory.

## Run inference
### Wan 2.1 I2V
The following snippet creates a directory for the model outputs (if not already existing) on the host system and runs one step of inference with end-to-end benchmarking.
This configuration creates a 832x480 video with 81 frames. The Wan 2.1 model uses 16 frames per second, resulting in an output video length of approximately 5 seconds.
```
HOST_OUT_DIRECTORY=$(pwd)/wan_output
mkdir -p "${HOST_OUT_DIRECTORY}"

docker run \
  --rm \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --user root \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --network host \
  --privileged \
  --shm-size 128G \
  --mount type=bind,src=${HOST_OUT_DIRECTORY},dst=/output \
  --mount type=bind,src=${HF_HOME},dst=${HF_HOME} \
  -e HF_HOME=${HF_HOME} \
  amdsiloai/pytorch-xdit:rocm6.4.0_ubuntu-22.04_py3.12_pytorch2.7 \
  torchrun --nproc_per_node=8 /app/Wan2.1/wan_i2v.py \
      --task i2v-14B \
      --size 832*480 \
      --ckpt_dir ${HF_HOME}/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots/6b73f84e66371cdfe870c72acd6826e1d61cf279/ \
      --image /app/Wan2.1/examples/i2v_input.JPG \
      --ulysses_size 8 \
      --ring_size 1 \
      --save_file /output/video_480p_$(date +"%Y-%m-%d").mp4 \
      --benchmark_output_directory /output \
      --num_benchmark_steps 1 \
      --skip_warmup \
      --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```
The generated video is found in `${HOST_OUT_DIRECTORY}`.

The supported video sizes are `720*1280`, `1280*720`, `480*832`, `832*480`. However, note that for the 720p variants you should download and use the `Wan-AI/Wan2.1-I2V-14B-720P` model instead.

### HunyuanVideo
The following snippet creates a directory for the model outputs (if not already existing) on the host system and runs one step of inference with end-to-end benchmarking.
This configuration creates a 544x960 video with 129 frames. With 24 frames per second this results in an output video length of approximately 5 seconds.
```
HOST_OUT_DIRECTORY=$(pwd)/huvideo_output
mkdir -p "${HOST_OUT_DIRECTORY}"

docker run \
  --rm \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --user root \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --network host \
  --privileged \
  --shm-size 128G \
  --mount type=bind,src=${HOST_OUT_DIRECTORY},dst=/output \
  --mount type=bind,src=${HF_HOME},dst=${HF_HOME} \
  -e HF_HOME=${HF_HOME} \
  -e TOKENIZERS_PARALLELISM=false \
  amdsiloai/pytorch-xdit:rocm6.4.0_ubuntu-22.04_py3.12_pytorch2.7 \
  torchrun --nproc_per_node=8 hunyuan_video_usp_example_custom.py \
    --model tencent/HunyuanVideo \
    --batch_size 1 \
    --height 544 \
    --width 960 \
    --num_frames 129 \
    --num_inference_steps 50 \
    --warmup_steps 0 \
    --n_repeats 1 \
    --sleep_dur 0 \
    --bench_output /output \
    --ulysses_degree 8 \
    --enable_tiling \
    --enable_slicing
```
The generated video is found in `${HOST_OUT_DIRECTORY}`.

The Docker image provides [MIOpen tuning data](https://rocm.docs.amd.com/projects/MIOpen/en/latest/conceptual/tuningdb.html) for 129 frame generation on `960x544` (batch sizes 1, 4, 8, 16), `1280x720` (batch sizes 1, 4, 8, 16), `1920x1088` (batch size 1), and `1280x720` (batch size 1).

## Information on repository contents
Besides the inference sample code in [`./scripts/](./scripts/), this repository contains additional files and data to enable users to build the Dockerfile themselves if needed.

MIOpen tuning data is provided in the subdirectory [`./miopen/](./miopen/).

An implementation of `torch.nn.GroupNorm` optimized for MI300X is included in the Docker image, and distributed as a prebuilt wheel in [`./wheels/`](./wheels/).

A set of predefined example prompts for HunyuanVideo T2V are provided in [`./data/video_descriptions.csv`](./data/video_descriptions.csv). These are used by the inference script in the order they appear in for batch sizes larger than one.