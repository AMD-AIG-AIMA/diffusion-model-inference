# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM rocm/dev-ubuntu-22.04:6.4-complete AS base

WORKDIR /opt
ARG DEBIAN_FRONTEND=noninteractive
ARG HIPBLASLT_COMMIT=08838a490b76cd23c4db6fef987d0c5b874887b5
ARG HIPBLAS_COMMON_COMMIT=74c51f29ceda1671b4e62e5b708e52601a9e7883
ARG MIOPEN_COMMIT=983e058f8e3b1d07ccd5b531e46890e544d26352
ARG ROCM_VERSION=6.4.0
ARG GPU_ARCHS=gfx942


ENV PIP_NO_CACHE_DIR=1
ENV GPU_ARCHS=${GPU_ARCHS}
ENV ROCM_PATH=/opt/rocm \
    MPI_HOME=/opt/ompi \
    UCX_HOME=/opt/ucx \
    PATH=/opt/ompi/bin:/opt/ucx/bin:/opt/cache/bin:/opt/rocm/llvm/bin:/opt/rocm/opencl/bin:/opt/rocm/hip/bin:/opt/rocm/hcc/bin:/opt/rocm/bin:/opt/conda/envs/py_3.10/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/ompi/lib:/opt/rocm/lib:/usr/local/lib

# Install required dependencies
RUN apt-get update && apt-get install -y \
    git \
    nano \
    wget \
    ninja-build \
    sqlite3 \
    libsqlite3-dev \
    libbz2-dev \
    python3.10-venv \
    python3-pip \
    libzstd-dev \
    linux-headers-$(uname -r) \
    libelf-dev \
    libmnl0 \
    udev \
    dmidecode \
    iproute2 \
    libbpf0 \
    libxtables12 \
    build-essential \
    gcc \
    libtool \
    autoconf \
    librdmacm-dev \
    rdmacm-utils \
    infiniband-diags \
    ibverbs-utils \
    perftest \
    ethtool \
    libibverbs-dev \
    rdma-core \
    strace \
    libibmad5 \
    libibnetdisc5 \
    ibverbs-providers \
    libibumad-dev \
    libibumad3 \
    libibverbs1 \
    libnl-3-dev \
    libnl-route-3-dev \
    unzip \
    libfmt-dev \
    libmsgpack-dev \
    libsuitesparse-dev \
    ccache \
    pkg-config \
    liblzma-dev \
    patchelf \
    libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache --upgrade pip setuptools \
    && pip install --no-cache ninja setuptools wheel ipython pytest fire pydantic pybind11 cmake==3.31.6 \
    && cmake --version

WORKDIR /opt/rocm/

# Upgrade ROCm libraries
# hipblaslt
RUN git clone https://github.com/ROCm/hipBLAS-common.git \
    && cd hipBLAS-common \
    && git checkout ${HIPBLAS_COMMON_COMMIT}    \
    && rm -rf /opt/rocm/include/hipblas-common \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make package install \
    && cd ../../ && rm -rf ./hipBLAS-common

RUN git clone https://github.com/rocm/hipblaslt  \
    && cd hipblaslt     \
    && git checkout ${HIPBLASLT_COMMIT}    \
    && rm -rf /opt/rocm/include/hipblaslt     \
    && rm -rf /opt/rocm/lib/libhipblaslt*     \
    && cmake --version \
    && ./install.sh -idc --architecture ${GPU_ARCHS}     \
    && cd ../ \
    && rm -rf hipblaslt

# MIOpen
FROM base AS miopen
ARG MIOPEN_COMMIT
ARG MIOPEN_DIR=/opt/MIOpen/install_dir

RUN apt remove -y rocm-libs && apt-get purge -y --allow-unauthenticated \
    composablekernel-dev miopen-hip miopen-hip-dev

RUN cd /opt \
    && git clone https://github.com/ROCm/MIOpen \
    && cd MIOpen \
    && git reset --hard ${MIOPEN_COMMIT} \
    && rm -rf /opt/rocm-${ROCM_VERSION}/include/miopen     \
    && rm -rf /opt/rocm-${ROCM_VERSION}/lib/cmake/miopen \
    && rm -f /opt/rocm-${ROCM_VERSION}/lib/libMIOpen* \
    && cmake -P install_deps.cmake --minimum --prefix ${MIOPEN_DIR}

RUN cd /opt/MIOpen \
    && mkdir -p build \
    && cd build \
    && CXX=/opt/rocm/llvm/bin/clang++ CXXFLAGS='-Werror' cmake -DMIOPEN_TEST_FLAGS=' --disable-verification-cache ' -DCMAKE_BUILD_TYPE=Release -DBUILD_DEV=Off -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/hip;${MIOPEN_DIR}" -DCMAKE_INSTALL_PREFIX=/opt/rocm  .. \
    && make -j $(nproc) \
    && make install \
    && make package

FROM base AS rocm_final
RUN dpkg --force-all --remove rocm-libs miopen-hip miopen-hip-dev
RUN --mount=type=bind,from=miopen,source=/opt/MIOpen/build,target=/build \
    DEBIAN_FRONTEND=noninteractive dpkg -i --force-all /build/*.deb

FROM rocm_final AS hipblastlt_miopen_base

ARG TRITON_BRANCH="981e987e"
ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ARG PYTORCH_BRANCH="295f2ed4"
ARG PYTORCH_VISION_BRANCH="v0.21.0"
ARG PYTORCH_REPO="https://github.com/pytorch/pytorch.git"
ARG PYTORCH_VISION_REPO="https://github.com/pytorch/vision.git"
ARG FA_BRANCH="fd2fc9d"
ARG FA_REPO="https://github.com/Dao-AILab/flash-attention.git"

ENV PATH=/opt/rocm/llvm/bin:$PATH
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:
ARG PYTORCH_ROCM_ARCH=gfx942
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}

ARG PYTHON_VERSION=3.12

RUN mkdir -p /app
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update -y \
    && apt-get install -y software-properties-common git curl sudo vim less libgfortran5 \
    && for i in 1 2 3; do \
        add-apt-repository -y ppa:deadsnakes/ppa && break || \
        { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
    done \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
       python${PYTHON_VERSION}-lib2to3 python-is-python3  \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

RUN pip install -U packaging 'cmake<4' ninja wheel 'setuptools<80' pybind11 Cython


FROM hipblastlt_miopen_base AS build_amdsmi
RUN cd /opt/rocm/share/amd_smi \
    && pip wheel . --wheel-dir=dist
RUN mkdir -p /app/install && cp /opt/rocm/share/amd_smi/dist/*.whl /app/install

FROM hipblastlt_miopen_base AS build_triton_pytorch
ARG TRITON_BRANCH
ARG TRITON_REPO
ARG PYTORCH_BRANCH
ARG PYTORCH_VISION_BRANCH
ARG PYTORCH_REPO
ARG PYTORCH_VISION_REPO
ARG FA_BRANCH
ARG FA_REPO
ARG MAX_JOBS=$nproc

RUN git clone ${TRITON_REPO}
RUN cd triton \
    && git checkout ${TRITON_BRANCH} \
    && cd python \
    && python3 setup.py bdist_wheel --dist-dir=dist
RUN mkdir -p /app/install && cp /app/triton/python/dist/*.whl /app/install

RUN git clone ${PYTORCH_REPO} pytorch
RUN cd pytorch && git checkout ${PYTORCH_BRANCH} \
    && git submodule update --init --recursive \
    && pip install -r requirements.txt \
    && pip install numpy==2.2.6 \
    && python3 tools/amd_build/build_amd.py \
    && _GLIBCXX_USE_CXX11_ABI=1 MAX_JOBS=${MAX_JOBS} BUILD_TEST=0 python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl
RUN git clone ${PYTORCH_VISION_REPO} vision
RUN cd vision && git checkout ${PYTORCH_VISION_BRANCH} \
    && python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl
RUN git clone ${FA_REPO}
RUN cd flash-attention \
    && git checkout ${FA_BRANCH} \
    && git submodule update --init \
    && GPU_ARCHS=$(echo ${PYTORCH_ROCM_ARCH} | sed -e 's/;gfx1[0-9]\{3\}//g') MAX_JOBS=${MAX_JOBS} python3 setup.py bdist_wheel --dist-dir=dist
RUN mkdir -p /app/install && cp /app/pytorch/dist/*.whl /app/install \
    && cp /app/vision/dist/*.whl /app/install \
    && cp /app/flash-attention/dist/*.whl /app/install


FROM hipblastlt_miopen_base AS final
RUN --mount=type=bind,from=build_triton_pytorch,src=/app/install/,target=/install \
    pip install /install/*.whl
RUN --mount=type=bind,from=build_amdsmi,src=/app/install/,target=/install \
    pip install /install/*.whl

ARG BASE_IMAGE
ARG TRITON_BRANCH
ARG TRITON_REPO
ARG PYTORCH_BRANCH
ARG PYTORCH_VISION_BRANCH
ARG PYTORCH_REPO
ARG PYTORCH_VISION_REPO
ARG FA_BRANCH
ARG FA_REPO
RUN echo "BASE_IMAGE: ${BASE_IMAGE}" > /app/versions.txt \
    && echo "TRITON_BRANCH: ${TRITON_BRANCH}" >> /app/versions.txt \
    && echo "TRITON_REPO: ${TRITON_REPO}" >> /app/versions.txt \
    && echo "PYTORCH_BRANCH: ${PYTORCH_BRANCH}" >> /app/versions.txt \
    && echo "PYTORCH_VISION_BRANCH: ${PYTORCH_VISION_BRANCH}" >> /app/versions.txt \
    && echo "PYTORCH_REPO: ${PYTORCH_REPO}" >> /app/versions.txt \
    && echo "PYTORCH_VISION_REPO: ${PYTORCH_VISION_REPO}" >> /app/versions.txt \
    && echo "FA_BRANCH: ${FA_BRANCH}" >> /app/versions.txt \
    && echo "FA_REPO: ${FA_REPO}" >> /app/versions.txt

ARG WAN_REPO="https://github.com/Wan-Video/Wan2.1.git"
ARG WAN_COMMIT="204f899b6436fe2e1705a0b67c464b30b8137799"
ARG YUNCHANG_REPO="https://github.com/feifeibear/long-context-attention"
ARG YUNCHANG_COMMIT="b192e97a0fff361dd395032d9daea2cb1259ab02"

WORKDIR /app

# Install OPTGroupNorm
RUN --mount=type=bind,source=./wheels/opt_groupnorm-0.0.0-cp312-cp312-linux_x86_64.whl,target=/wheels/opt_groupnorm-0.0.0-cp312-cp312-linux_x86_64.whl \
    pip install --no-cache-dir /wheels/opt_groupnorm-0.0.0-cp312-cp312-linux_x86_64.whl

# Install MIOpen database
RUN --mount=type=bind,source=./miopen,target=/miopen \
    mkdir -p /root/.config/miopen \
    && mkdir -p /root/.cache/miopen \
    && cp -r /miopen/config/* /root/.config/miopen \
    && cp -r /miopen/cache/* /root/.cache/miopen

# Install yunchang, diffusers xDiT
RUN git clone ${YUNCHANG_REPO} long-context-attention && \
    cd long-context-attention && \
    git checkout ${YUNCHANG_COMMIT} && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir \
    diffusers==0.32.2 \
    xfuser==0.4.3.post3

# Install Wan2.1
RUN git clone ${WAN_REPO} Wan2.1 \
    && cd Wan2.1 && git checkout ${WAN_COMMIT} && cd .. \
    && pip install --no-cache-dir -r Wan2.1/requirements.txt

# Copy all scripts and data
COPY scripts/wan_i2v.py ./Wan2.1
COPY scripts/hunyuan_video_usp_example_custom.py scripts/huvideo_utils.py ./
COPY data/* ./