# Reproduction of Spatial-MLLM

> Reproduced by AkagawaTsurunaki@Github.

## Paper Info

The original paper: [Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence](https://arxiv.org/abs/2505.23747)

Official repository: [diankun-wu/Spatial-MLLM](https://github.com/diankun-wu/Spatial-MLLM)

## Results

### Evaluation Results on VSI-Bench

Data presented in **bold** are excerpted from Table 1 of the original paper.

**Rel. Dir.** is the average of `object_rel_direction_easy`, `object_rel_direction_hard`, `object_rel_direction_medium` from the metrics file. 

| Methods                                 | Obj. Cnt. | Abs. Dist. | Obj. Size | Room Size | Rel. Dist. | Rel. Dir. | Route Plan | Appr. Order |   Avg.   |
| :-------------------------------------- | :-------: | :--------: | :-------: | :-------: | :--------: | :-------: | :--------: | :---------: | :------: |
| **Spatial-MLLM-4B**             | **65.3** |    **34.8**    | **63.1** | **45.1** |    **41.3**    | **46.2** |    **33.5**    | **46.3** |   **48.4**   |
| Spatial-MLLM-v1.1-Instruct-135K | 63.9  |  38.4  | 58.7  | 54.2  |  39.4  | 53.2  |  36.1  |  59.4   | 50.4 |
| Spatial-MLLM-v1.1-Instruct-820K |  66.7 |    36.1    |  70.2 |  56.9 |    49.3    |  47.5 |    37.6    |    58.9 | 52.9 |
| Spatial-MLLM-v1.1-Instruct-135K-16f (SA Sampling) | 66.2 | 38.1 | 58.4 | 56.7 | 41.0 | 61.2 | 34.5 | 53.7 | 51.2 |
| Spatial-MLLM-v1.1-Instruct-820K-16f | 66.4 | 38.1 | 69.6 | 56.8 | 46.3 | 54.3 | 40.2 | 54.0 | 53.2 |

### Using Space-aware Frame Sampling

Data presented in **bold** are excerpted from the official repository.


| Model                                             | VSIBench Micro |  |       | VSIBench Macro |       |       |
| ------------------------------------------------- | -------------- | -------------- | ----- | ----- | ----- | ----- |
| | Acc                                               | MRA            | All            | Acc   | MRA   | All   |
| **Spatial-MLLM-v1.1-Instruct-135K**       | **49.28**   | **52.88**    | **51.13** | **47.07** | **53.88** | **50.48** |
| Spatial-MLLM-v1.1-Instruct-135K | 49.28 | 52.91 | 51.15 | 47.03 | 53.80 | 50.42 |
| **Spatial-MLLM-v1.1-Instruct-135K (SA Sampling)** | **52.13**      | **53.33**      | **52.75** | **49.15** | **54.46** | **51.81** |
| Spatial-MLLM-v1.1-Instruct-135K (SA Sampling) | 51.04 | 53.47 | 52.29 | 47.61 | 54.86 | 51.24 |
| **Spatial-MLLM-v1.1-Instruct-820K**         | **49.56**      | **57.27**      | **53.53** | **48.60** | **57.39** | **53.00** |
| Spatial-MLLM-v1.1-Instruct-820K | 49.72 | 57.23 | 53.58 | 48.34 | 57.47 | 52.91 |
| **Spatial-MLLM-v1.1-Instruct-820K (SA Sampling)** | **50.60**      | **57.68**      | **54.24** | **48.78** | **58.09** | **53.43** |
| Spatial-MLLM-v1.1-Instruct-820K (SA Sampling) | 50.44 | 57.58 | 54.12 | 48.73 | 57.74 | 53.23 |

## Reproduction Details

All results were reproduced using **Ubuntu 22.04** with **CUDA 11.8** and were evaluated on the official models provided on Hugging Face.

### Hardware Info

- CPU: Intel i9 14900HF
- GPU: NVIDIA GeForce RTX 4090 (Driver Version: 580.82.07)
- Memory: 192 GB RAM

### Environment of Anaconda

**Python 3.10.19**

The following code block is the `environment.yml` created by Anaconda:

```yaml
name: spatial-mllm
channels:
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=1_gnu
  - bzip2=1.0.8=h5eee18b_6
  - ca-certificates=2025.12.2=h06a4308_0
  - expat=2.7.4=h7354ed3_0
  - ld_impl_linux-64=2.44=h153f514_2
  - libexpat=2.7.4=h7354ed3_0
  - libffi=3.4.4=h6a678d5_1
  - libgcc=15.2.0=h69a1729_7
  - libgcc-ng=15.2.0=h166f726_7
  - libgomp=15.2.0=h4751f2c_7
  - libnsl=2.0.0=h5eee18b_0
  - libstdcxx=15.2.0=h39759b7_7
  - libstdcxx-ng=15.2.0=hc03a8fd_7
  - libuuid=1.41.5=h5eee18b_0
  - libxcb=1.17.0=h9b100fa_0
  - libzlib=1.3.1=hb25bd0a_0
  - ncurses=6.5=h7934f7d_0
  - openssl=3.5.5=h1b28b03_0
  - packaging=25.0=py310h06a4308_1
  - pip=26.0.1=pyhc872135_0
  - pthread-stubs=0.3=h0ce48e5_1
  - python=3.10.19=h6fa692b_0
  - readline=8.3=hc2a1206_0
  - setuptools=80.10.2=py310h06a4308_0
  - sqlite=3.51.1=he0a8d7e_0
  - tk=8.6.15=h54e0aa7_0
  - wheel=0.46.3=py310h06a4308_0
  - xorg-libx11=1.8.12=h9b100fa_1
  - xorg-libxau=1.0.12=h9b100fa_0
  - xorg-libxdmcp=1.1.5=h9b100fa_0
  - xorg-xorgproto=2024.1=h5eee18b_1
  - xz=5.6.4=h5eee18b_1
  - zlib=1.3.1=hb25bd0a_0
  - pip:
      - accelerate==1.12.0
      - aiohappyeyeballs==2.6.1
      - aiohttp==3.13.3
      - aiosignal==1.4.0
      - annotated-types==0.7.0
      - anyio==4.12.1
      - async-timeout==5.0.1
      - attrs==25.4.0
      - av==16.1.0
      - certifi==2026.2.25
      - charset-normalizer==3.4.4
      - click==8.3.1
      - contourpy==1.3.2
      - cycler==0.12.1
      - datasets==4.6.1
      - decord==0.6.0
      - deepspeed==0.18.6
      - dill==0.4.0
      - docstring-parser==0.17.0
      - einops==0.8.2
      - exceptiongroup==1.3.1
      - filelock==3.20.0
      - flash-attn==2.7.4.post1
      - fonttools==4.61.1
      - frozenlist==1.8.0
      - fsspec==2025.12.0
      - gitdb==4.0.12
      - gitpython==3.1.46
      - h11==0.16.0
      - hf-xet==1.3.2
      - hjson==3.1.0
      - httpcore==1.0.9
      - httpx==0.28.1
      - huggingface-hub==0.36.2
      - idna==3.11
      - jinja2==3.1.6
      - jsonschema==4.26.0
      - jsonschema-specifications==2025.9.1
      - kiwisolver==1.4.9
      - levenshtein==0.27.3
      - markupsafe==3.0.2
      - matplotlib==3.10.8
      - mpmath==1.3.0
      - msgpack==1.1.2
      - multidict==6.7.1
      - multiprocess==0.70.18
      - networkx==3.4.2
      - ninja==1.13.0
      - numpy==2.2.6
      - nvidia-cublas-cu12==12.4.5.8
      - nvidia-cuda-cupti-cu12==12.4.127
      - nvidia-cuda-nvrtc-cu12==12.4.127
      - nvidia-cuda-runtime-cu12==12.4.127
      - nvidia-cudnn-cu12==9.1.0.70
      - nvidia-cufft-cu12==11.2.1.3
      - nvidia-curand-cu12==10.3.5.147
      - nvidia-cusolver-cu12==11.6.1.9
      - nvidia-cusparse-cu12==12.3.1.170
      - nvidia-cusparselt-cu12==0.6.2
      - nvidia-nccl-cu12==2.21.5
      - nvidia-nvjitlink-cu12==12.4.127
      - nvidia-nvtx-cu12==12.4.127
      - pandas==2.3.3
      - pillow==12.0.0
      - platformdirs==4.9.2
      - propcache==0.4.1
      - protobuf==6.33.5
      - psutil==7.2.2
      - py-cpuinfo==9.0.0
      - pyarrow==23.0.1
      - pydantic==2.12.5
      - pydantic-core==2.41.5
      - pyparsing==3.3.2
      - python-dateutil==2.9.0.post0
      - python-levenshtein==0.27.3
      - pytz==2025.2
      - pyyaml==6.0.3
      - qwen-vl-utils==0.0.14
      - rapidfuzz==3.14.3
      - ray==2.54.0
      - referencing==0.37.0
      - regex==2026.2.28
      - requests==2.32.5
      - rpds-py==0.30.0
      - safetensors==0.7.0
      - sentry-sdk==2.53.0
      - six==1.17.0
      - smmap==5.0.2
      - sympy==1.13.1
      - tokenizers==0.21.4
      - torch==2.6.0+cu124
      - torchaudio==2.6.0+cu124
      - torchvision==0.21.0+cu124
      - tqdm==4.67.3
      - transformers==4.51.3
      - triton==3.2.0
      - typeguard==4.5.1
      - typing-extensions==4.15.0
      - typing-inspection==0.4.2
      - tyro==1.0.8
      - tzdata==2025.3
      - urllib3==2.6.3
      - wandb==0.25.0
      - xxhash==3.6.0
      - yarl==1.22.0
```

If you have any questions, contact me through [AkagawaTsurunaki@outlook.com](mailto:AkagawaTsurunaki@outlook.com).

