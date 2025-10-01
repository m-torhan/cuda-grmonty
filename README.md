# cuda-grmonty

[![pre-commit](https://github.com/m-torhan/cuda-grmonty/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/m-torhan/cuda-grmonty/actions/workflows/pre-commit.yml)
[![bazel-build](https://github.com/m-torhan/cuda-grmonty/actions/workflows/bazel-build.yml/badge.svg)](https://github.com/m-torhan/cuda-grmonty/actions/workflows/bazel-build.yml)
[![bazel-test](https://github.com/m-torhan/cuda-grmonty/actions/workflows/bazel-test.yml/badge.svg)](https://github.com/m-torhan/cuda-grmonty/actions/workflows/bazel-test.yml)

CUDA implementation of relativistic Monte Carlo code.
Based on [grmonty](https://github.com/pseudotensor/grmonty) and [grmonty: A Monte Carlo Code for Relativistic Radiative Transport ](https://arxiv.org/abs/0909.0708).

## Building

Build dependencies:

- C++20 compiler (gcc/clang)
- CUDA
- [Bazel](https://github.com/bazelbuild/bazel)
- Python 3 (optional, for spectrum plotting)

### CPU build

To build CPU-only version, run:

```bash
bazel build -c opt //cuda_grmonty:main
```

then run it with:

```bash
./bazel-bin/cuda_grmonty/main --harm_dump_path ./dump019 --spectrum_path ./spectrum -photon_n 5000000
```

The HARM dump file can be downloaded from [grmonty](https://github.com/pseudotensor/grmonty) repository.

### GPU build

To build the program with CUDA acceleration, run:

```bash
bazel build -c opt //cuda_grmonty:main --config=cuda
```

and run it the same as above.

## Spectrum plot

To plot the spectrum, run `plot_spectrum.py`, preferably using [uv](https://github.com/astral-sh/uv) as follows:

```bash
uv run ./plot_spectrum.py --spectrum_path ./spectrum --plot_path ./spectrum.png
```

If you want to use Python directly instead, install dependencies specified in [plot_spectrum.py#4](plot_spectrum.py#L4). And run:

```bash
python3 ./plot_spectrum.py --spectrum_path ./spectrum --plot_path ./spectrum.png
```

## Benchmarks

Input parameters:

- HARM dump: [dump019](https://raw.githubusercontent.com/pseudotensor/grmonty/refs/heads/master/dump019)
- Estimate of photon number: 1000000
- Mass unit: 4.e19

System:

- Arch Linux (6.15.9-arch1-1)
- CUDA 12.9
- clang 20.1.8

Hardware:

- CPU: i9-14900k
- RAM: 2x32 GB (4800 Mhz)
- GPU: RTX 3060 12 GB

| Version                                                                | Other parameters / notes | Photon rate \[1/s\] | Total duration \[s\] |
| ---------------------------------------------------------------------- | ------------------------ | ------------------- | -------------------- |
| [grmonty](https://github.com/pseudotensor/grmonty)                     | `OMP_NUM_THREADS=1`      | 59743               | 272.51               |
| [grmonty](https://github.com/pseudotensor/grmonty)                     | `OMP_NUM_THREADS=32`     | 289866              | 55.36                |
| [v0.0.1](https://github.com/m-torhan/cuda-grmonty/releases/tag/v0.0.1) | CPU only                 | 51429               | 317.02               |
| [v0.1.0](https://github.com/m-torhan/cuda-grmonty/releases/tag/v0.1.0) |                          | 65119               | 241.84               |
| [v0.1.1](https://github.com/m-torhan/cuda-grmonty/releases/tag/v0.1.1) |                          | 137939              | 114.95               |
| [v0.1.2](https://github.com/m-torhan/cuda-grmonty/releases/tag/v0.1.2) |                          | 246538              | 64.91                |
| [v0.1.3](https://github.com/m-torhan/cuda-grmonty/releases/tag/v0.1.3) |                          | 307186              | 52.41                |

## Development

### Testing

The tests can be ran with:

```bash
bazel test //cuda_grmonty/tests/...
```
