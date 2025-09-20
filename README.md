# cuda-grmonty

[![pre-commit](https://github.com/m-torhan/cuda-grmonty/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/m-torhan/cuda-grmonty/actions/workflows/pre-commit.yml)
[![bazel-build](https://github.com/m-torhan/cuda-grmonty/actions/workflows/bazel-build.yml/badge.svg)](https://github.com/m-torhan/cuda-grmonty/actions/workflows/bazel-build.yml)
[![bazel-test](https://github.com/m-torhan/cuda-grmonty/actions/workflows/bazel-test.yml/badge.svg)](https://github.com/m-torhan/cuda-grmonty/actions/workflows/bazel-test.yml)
[![bazel-coverage](https://github.com/m-torhan/cuda-grmonty/actions/workflows/bazel-coverage.yml/badge.svg)](https://github.com/m-torhan/cuda-grmonty/actions/workflows/bazel-coverage.yml)

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

## Development

### Testing

The tests can be ran with:

```bash
bazel test //cuda_grmonty/tests/...
```
