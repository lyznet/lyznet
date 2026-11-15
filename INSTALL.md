# Installation

LyZNet 0.1.0 is tested with Python 3.11 on Linux.

## Base package

```bash
python -m pip install lyznet
```

This installs NumPy 1.26 or later, SciPy 1.12 or later, and SymPy 1.12 or
later.

## Optional Python dependencies

Install only the capabilities you need:

```bash
python -m pip install 'lyznet[z3]'
python -m pip install 'lyznet[learning,plot]'
```

The extras install the following packages:

| Extra | Requirement |
| --- | --- |
| `z3` | `z3-solver>=4.13` |
| `dreal` | `dreal>=4.21.6.2` Python binding |
| `learning` or `crown` | `torch>=2.11,<2.12` |
| `plot` | `matplotlib>=3.8` |
| `test` | `pytest>=8` |

For GPU learning, install the appropriate PyTorch 2.11 build from the
[PyTorch installation guide](https://pytorch.org/get-started/locally/)
before installing LyZNet.

## dReal

Install dReal and its prerequisites by following the
[dReal installation guide](https://github.com/dreal/dreal4#how-to-install).
Then install its Python binding:

```bash
python -m pip install 'lyznet[dreal]'
```

## α,β-CROWN (alpha-beta-CROWN)

Install α,β-CROWN 0.7 by following its
[installation guide](https://github.com/Verified-Intelligence/alpha-beta-CROWN).
Then install LyZNet's matching Python dependencies. LyZNet exposes the
verifier through its `crown` backend.

```bash
python -m pip install 'lyznet[crown,plot]'
```
