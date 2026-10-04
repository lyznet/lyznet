# LyZNet

LyZNet supports the analysis, learning, and formal verification of
certificates for dynamical systems.

This repository contains the new implementation and has not yet produced a
public release. See [DEVELOPMENT.md](DEVELOPMENT.md) for the current task and
release plan.

The previous implementation is preserved in
[lyznet-legacy](https://github.com/lyznet/lyznet-legacy).

## Current minimal workflow

After installing the package with `python -m pip install -e '.[test]'`, run:

```bash
python examples/minimal_workflows/system_definition/vdp.py
```

This script defines autonomous and control-affine Van der Pol systems and
evaluates their dynamics. Analysis and verification are added in later,
separately reviewable stages.
