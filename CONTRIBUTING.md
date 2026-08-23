# Contributing

Thank you for improving MagRobotNavGym. Please keep algorithmic changes small,
document whether they alter the paper baseline, and add a regression test.

This is a source-repository research release and is not published to PyPI.

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Before opening a pull request, verify that training configuration files remain
backward compatible and that an existing checkpoint can still be loaded.

Use `verbose=False` by default in reusable APIs. Plotting and interactive
examples belong under `examples/`, not in planner core modules.

Code adapted from another project must include its upstream URL, copyright and
license notice, and a corresponding entry in `THIRD_PARTY_LICENSES.md`.
