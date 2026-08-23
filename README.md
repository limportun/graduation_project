# STTRL-DVO

[![IEEE T-RO](https://img.shields.io/badge/IEEE%20T--RO-10.1109%2FTRO.2026.3701578-blue)](https://doi.org/10.1109/TRO.2026.3701578)
[![Tests](https://github.com/fanghaow/STTRL-DVO/actions/workflows/tests.yml/badge.svg)](https://github.com/fanghaow/STTRL-DVO/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official source code for **STTRL-DVO: Transformer-based Reinforcement
Learning for Robust Dynamic Target Tracking in Cluttered Environments**,
accepted by *IEEE Transactions on Robotics*.

STTRL-DVO combines a spatial-temporal transformer reinforcement-learning
policy with deterministic velocity-obstacle guidance for magnetic microrobot
navigation and dynamic-target tracking.

## Demonstrations

| Simulation | Real-world experiment |
| --- | --- |
| ![Simulation navigation](demo/dynamic_sim.gif) | ![Real-world navigation](demo/dynamic_real.gif) |

## Repository scope

This is a **source-repository research release**. Clone the repository and install it in editable mode so that the
training scripts and experiment configuration remain available.

The repository provides the simulation environment, STTRL network, DVO reward,
training/evaluation scripts, and rule-based baselines.

## Installation

```bash
git clone https://github.com/fanghaow/STTRL-DVO.git
cd STTRL-DVO
conda env create -f environment.yml
conda activate sttrl-dvo
python -m pip install -e . --no-deps
```

The provided Conda environment installs the reference CUDA 12.4 build of
PyTorch. For a CPU-only installation, create a Python 3.9 environment and run
`python -m pip install -e .` instead.

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The reference versions observed in the authors' training environment are
listed in [requirements.txt](requirements.txt) and [environment.yml](environment.yml).
They intentionally contain only direct dependencies required by this project.

### Reference hardware and training setup

- OS: Ubuntu 20.04
- GPU: NVIDIA GeForce RTX 4090 D, 24 GB
- CPU: Intel Core i7
- Parallel simulation environments: 8
- Training budget: 3,000,000 aggregate environment steps per seed,
collected across eight parallel environments (approximately 376,832 steps per env).
- Random seeds used for learning-based comparisons: 1 through 8
- Reported STTRL-DVO training time: 1.68 ± 0.02 hours per run

The local reference installation uses Python 3.9.7, PyTorch 2.5.0+cu124,
Gymnasium 0.29.1, and Stable-Baselines3 2.2.1. The CUDA environment follows
PyTorch's official 2.5.0 installation matrix; use a different supported build
when required by the local driver or platform.

## Training

The default configuration is
[params_baseline/STTRL_DVO/config.yaml](params_baseline/STTRL_DVO/config.yaml).
It uses eight parallel environments and the default anti-jitter ("smoothed") DVO heading
selection. Run one seed with:

```bash
python -m scripts.train --seeds 1
```

Run the eight learning-based seeds with:

```bash
python -m scripts.train --seeds 1-8
```

Checkpoints and TensorBoard logs are written under `checkpoint/` and
`tensorboard/`; both directories are ignored by Git.

## Pretrained checkpoint

The pretrained 3,000,000-step policy is included at:

```text
checkpoint/STTRL_DVO/pretrained/ppo_multi_3000000_steps.zip
```

## Evaluation

```bash
# Learned STTRL-DVO policy
python -m scripts.test --methods sttrl_dvo --seeds 1 --episodes 100

# Evaluate a specific checkpoint; {seed} and {model_step} are supported
python -m scripts.test --methods sttrl_dvo --model-step 2700000 \
  --checkpoint-path "checkpoint/STTRL_DVO/PPO_{seed}/ppo_multi_{model_step}_steps.zip"

# Rule-based comparisons in the base Circle environment
python -m scripts.test --methods dwa,dstar_mpc,pid_dvo,pid_vo --seeds 1-8 --episodes 100

# Evaluate an unseen environment
python -m scripts.test --methods sttrl_dvo --env maze --episodes 100

# Keep the base Circle obstacle density in a different environment
python -m scripts.test --methods sttrl_dvo --env square \
  --obstacle-density-ratio 1 --episodes 100

# Short rendered evaluation with LiDAR-patch attention weights
python -m scripts.test --methods sttrl_dvo --seeds 1 --episodes 5 --visualize-attention
```

Cumulative results are appended to `algorithm_statistics.csv` after every
episode, so interrupted evaluations retain completed work. Add `--verbose` to
print controller timing and planner-fallback diagnostics. Run
`python -m scripts.test --help` for all scenario and output options.

By default, each environment uses the Circle reference density (10 dynamic
obstacles in the base Circle). Counts for Square, Vessel, and Maze are computed
from their outer boundary Polygon and rounded down. An explicit
`--obstacle-density-ratio` takes priority over `--obstacle-num`; an explicit
absolute count is used only when the density ratio is omitted.

## Project structure

```text
mag_robot_nav/
  envs/                         Gymnasium environment, DVO, LiDAR, rendering
  planners/                     Classical planning and control baselines
scripts/
  train.py                      PPO/STTRL-DVO training
  test.py                       Learned and rule-based evaluation
torchrl/networks/               Project-internal RL neural-network modules
params_baseline/STTRL_DVO/      Reference experiment configuration
examples/                       Standalone planner examples
tests/                          Regression tests
```

The top-level `torchrl` directory is an internal module containing the neural
networks used by this project. It is not the external TorchRL library.

Planner APIs are exported from `mag_robot_nav.planners`; optional heavy
dependencies such as CVXPY are loaded only when the corresponding planner is
requested. Reusable APIs are quiet by default and accept `verbose=True` for
ordinary diagnostics.

## Citation

If this repository supports your research, please cite:

```bibtex
@article{wang2026sttrldvo,
  author  = {Fanghao Wang and Binghong Chen and Youchao Zhang and Xiangyu Guo and
             Yining Lyu and Chuanjie Liu and Alois Knoll and Di Cui and
             Huanyu Jiang and Yibin Ying and Mingchuan Zhou},
  title   = {{STTRL-DVO}: Transformer-based Reinforcement Learning for Robust
             Dynamic Target Tracking in Cluttered Environments},
  journal = {IEEE Transactions on Robotics},
  year    = {2026},
  doi     = {10.1109/TRO.2026.3701578}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## Third-party software

Several classical planner implementations were adapted from the MIT-licensed
[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) project.
Attribution, source paths, modification notes, and the upstream license are
preserved in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## License

Project-original code is released under the [MIT License](LICENSE). Third-party
components remain subject to their respective notices in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Planned updates

The learning-based baselines, physical microrobot perception, ROS integration, and low-level current-
control code are planned for the next repository update.
