# MORL Traffic-Light Control

Multi-objective traffic-light control with SUMO and Pareto-Conditioned Networks
(PCN) from MORL-Baselines.

## Setup

Install SUMO and its tools, then install the Python dependencies:

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
export SUMO_HOME=/usr/share/sumo
```

The catalogue MORL environment uses the three included SUMO assets and
generates randomized traffic by default.

For agent-controlled phase changes, the environment enforces a five-second
minimum main-green interval. It holds the initial green for that minimum before
the first switch, applies the catalogue's ordered intergreen sequence, then
holds the selected main green for at least five seconds before the next
decision. This is simulation behavior, not a real-world signal-safety
certification.

## Train

Train against the catalogue:

```bash
python3 train_morl.py \
  --traffic-light-id R3C2_C \
  --model-name PCN_catalogue_run \
  --num-seconds 3600 \
  --reward-components co2 throughput average_speed
```

Choose any non-empty ordered subset of these reward objectives with
`--reward-components`:

| Name | Objective |
|---|---|
| `co2` | Minimize CO₂ emissions |
| `throughput` | Maximize vehicles detected |
| `average_speed` | Maximize detector vehicle speed |
| `waiting_time` | Minimize average waiting time on detector lanes |
| `queue_length` | Minimize vehicles moving below 0.1 m/s on detector lanes |
| `time_in_state_penalty` | Penalize staying in one state beyond the configured threshold |

Use `--net-file`, `--logic-json-file`, and `--detector-file` to select other
catalogue assets. Set `--no-random-traffic --route-file path/to/routes.rou.xml`
to use fixed routes.

Training saves the model under `weights/` and its configuration and evaluation
returns under `runs/`. Both directories are local outputs and are not part of
the source tree.

## Evaluate and plot

Compare a trained catalogue policy against the network's native fixed-time
program using matched traffic seeds:

```bash
python3 compare_morl.py runs/PCN_catalogue_run_results.json \
  --episodes 20 \
  --seed 100 \
  --num-seconds 900
```

Plot saved MORL results without rerunning SUMO:

```bash
python3 plot_morl.py runs/PCN_catalogue_run_results.json
```

Plots are written to `plots/`. Pass multiple result files to compare runs that
use the same ordered objectives. `plot_morl.py --state-trace` can plot the
per-second signal-state trace produced by `compare_morl.py`.

## Docker

Build an image with SUMO and the Python dependencies:

```bash
docker build -t traffic-rl-morl .
```

Run the default MORL training command:

```bash
docker run --rm -it -v "$(pwd)":/app -w /app traffic-rl-morl
```

To pass training options, invoke the trainer explicitly:

```bash
docker run --rm -it -v "$(pwd)":/app -w /app traffic-rl-morl \
  python3 train_morl.py --num-seconds 3600
```

## Project files

- `train_morl.py` — PCN training entry point
- `compare_morl.py` — matched-seed policy evaluation
- `plot_morl.py` — MORL results and state-trace plots
- `gym_envs/envs/morl_env.py` — catalogue-file MORL environment
- `gym_envs/envs/traffic_lights_parser.py` — catalogue parsing and validation
- `tests/` — MORL environment, trainer, comparison, and plotting tests
