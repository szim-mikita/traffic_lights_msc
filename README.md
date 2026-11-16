# State Graph Traffic Lights

A traffic light control environment using SUMO simulation with scalar DQN and multi-objective MORL training.

## New Features

### Fixed Green States & Transition Sequences (TrafficSignal)

The `TrafficSignal` class now supports explicit green phase definitions and sequences of intermediate transition states.

**Constructor parameters:**
- `fixed_green_states: Optional[List[str]]` — List of SUMO traffic light state strings (e.g., `["ggrr", "rggr", "rrgg"]`)
- `transition_sequences: Optional[Dict[Tuple[int, int], List[Tuple[int, Union[str, int]]]]]` — Mapping of `(from_phase, to_phase)` to a list of `(duration_seconds, state_string)` tuples

**Example usage:**

```python
from gym_envs.envs.sumorl_env import SumoRLEnv

env = SumoRLEnv(
    net_file="path/to/net.xml",
    route_file="path/to/routes.xml",
    fixed_green_states={
        "TL1": ["ggrr", "rggr", "rrgg"]
    },
    transition_sequences={
        "TL1": {
            (0, 1): [(3, "ygrr"), (2, "rgrr")],  # From phase 0→1: 3s yellow, then 2s red
            (1, 2): [(4, "ryyr")],                # From phase 1→2: 4s yellow
            # ... more transitions
        }
    }
)
```

**How it works:**
- When `set_next_phase()` is called, the signal transitions through intermediate states with explicit durations
- The `update()` method manages a transition queue, decrementing timers and advancing states
- Falls back to direct state change if no explicit transition sequence is defined
- Existing dynamic yellow-based logic is preserved when no fixed states are provided

## Docker Setup

A containerized environment with SUMO, Python 3, PyTorch, and all dependencies pre-installed.

### Build

```bash
docker build -t traffic-rl .
```

### Run Training

```bash
docker run --rm -it -v "$(pwd)":/app -w /app traffic-rl
```

### Run Evaluation

```bash
docker run --rm -it -v "$(pwd)":/app -w /app -e TRAIN_MODE=eval traffic-rl
```

### Custom Parameters

```bash
docker run --rm -it -v "$(pwd)":/app -w /app \
  -e TRAIN_MODE=eval \
  -e EPISODES=20 \
  -e MAX_STEPS=5000 \
  traffic-rl
```

### Environment Variables

- `TRAIN_MODE` — `train` or `eval` (default: `train`)
- `EPISODES` — number of episodes (default: `10`)
- `MAX_STEPS` — max steps per episode (default: `10000`)

### Override Command Entirely

```bash
docker run --rm -it -v "$(pwd)":/app -w /app traffic-rl \
  python3 train.py --mode eval --episodes 3
```

### Multi-objective training

The image includes SUMO and the MO-Gymnasium/MORL-Baselines dependencies. The bundled SUMO configuration is used by default, so no host SUMO installation or `SUMO_HOME` setup is needed. Keep the existing scalar trainer as the default, or explicitly run PCN from the image:

```bash
docker run --rm -it -v "$(pwd)":/app -w /app traffic-rl \
  python3 train_morl.py \
  --num-seconds 60 \
  --total-timesteps 480 \
  --initial-episodes 4 \
  --step-episodes 1 \
  --model-updates 1 \
  --batch-size 8
```

The model is saved to `weights/PCN_model.pt` and a JSON record of the
configuration and evaluated returns is saved under `runs/`. Actions are
discrete choices of the target traffic-light state. The flattened observation
contains per-induction-loop mean speed (m/s), occupancy (percent), and vehicle
count, followed by the current state and time in state.
Training progress counts each environment action. It updates in place in an
interactive terminal and prints regular progress lines when output is captured;
the PCN rollout summaries remain separate.

Choose any non-empty ordered subset of these reward objectives with `--reward-components`:

| Name | Objective |
|---|---|
| `co2` | Minimize CO₂ emissions |
| `throughput` | Maximize vehicles detected |
| `average_speed` | Maximize detector vehicle speed |
| `waiting_time` | Minimize average waiting time for vehicles on detector lanes |
| `queue_length` | Minimize vehicles moving below 0.1 m/s on detector lanes |
| `time_in_state_penalty` | Penalize remaining in one state, with no cost through 45 s then capped exponential growth |

To train against the catalogue SUMO assets without a `.sumocfg`, select the
catalogue environment. It takes a network file, traffic-light JSON catalogue,
and detector additional file, and generates randomized lane-level trips by
default. The checked-in catalogue assets and `R3C2_C` are the defaults; override
the files or junction ID when using another catalogue:

```bash
python3 train_morl.py \
  --environment catalogue \
  --traffic-light-id R3C2_C \
  --model-name PCN_catalogue_run \
  --num-seconds 3600 \
  --reward-components co2 throughput average_speed
```

Use `--net-file`, `--logic-json-file`, and `--detector-file` to supply another
catalogue SUMO network. Set `--no-random-traffic --route-file path/to/routes.rou.xml`
to use a fixed route file instead. The catalogue env preserves a per-junction
`TrafficAgent` structure while exposing one selected junction as the discrete
single-agent action interface required by PCN.
Use `--render-mode gui` to open and automatically start SUMO-GUI, and
`--model-name` to name the saved model and results. Use `--results-dir` to
choose another directory for the JSON results record. Actions are capped at
the configured `--num-seconds` horizon, including when a phase transition
would otherwise extend the final action interval.

For example, train against throughput, speed, and queue length:

```bash
docker run --rm -it -v "$(pwd)":/app -w /app traffic-rl \
  python3 train_morl.py \
  --reward-components throughput average_speed queue_length \
  --reward-normalization normalized_average \
  --num-seconds 60 \
  --total-timesteps 480 \
  --initial-episodes 4 \
  --step-episodes 1 \
  --model-updates 1 \
  --batch-size 8
```

Each objective is measured per SUMO second across the action interval. `--reward-normalization sum` accumulates over elapsed seconds, `average` returns the time-average, and `normalized_average` returns the time-average divided by a per-objective scale. The defaults are respectively 10,000 mg CO₂, 1 vehicle, 13.89 m/s, 60 s waiting time, 10 queued vehicles, and 1 for the state penalty. Override them in objective order with `--reward-scales`. Tune the post-45-second penalty using `--state-penalty-threshold` and `--state-penalty-growth`.

`--scaling-factor` takes one value per selected objective plus one for horizon; `--max-return` and `--ref-point` take one per selected objective. If omitted, the trainer derives dimension-matched defaults. The usual training limits (`--total-timesteps`, `--num-seconds`, and `--traffic-scale`) can also be adjusted. Use `--sumo-config` to select another SUMO configuration file.

Plot one or more saved MORL evaluation result files without rerunning SUMO:

```bash
docker run --rm -v "$(pwd)":/app -w /app traffic-rl-morl:latest \
  python3 plot_morl.py \
  runs/PCN_catalogue_co2_queue_seed42_20261115_results.json
```

This saves objective-pair return plots and per-objective distribution plots
under `plots/`. Pass multiple results JSON files to compare runs that use the
same ordered objectives, or use `--output-dir` and `--prefix` to select output
paths. A non-dominated subset of evaluated returns is highlighted in the
objective-pair plots.

Compare a trained PCN checkpoint against the catalogue network's native
fixed-time SUMO program on the same generated-traffic seeds:

```bash
docker run --rm -v "$(pwd)":/app -w /app traffic-rl-morl:latest \
  python3 compare_morl.py \
  runs/PCN_catalogue_co2_queue_statepenalty_900s_seed42_20261115_pilot_results.json \
  --episodes 20 \
  --seed 100 \
  --num-seconds 900 \
  --desired-horizon 90 \
  --prefix PCN_statepenalty_vs_native_fixed_time_900s

docker run --rm -v "$(pwd)":/app -w /app traffic-rl-morl:latest \
  python3 plot_morl.py \
  runs/PCN_statepenalty_vs_native_fixed_time_900s_pcn_results.json \
  runs/PCN_statepenalty_vs_native_fixed_time_900s_native_fixed_time_results.json \
  --prefix PCN_statepenalty_vs_native_fixed_time_900s \
  --state-trace runs/PCN_statepenalty_vs_native_fixed_time_900s_state_trace.json
```

The comparison records paired seeds and returns in `runs/`. PCN is evaluated
with one fixed desired-return vector (the mean of the saved replay returns)
and a fixed desired horizon; it does not sweep explicit objective weights.
If PCN includes `time_in_state_penalty`, both policies' plots and primary
comparison CSV use only shared objectives (such as CO₂ and queue length).
PCN's full returns and its additional penalty return are retained in the PCN
results JSON and CSV; the fixed-time policy does not receive a fabricated
state-time penalty score.
The native baseline resumes the network's SUMO program (program `0` by
default), rather than controlling the light with the catalogue actions.
Returns remain negative for cost objectives such as CO₂ and queue length;
higher/closer to zero is better.
The comparison also saves a per-second signal-state trace for one matched seed
(the first evaluated seed by default; override with `--trace-seed`). Pass it to
`plot_morl.py --state-trace` to produce side-by-side PCN and native-plan state
trajectories. Signal states are categorical, so trajectories are shown for one
seed rather than averaged across episodes. The chart labels each state as
main, transition, or other; PCN roles come from the traffic-light catalogue,
and fixed-time roles come from the SUMO program phases. Use a longer
`--num-seconds` (for example 180 seconds) for signal-state plots when you want
to see more than one 90-second fixed-time cycle.

## Local Development

Install dependencies:

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Ensure `SUMO_HOME` is set:

```bash
export SUMO_HOME=/usr/share/sumo
```

Run training:

```bash
python3 train.py --mode train --episodes 10
```

Run evaluation:

```bash
python3 train.py --mode eval --episodes 5
```

## Files

- `gym_envs/envs/traffic_signal.py` — Traffic signal with fixed-state & transition support
- `gym_envs/envs/sumorl_env.py` — Main environment (now accepts fixed states config)
- `gym_envs/envs/state_traffic_env.py` — Scalar and multi-objective state-based environments
- `gym_envs/envs/catalogue_morl_env.py` — Catalogue-file SUMO environment for multi-objective training
- `train.py` — DQN training/evaluation script
- `train_morl.py` — MORL-Baselines PCN trainer for the multi-objective environment
- `Dockerfile` — Container definition
- `requirements.txt` — Python dependencies
- `docker-entrypoint.sh` — Container entry logic with env var support
