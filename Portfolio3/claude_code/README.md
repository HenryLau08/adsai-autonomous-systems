# HEPPO-style PPO for PettingZoo Atari Warlords (RAM)

A software (PyTorch) implementation of the **algorithmic** contributions
of *"HEPPO: Hardware-Efficient Proximal Policy Optimization"*
(Taha & Abdelhadi, arXiv:2501.12703) — dynamic reward standardization,
block value standardization, and uniform quantization ahead of GAE —
wired up to train on `pettingzoo.atari.warlords_v3` with RAM
observations.

**What this does *not* do:** the paper's core contribution is an FPGA
accelerator (a Verilog/SoC design). This code does not synthesize
hardware or run on an FPGA — it reproduces the *algorithm* (Section II
of the paper) in standard PyTorch so you can train an agent and study
the standardization/quantization tricks' effect on learning, which is
also what the paper evaluates in software before describing the
hardware mapping (Section V).

## Paper → code map

| Paper section | Code |
|---|---|
| Eq. 6-9, Welford's running mean/std (Section II-A) | `heppo/standardization.py::DynamicRewardStandardizer` |
| Block standardization of values (Section II-B) | `heppo/standardization.py::block_standardize_values/_destandardize_values` |
| n-bit uniform quantization (Section II-C) | `heppo/standardization.py::uniform_quantize` |
| GAE recursion, Eq. 2-5 | `heppo/gae.py::compute_gae` |
| Algorithm 1 (PPO-Clip) | `heppo/ppo_agent.py::HEPPOAgent.update` |
| Memory layout intuition (Section IV, Algorithm 2) | `heppo/buffer.py::RolloutBuffer` (timestep-major, not literal BRAM) |
| Table III, Experiments 1-5 | `run_ablation.py` |

The k-step lookahead pipelining trick (Section III-B) is a hardware
retiming technique for breaking a feedback loop across FPGA clock
cycles — it computes the *same* recursion as plain GAE, just
re-associated. It isn't reproduced here because it has no effect in
software (no clock-cycle feedback loop to break).

## Setup

```bash
pip install "pettingzoo[atari]" autorom torch numpy
AutoROM --accept-license   # downloads Atari ROMs, needs internet access
```

> **Note:** `AutoROM` fetches ROMs from an external CDN. If you're
> running this in a network-restricted environment (e.g. a sandboxed
> container), that download will fail — run it on a machine with open
> internet access, or pass `rom_path=...` to `warlords_v3.parallel_env`
> pointing at ROMs you've installed another way.

## Files

```
heppo/
  standardization.py   # dynamic reward std, block value std, quantization
  gae.py                # GAE computation (vectorized over trajectories)
  network.py            # ActorCritic MLP for 128-byte RAM observations
  buffer.py              # RolloutBuffer (T, N) layout
  ppo_agent.py            # HEPPOAgent: wires standardization+GAE+PPO update
train_warlords.py         # main training script (warlords_v3, ram, parallel API)
run_ablation.py           # reproduces paper's Table III experiment comparison
```

## Training

```bash
python train_warlords.py \
    --num-envs 8 \
    --rollout-len 128 \
    --total-steps 5000000 \
    --quant-bits 8
```

This runs 8 parallel `warlords_v3` instances (4 agents each → N=32
trajectories), sharing one policy network across all agents (a
standard parameter-sharing simplification since Warlords' 4 players
are symmetric). Observations are the env's 128-byte RAM vector, so a
small MLP is used (no CNN).

Useful flags:
- `--no-quantization` — disable HEPPO's quantization step (ablation)
- `--no-adv-standardize` — disable final advantage standardization
- `--quant-bits N` — paper recommends `N>=8` for stable training (Section V-B); 5 and 7 were found to be an unstable middle ground

Checkpoints save to `--checkpoint heppo_warlords.pt` (network weights +
running reward-standardizer state, so you can resume the running
statistics correctly).

## Ablation (paper's Table III)

```bash
python run_ablation.py --total-steps 1000000 --experiments 1 2 3 4 5
```

Runs the 5 configurations from the paper's Table III back-to-back and
prints a final comparison of mean episode return. Per the paper,
Experiment 5 (dynamic reward standardization, kept standardized
through GAE, + block value standardization with de-standardization) is
expected to perform best, and Experiment 4 (no de-standardization
anywhere) is expected to perform worst.

## Verified without ROMs

Atari ROM downloads aren't reachable from this development sandbox, so
the environment-stepping code (`WarlordsVecEnv`) was tested against a
mock of PettingZoo's parallel API that reproduces its exact
agent-death / episode-end bookkeeping behavior. Everything else —
Welford's running statistics (checked against closed-form mean/std),
block standardization round-trip, quantization error scaling with
bit-width, the GAE recursion (checked against an independent reference
implementation, including episode-boundary masking), and the full
PPO update — was run end-to-end on synthetic data shaped exactly like
a Warlords RAM rollout (128-byte obs, 6 actions, N=32 trajectories),
including a toy bandit-style task confirming the full pipeline
produces a working learning signal (action preference goes from ~17%
to 100% within 20 iterations).

Before running for real, sanity-check the live env on your machine:

```bash
python -c "
from pettingzoo.atari import warlords_v3
env = warlords_v3.parallel_env(obs_type='ram')
obs, _ = env.reset(seed=0)
print(env.agents, {a: o.shape for a, o in obs.items()})
"
```
