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

## Self-play for delayed (+1/-1) win/lose rewards

`warlords_v3`'s native reward is sparse and terminal: 0 every step,
then +1 (win) or -1 (lose) once your fortress falls or you're the last
one standing. This needs three things `train_warlords.py` (the
original, dense-reward-style script) doesn't have, all added in
`train_warlords_selfplay.py` + `heppo/opponent_pool.py`:

**1. A longer credit-assignment horizon.** A reward 500 steps in the
past decays to `0.99^500 ≈ 0.007` under the old `gamma=0.99` default —
essentially invisible to GAE for a delayed terminal reward. The new
default `PPOConfig` raises `gamma=0.999, lambda=0.97`, which keeps that
same reward at `0.999^500 ≈ 0.61`. Concretely, on a 500-step episode
with reward only at the final step, the new defaults give the signal
**~2x the reach back through time** (visible above 1% of peak
magnitude ~147 steps back vs ~76 under the old defaults — verified in
testing). No reward shaping is added by default: shaping a competitive
win/lose objective risks the agent optimizing the shaping term instead
of actually winning. An opt-in hook exists if you want to experiment
(`PPOConfig.use_reward_shaping` + `HEPPOAgent.set_reward_shaping_fn`).

**2. A self-play opponent pool (`heppo/opponent_pool.py`).** One agent
slot per env (`first_0`) is the trainee; the other 3 are filled by
frozen snapshots sampled from a pool of the trainee's own past
checkpoints. The pool is **ELO-weighted**: opponents close in skill to
the trainee's current rating are sampled far more often than mismatched
ones (~91% of sampling mass concentrates on same-rating opponents in
testing), because those are the matches with the most learning signal
— blowouts in either direction give a near-constant advantage estimate
and little gradient. This is a lightweight version of the
Prioritized-Fictitious-Self-Play idea used in AlphaStar/OpenAI Five.
The pool evicts the *oldest* checkpoint at capacity (not the weakest),
to keep a spread of past "eras" of the policy rather than collapsing
onto only recent/strong snapshots — helps avoid the policy forgetting
how to beat earlier strategies.

**3. ELO as the primary progress metric, not raw return.** Under
self-play, the opponent distribution keeps changing — beating a string
of weak early-checkpoint opponents and beating your current self both
just look like "won," so raw mean episode return stops tracking
absolute skill. ELO normalizes for who you played, the same way it
does in chess. `train_warlords_selfplay.py` logs `current_elo` and a
rolling win rate every iteration; that's the number to watch, not
return.

**Do the HEPPO standardizations still apply?** Yes, unchanged in
mechanism — keep both:
- *Dynamic reward standardization* (Section II-A) still runs on the
  ±1/0 reward stream. One thing worth knowing: under self-play its
  running mean/std partially absorbs shifts in the *opponent pool's*
  difficulty over time, not just shifts in the trainee's raw skill —
  that's the intended non-stationarity-handling behavior (it's the
  same thing the paper designed it for), but it does mean a
  standardized-reward-derived number isn't a skill metric on its own;
  use ELO for that instead.
- *Advantage standardization* (the final per-batch normalize-to-N(0,1)
  before the policy loss) is orthogonal to self-play — it's a PPO
  numerics stabilizer, unaffected by any of the above. Keep it on
  (`PPOConfig.standardize_advantages=True`, the default).

## Files

```
heppo/
  standardization.py     # dynamic reward std, block value std, quantization
  gae.py                  # GAE computation (vectorized over trajectories)
  network.py              # ActorCritic MLP for 128-byte RAM observations
  buffer.py                # RolloutBuffer (T, N) layout
  ppo_agent.py              # HEPPOAgent: wires standardization+GAE+PPO update
  opponent_pool.py          # ELO tracking + ELO-weighted self-play opponent pool
train_warlords.py           # baseline: all 4 agent slots train together (dense-reward style)
train_warlords_selfplay.py  # self-play: 1 trainee slot vs 3 ELO-sampled frozen opponents
run_ablation.py              # reproduces paper's Table III experiment comparison
```

```bash
pip install "pettingzoo[atari]" autorom torch numpy
AutoROM --accept-license   # downloads Atari ROMs, needs internet access
```

> **Note:** `AutoROM` fetches ROMs from an external CDN. If you're
> running this in a network-restricted environment (e.g. a sandboxed
> container), that download will fail — run it on a machine with open
> internet access, or pass `rom_path=...` to `warlords_v3.parallel_env`
> pointing at ROMs you've installed another way.

## Training

**Self-play (recommended for the real win/lose reward structure):**

```bash
python train_warlords_selfplay.py \
    --num-envs 16 \
    --rollout-len 128 \
    --total-steps 5000000 \
    --snapshot-every 20 \
    --pool-size 20 \
    --elo-k 16 \
    --elo-temperature 200
```

Watch `current_elo` and `win_rate(last200)` in the logs — these are
the progress metrics, not raw return (see "Self-play" section above).
`--snapshot-every 20` adds the trainee to the opponent pool every 20
PPO iterations; tune this against how fast your trainee improves
(too frequent = pool fills with near-duplicate skill levels; too rare
= trainee can lap its own pool and face a curriculum that's stale or
too easy).

**Dense/baseline (all 4 slots train together; simpler, but less
appropriate for Warlords' real reward structure — kept for
comparison/debugging):**

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

Useful flags (both scripts):
- `--no-quantization` — disable HEPPO's quantization step (ablation)
- `--no-adv-standardize` — disable final advantage standardization
- `--quant-bits N` — paper recommends `N>=8` for stable training (Section V-B); 5 and 7 were found to be an unstable middle ground
- `--gamma` / `--lam` — defaults are tuned for delayed terminal reward (see above); the original dense-style script also now defaults to these unless overridden

Checkpoints save to `--checkpoint heppo_warlords.pt` (network weights +
running reward-standardizer state, so you can resume the running
statistics correctly). The self-play script additionally saves
`--pool-checkpoint heppo_warlords_pool.pt` (the full opponent pool with
ELO ratings, so you can resume self-play without losing the curriculum).

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
the environment-stepping code (`WarlordsVecEnv`,
`SelfPlayWarlordsVecEnv`) was tested against a mock of PettingZoo's
parallel API reproducing its exact agent-death / episode-end
bookkeeping behavior — including the asymmetric case where the trainee
dies before the other 3 finish their own game. Everything else —
Welford's running statistics (checked against closed-form mean/std),
block standardization round-trip, quantization error scaling with
bit-width, the GAE recursion (checked against an independent reference
implementation, including episode-boundary masking), the new
gamma/lambda defaults' effect on credit-assignment reach (~2x further
back in time vs the old defaults, measured directly), the full PPO
update, and the self-play pipeline end-to-end (opponent sampling, ELO
updates — verified zero-sum and correctly weighting upsets more than
expected wins, pool eviction, checkpoint snapshotting) — was run on
synthetic/mocked data shaped exactly like a Warlords RAM rollout (128-
byte obs, 6 actions), including a toy bandit-style task confirming the
core PPO pipeline produces a working learning signal (action
preference goes from ~17% to 100% within 20 iterations).

Before running for real, sanity-check the live env on your machine:

```bash
python -c "
from pettingzoo.atari import warlords_v3
env = warlords_v3.parallel_env(obs_type='ram')
obs, _ = env.reset(seed=0)
print(env.agents, {a: o.shape for a, o in obs.items()})
"
```
