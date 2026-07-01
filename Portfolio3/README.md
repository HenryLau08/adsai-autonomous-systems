# Portfolio 3 - Multi-Agent Reinforcement Learning (Atari *Warlords*)

Een Multi-Agent Reinforcement Learning (MARL) systeem voor het 4-speler
Atari-spel *Warlords* via PettingZoo. We trainen:

1. **Vier independent PPO-policies** (IPPO), één per hoek, met generatie-gewijs
   bevroren tegenstanders.
2. **Een hoek-robuust league-beleid** in 4 fases (random -> rule-based -> 1
   snapshot -> 2 snapshots) als alternatieve inzending.

De prestaties zijn afgezet tegen random- en rule-based baselines en tegen het
zelf-getrainde model van een teamgenoot (zie §9.10 in het notebook).

> De Atari-backend `multi_agent_ale_py` heeft geen Windows-wheels. Draai dit
> project op Linux/WSL2 (aanbevolen, met NVIDIA GPU) of op Google Colab.

## Aanpak in het kort

- Algoritme: PPO (Proximal Policy Optimization) - on-policy, stabiel in een
  niet-stationaire multi-agent-omgeving (Schulman et al., 2017).
- MARL-strategie: independent learners (IPPO) per hoek + league-curriculum voor
  een hoek-robuust alternatief.
- Observatie: 128-byte RAM, identiek aan het toernooi. Alle hoeken zien dezelfde
  globale RAM zonder hoek-indicator - dit maakt per-hoek independent learners de
  natuurlijke keuze (zie notebook §1.3).
- Reward shaping: kleine survival-bonus om het spaarzame +-1 eindbeloning te
  densificeren. Evaluatie gebruikt altijd pure win/verlies-uitkomst.

## Mappenstructuur

```
Portfolio3/
├── Portfolio3.ipynb          # Rapport + pijplijn 
├── README.md
├── requirements.txt
├── warlords_marl/            # Herbruikbare library
│   ├── __init__.py           #   constanten (agenten, acties)
│   ├── baselines.py          #   RandomPolicy, RuleBasedPolicy, PolicyOpponent
│   ├── env_wrapper.py        #   single-agent Gym-wrapper + vec-env fabrieken
│   ├── ram_tools.py          #   RAM-bytes analyseren
│   ├── train.py              #   IPPO + hoek-robuuste training
│   └── evaluate.py           #   toernooi, metrieken, plots
├── tournament/               # Toernooi-klare agenten
│   ├── agent1.py             #   random baseline
│   ├── agent2.py             #   rule-based baseline
│   ├── agent3.py             #   PPO-agent (third_0) + herbruikbare PPOAgent
│   ├── agent4.py             #   PPO-agent (fourth_0)
│   ├── warlords_tournament_ram_mode.ipynb
│   └── models/               #   ppo_first_0.zip ... ppo_fourth_0.zip + ppo_corner_robust.zip
├── tests/
│   └── test_env_wrapper.py   # Lokale unit-tests (geen Atari/torch nodig)
├── attachments/              # Gegenereerde figuren (worden door notebook geschreven)
└── starter_code_bs/          # Originele starterscode (ongewijzigd)
```

## Installatie

Lokaal op Linux/WSL2 (aanbevolen):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
AutoROM --accept-license
```

Op Google Colab:

```python
!pip install "pettingzoo[atari]" "autorom[accept-rom-license]" "stable-baselines3>=2.0.0" imageio imageio-ffmpeg tqdm rich
!AutoROM --accept-license
```

## Gebruik

### 1. Reproduceer alles via het notebook

Open `Portfolio3.ipynb` en draai de cellen van boven naar beneden. De
`BUDGET`-knop in §2.3 schaalt alle trainingen mee:

- `BUDGET = "fast"` -> ~30 min (demo)
- `BUDGET = "medium"` -> ~2 uur
- `BUDGET = "long"` -> ~5-8 uur (toernooi-niveau)

### 2. Of gebruik de library direct

```python
from warlords_marl import train, evaluate, AGENT_ORDER
from warlords_marl.env_wrapper import make_parallel_env
from warlords_marl.baselines import PolicyOpponent
from stable_baselines3 import PPO

# Train independent PPO per hoek
train.train_independent(generations=3, steps_per_agent=200_000, n_envs=8,
                        save_dir="tournament/models")

# Evalueer
models = {a: PPO.load(f"tournament/models/ppo_{a}.zip", device="cpu") for a in AGENT_ORDER}
policies = {a: PolicyOpponent(models[a]) for a in AGENT_ORDER}
print(evaluate.run_tournament(make_parallel_env, policies, n_games=20)["win_rate"])
```

### 3. Toernooi spelen

In `tournament/models/` horen vijf zip-bestanden:

- `ppo_first_0.zip`, `ppo_second_0.zip`, `ppo_third_0.zip`, `ppo_fourth_0.zip`
  - IPPO per-hoek specialisten
- `ppo_corner_robust.zip` - hoek-onafhankelijke fallback

`agent3.py` en `agent4.py` zoeken eerst het hoek-specifieke model en vallen
terug op `ppo_corner_robust.zip`. Zonder enig model spelen ze willekeurig, zodat
`warlords_tournament_ram_mode.ipynb` altijd draait.


## Reproduceerbaarheid

- Alle willekeur loopt via een vaste `SEED`.
- Trainings-output (`experiments/`, `tournament/models/league/`,
  `monitor/`, `snapshots/`, video's) worden niet meegecommit; zie `.gitignore`.
- De vijf finale `ppo_*.zip` modellen wel.

## Bronnen

De volledige APA-referentielijst staat in `Portfolio3.ipynb` (§12).
Kernbronnen: Schulman et al. (2017, PPO), de Witt et al. (2020, IPPO),
Heinrich & Silver (2016, fictitious self-play), Vinyals et al. (2019, AlphaStar
league-training), Terry et al. (2021, PettingZoo) en Raffin et al. (2021,
Stable-Baselines3).
