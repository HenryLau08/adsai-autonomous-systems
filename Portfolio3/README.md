# Portfolio 3 — Multi-Agent Reinforcement Learning (Atari *Warlords*)

Een Multi-Agent Reinforcement Learning (MARL) systeem voor het 4-speler Atari-spel
**Warlords** via PettingZoo. We trainen vier **independent PPO**-policies (één per
hoek) die met en tegen elkaar leren, en vergelijken ze met een random- en een
rule-based baseline.

> **Belangrijk:** de Atari-backend `multi_agent_ale_py` heeft **geen Windows-wheels**.
> Draai dit project op **Google Colab** (aanbevolen) of op **WSL2/Linux**. De code
> die geen Atari nodig heeft (wrapper-logica, baselines) is wél lokaal te testen.

## Aanpak in het kort

- **Algoritme:** PPO (Proximal Policy Optimization) — on-policy, stabiel en robuust
  in een niet-stationaire multi-agent-omgeving.
- **MARL-strategie:** *Independent learners* (IPPO). Elke hoek heeft een eigen
  PPO-beleid; tegenstanders worden per **generatie bevroren** om de
  niet-stationariteit te temmen (verwant aan fictitious self-play).
- **Observatie:** RAM (128 bytes), net als het toernooi. Cruciaal detail: **alle
  hoeken zien dezelfde globale RAM**, wat per-hoek (independent) policies een
  logische keuze maakt — een gedeeld beleid kan de hoeken niet onderscheiden.
- **Reward shaping:** een kleine survival-bonus densificeert de spaarzame ±1
  eindbeloning; de evaluatie gebruikt altijd de pure win/verlies-uitkomst.

## Mappenstructuur

```
Portfolio3/
├── Portfolio3.ipynb          # Rapport + reproduceerbare Colab-pijplijn (hoofddeliverable)
├── generate_notebook.py      # Bouwt Portfolio3.ipynb (reproduceerbaar)
├── README.md
├── requirements.txt
├── warlords_marl/            # Herbruikbare library
│   ├── __init__.py           #   constanten (agenten, acties)
│   ├── baselines.py          #   RandomPolicy, RuleBasedPolicy, PolicyOpponent
│   ├── env_wrapper.py        #   single-agent Gym-wrapper + vec-env fabrieken
│   ├── ram_tools.py          #   RAM-bytes analyseren (bal/paddle vinden)
│   ├── train.py              #   IPPO-training + hoek-robuuste variant
│   └── evaluate.py           #   toernooi, metrieken en plots
├── tournament/               # Toernooi-klare agenten (los inleverbaar)
│   ├── agent1.py             #   random baseline
│   ├── agent2.py             #   rule-based baseline
│   ├── agent3.py             #   PPO-agent (third_0) + herbruikbare PPOAgent
│   ├── agent4.py             #   PPO-agent (fourth_0)
│   ├── models/               #   hier komen de getrainde .zip-modellen
│   └── warlords_tournament_ram_mode.ipynb
├── tests/
│   └── test_env_wrapper.py   # Lokale unit-tests (geen Atari/torch nodig)
└── starter_code_bs/          # Originele starterscode (ongewijzigd)
```

## Installatie

Op Google Colab (aanbevolen):

```python
!pip install "pettingzoo[atari]" "autorom[accept-rom-license]" "stable-baselines3>=2.0.0" imageio imageio-ffmpeg tqdm rich
!AutoROM --accept-license
```

Of lokaal op Linux/WSL2:

```bash
pip install -r requirements.txt
AutoROM --accept-license
```

## Gebruik

### 1. Reproduceer alles via het notebook
Open **`Portfolio3.ipynb`** in Google Colab (kies een GPU-runtime) en draai de
cellen van boven naar beneden. Het notebook installeert de libraries, verkent de
omgeving, meet de baselines, traint de agenten, en produceert de figuren.

### 2. Of gebruik de library direct

```python
from warlords_marl import train, evaluate
from warlords_marl.env_wrapper import make_parallel_env

# Train vier independent PPO-policies (schaal de waarden op voor sterke agenten).
train.train_independent(generations=3, steps_per_agent=200_000, n_envs=8,
                        save_dir="tournament/models")

# Evalueer tegen de random baseline.
from warlords_marl.baselines import RandomPolicy, PolicyOpponent
from stable_baselines3 import PPO
from warlords_marl import AGENT_ORDER
models = {a: PPO.load(f"tournament/models/ppo_{a}.zip", device="cpu") for a in AGENT_ORDER}
policies = {a: PolicyOpponent(models[a]) for a in AGENT_ORDER}
print(evaluate.run_tournament(make_parallel_env, policies, n_games=20)["win_rate"])
```

### 3. Toernooi spelen
Plaats de getrainde `ppo_*.zip`-modellen in `tournament/models/` en draai
`tournament/warlords_tournament_ram_mode.ipynb`. De vier `agentN.py`-bestanden
volgen het `act(observation)`-contract van de toernooi-omgeving. Zonder getraind
model vallen de PPO-agenten veilig terug op willekeurig spel, zodat het notebook
altijd draait.

## Tests

De lastige boekhouding van de single-agent wrapper (actie-dict samenstellen,
beloning/terminatie doorgeven, geëlimineerde agenten afhandelen) is geverifieerd
met een mock van de PettingZoo-API — **zonder Atari, PyTorch of SB3**:

```bash
python tests/test_env_wrapper.py
```

## Reproduceerbaarheid

- Alle willekeur loopt via een vaste `seed`.
- `generate_notebook.py` bouwt het notebook deterministisch op.
- De getrainde modellen en logs (`tournament/models/`, `monitor/`, `snapshots/`,
  video's) worden niet meegecommit; zie `.gitignore`.

## Bronnen

De volledige APA-referentielijst en de verantwoording van Generative AI staan in
`Portfolio3.ipynb` (§11–12). Kernbronnen: Schulman et al. (2017, PPO),
de Witt et al. (2020, IPPO), Terry et al. (2021, PettingZoo) en Raffin et al.
(2021, Stable-Baselines3).
