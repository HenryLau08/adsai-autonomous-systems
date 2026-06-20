"""Genereert Portfolio3.ipynb (rapport + reproduceerbare Colab-pijplijn).

Het notebook wordt programmatisch opgebouwd zodat de inhoud in versiebeheer
leesbaar blijft en eenvoudig te regenereren is:

    python generate_notebook.py

De code-cellen roepen het ``warlords_marl``-pakket aan (zie de map ernaast) en
zijn ontworpen om op Google Colab te draaien.
"""

from __future__ import annotations

import json


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str):
    text = text.strip("\n")
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


CELLS = [
    # ---------------------------------------------------------------- titel
    md(
        r"""
# Portfolio 3 — Multi-Agent Reinforcement Learning (Atari *Warlords*)

**Vak:** Autonomous Systems · De Haagse Hogeschool

Dit notebook is zowel het **rapport** als de **reproduceerbare pijplijn**. We
trainen een Deep RL-agent die in de multi-agent Atari-omgeving *Warlords*
concurreert met drie andere agenten, en vergelijken die met baselines.

> **Uitvoeren:** dit notebook draait op **Google Colab** (Linux + GPU). De
> Atari-backend `multi_agent_ale_py` heeft geen Windows-wheels; op Windows werkt
> alleen WSL2 of Colab. Kies in Colab *Runtime → Change runtime type → GPU*.
"""
    ),
    # ---------------------------------------------------------------- 1. intro
    md(
        r"""
## 1. Inleiding & Probleemanalyse

### 1.1 De omgeving: *Warlords*
*Warlords* is een Atari-spel met **vier spelers** die elk een kasteel in een hoek
verdedigen met een paddle. Een bal stuitert rond; wordt je kasteel geraakt, dan
lig je eruit. De laatste speler die overblijft wint. In de PettingZoo-implementatie
(`warlords_v3`) betekent dit:

| Eigenschap | Waarde |
|---|---|
| Agenten | `first_0`, `second_0`, `third_0`, `fourth_0` (4) |
| Actieruimte | `Discrete(6)`: 0=noop, 1=fire, 2=up, 3=right, 4=left, 5=down |
| Observatie (`obs_type="ram"`) | 128 bytes console-RAM (`uint8`, 0–255) |
| Beloning | **spaarzaam/terminaal**: −1 als je kasteel valt, +1 als je als laatste overblijft |

Dit is een **competitieve, gemengde multi-agent-omgeving**: de agenten delen één
omgeving maar hebben tegengestelde belangen.

### 1.2 Waarom dit een multi-agent-probleem is
Vanuit het perspectief van één agent zijn de andere drie agenten onderdeel van de
omgeving. Omdat die anderen óók leren en hun gedrag veranderen, is de omgeving
**niet-stationair**: de optimale strategie verschuift terwijl tegenstanders beter
worden (Busoniu et al., 2008; Lowe et al., 2017). Een goede aanpak moet hier
expliciet rekening mee houden.

### 1.3 Een cruciale observatie over de RAM
In `warlords_v3` krijgen **alle vier de agenten dezelfde globale 128-byte RAM**
(zie `base_atari_env.py`: `{agent: obs for agent in self.agents}`). Er is dus geen
hoek-specifieke observatie en geen indicator van *welke* hoek je bent. Een enkel
gedeeld beleid kan de hoeken daardoor niet uit elkaar houden. Dit motiveert direct
onze keuze voor **independent learners**: vier afzonderlijke policies, één per hoek,
die elk hun eigen hoek leren verdedigen.

### 1.4 Keuze van algoritme en trainingsstrategie
- **Algoritme: PPO** (Proximal Policy Optimization; Schulman et al., 2017). PPO is
  een on-policy actor-critic methode die door zijn *clipped* objective stabiel en
  robuust traint met weinig hyperparameter-tuning. Dat is een groot voordeel in een
  niet-stationaire multi-agent-setting, waar waarde-gebaseerde methoden (zoals DQN)
  gevoeliger zijn voor instabiliteit (de Witt et al., 2020).
- **Trainingsstrategie: Independent PPO (IPPO) met generatie-gewijs bevroren
  tegenstanders.** Elke hoek heeft een eigen PPO-beleid. We temmen de
  niet-stationariteit door de tegenstanders per *generatie* te bevriezen: elk
  beleid traint tegen vaste snapshots van de andere drie, waarna alle snapshots
  geüpdatet worden. Dit is verwant aan *fictitious self-play* (Heinrich & Silver,
  2016) en aan IPPO (de Witt et al., 2020), dat verrassend sterk presteert in
  competitieve benchmarks.
- **Observatie: RAM.** Het toernooi gebruikt `obs_type="ram"`, en een kleine MLP op
  128 bytes traint veel sneller dan een CNN op pixels — ideaal voor de vele
  benodigde environment-stappen.

### 1.5 Baseline
Als referentie gebruiken we twee baselines (opdracht 2a): een **random policy** en
een **rule-based policy** die actief de eigen hoek patrouilleert en periodiek vuurt.
"""
    ),
    # ---------------------------------------------------------------- 2. setup
    md(
        r"""
## 2. Setup (Colab)

Draai de cellen hieronder **één keer van boven naar beneden**. Ze installeren de
libraries, plaatsen de Atari-ROMs (de bekende valkuil van PettingZoo-Atari), halen
de projectcode op en doen de imports. Zet eerst *Runtime → Change runtime type → GPU*.

> Krijg je na de installatie tóch een import-fout? Kies dan *Runtime → Restart
> session* en draai de setup-cellen opnieuw. Dat is soms eenmalig nodig wanneer
> Colab een al ingeladen pakket (zoals NumPy) vervangt.
"""
    ),
    code(
        r"""
# 2.1 Installeer de benodigde libraries.
#     (PyTorch zit al voorgeïnstalleerd op Colab; SB3 gebruikt die versie.)
!pip -q install "pettingzoo[atari]>=1.24" "autorom[accept-rom-license]" "stable-baselines3>=2.0.0" imageio imageio-ffmpeg tqdm rich pandas
print("Libraries geïnstalleerd.")
"""
    ),
    code(
        r"""
# 2.2 Installeer de Atari-ROMs op de plek waar multi_agent_ale_py ze zoekt.
#     Dit is dé valkuil van PettingZoo-Atari. AutoROM accepteert de licentie en
#     installeert de ROM-bestanden; het vangnet kopieert warlords.bin zo nodig naar
#     elke map die multi_agent_ale_py controleert.
import os, glob, shutil

# 1) Accepteer de licentie en installeer de ROMs (standaardlocatie).
!AutoROM --accept-license

import multi_agent_ale_py
ma_dir = os.path.dirname(multi_agent_ale_py.__file__)

# 2) Installeer de ROMs óók rechtstreeks in de map van multi_agent_ale_py.
!AutoROM --accept-license --install-dir "$ma_dir"


def _ensure_warlords_rom():
    targets = [
        os.path.join(ma_dir, "warlords.bin"),
        os.path.join(ma_dir, "roms", "warlords.bin"),
    ]
    if any(os.path.exists(t) for t in targets):
        return True
    # Zoek een warlords-ROM ergens tussen de geïnstalleerde packages.
    search_root = os.path.dirname(ma_dir)  # .../site-packages
    found = glob.glob(os.path.join(search_root, "**", "[Ww]arlords.bin"), recursive=True)
    if not found:
        return False
    for t in targets:
        os.makedirs(os.path.dirname(t), exist_ok=True)
        shutil.copy(found[0], t)
    return True


print("Warlords-ROM aanwezig:", _ensure_warlords_rom())

# Verifieer meteen dat de omgeving laadt (faalt anders pas in §3).
from pettingzoo.atari import warlords_v3

_env = warlords_v3.parallel_env(obs_type="ram")
_env.reset(seed=0)
print("OK: Warlords-omgeving geladen. Agenten:", _env.agents)
_env.close()
"""
    ),
    code(
        r"""
# 2.3 Haal de projectcode op en zet het werkpad op Portfolio3/.
#     Let op: de Portfolio 3-code staat op de branch 'portfolio3_michal', dus we
#     klonen expliciet die branch (-b). Werkt het clonen niet? Upload dan de map
#     Portfolio3 handmatig naar Colab (mapicoon links) en draai deze cel opnieuw.
import os, sys

if not os.path.isdir("adsai-autonomous-systems"):
    !git clone -q -b portfolio3_michal https://github.com/HenryLau08/adsai-autonomous-systems.git

# Zoek de map die het pakket 'warlords_marl' bevat (na clone of na upload).
for cand in ["adsai-autonomous-systems/Portfolio3", "Portfolio3", "."]:
    if os.path.isdir(os.path.join(cand, "warlords_marl")):
        os.chdir(cand)
        break
sys.path.insert(0, os.getcwd())

assert os.path.isdir("warlords_marl"), (
    "Map 'warlords_marl' niet gevonden. Push je code eerst naar de branch "
    "'portfolio3_michal', of upload de map Portfolio3 handmatig naar Colab."
)
print("Werkmap:", os.getcwd())
"""
    ),
    code(
        r"""
# 2.4 Imports en reproduceerbaarheid.
import numpy as np
import matplotlib.pyplot as plt

from warlords_marl import AGENT_ORDER, ACTION_MEANINGS
from warlords_marl.env_wrapper import make_parallel_env
from warlords_marl.baselines import RandomPolicy, RuleBasedPolicy, PolicyOpponent
from warlords_marl import ram_tools, train, evaluate

SEED = 0
np.random.seed(SEED)
print("Agenten:", AGENT_ORDER)
print("Acties:", ACTION_MEANINGS)
"""
    ),
    md(
        r"""
## 2.5 Google Drive — modellen bewaren (sterk aangeraden)

Gratis Colab verbreekt de sessie na verloop van tijd en **wist dan alles**, ook je
getrainde modellen. Door Google Drive te koppelen schrijven we de modellen naar je
Drive, zodat ze een herstart overleven. Loopt de sessie eruit? Dan laad je de
modellen later gewoon terug (§7.0) in plaats van opnieuw te trainen.

Optioneel: zonder Drive werkt alles ook, maar dan ben je je modellen kwijt bij een
disconnect.
"""
    ),
    code(
        r"""
# 2.5 Koppel Google Drive (optioneel maar aangeraden) + hulpfuncties.
import os, glob, shutil

DRIVE_DIR = None
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = "/content/drive/MyDrive/portfolio3_models"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    print("Drive gekoppeld. Modellen worden bewaard in:", DRIVE_DIR)
except Exception as exc:
    print("Geen Drive gekoppeld (lokaal of geweigerd):", exc)


def backup_to_drive():
    "Kopieer getrainde modellen naar Drive (no-op zonder Drive)."
    if not DRIVE_DIR:
        return
    for path in glob.glob("tournament/models/ppo_*.zip"):
        shutil.copy(path, DRIVE_DIR)
    print("Modellen geback-upt naar Drive.")


def restore_from_drive():
    "Zet eerder getrainde modellen terug van Drive naar tournament/models/."
    if not DRIVE_DIR:
        print("Geen Drive gekoppeld.")
        return
    os.makedirs("tournament/models", exist_ok=True)
    files = glob.glob(os.path.join(DRIVE_DIR, "ppo_*.zip"))
    for path in files:
        shutil.copy(path, "tournament/models")
    print(f"{len(files)} model(len) teruggezet van Drive.")
"""
    ),
    # ---------------------------------------------------------------- 3. verken
    md(
        r"""
## 3. De omgeving verkennen

We controleren de actie- en observatieruimte en bekijken welke RAM-bytes het meest
veranderen — die zijn kandidaat voor bal- en paddleposities (de starterscode bevat
hiervoor de basis; `ram_tools` aggregeert dit over een hele episode).
"""
    ),
    code(
        r"""
# 3.1 Actie- en observatieruimte controleren.
env = make_parallel_env(obs_type="ram")
obs, _ = env.reset(seed=SEED)
a0 = env.agents[0]
print("Aantal agenten:", len(env.agents))
print("Actieruimte:", env.action_space(a0))
print("Observatieruimte:", env.observation_space(a0))
print("RAM shape:", np.asarray(obs[a0]).shape, "dtype:", np.asarray(obs[a0]).dtype)
print("Alle agenten zien dezelfde RAM:",
      all(np.array_equal(obs[a0], obs[a]) for a in env.agents))
env.close()
"""
    ),
    code(
        r"""
# 3.2 Welke RAM-bytes veranderen het meest? (kandidaten voor bal/paddle)
env = make_parallel_env(obs_type="ram")
policies = {a: RandomPolicy(seed=SEED) for a in AGENT_ORDER}
trace = ram_tools.collect_ram_trace(env, policies, max_steps=1500, seed=SEED)
print("Trace shape:", trace.shape)
print("\nTop-12 meest veranderende bytes (index, #unieke waarden, std):")
for idx, uniq, std in ram_tools.rank_changing_bytes(trace, top_k=12):
    print(f"  byte {idx:3d}: {uniq:3d} unieke waarden, std={std:5.1f}")
# Vul deze indexen later in bij RuleBasedPolicy(ball_byte=..., paddle_byte=...)
"""
    ),
    # ---------------------------------------------------------------- 4. baselines
    md(
        r"""
## 4. Baselines

We meten eerst hoe random- en rule-based policies presteren. Dit is het
referentiepunt waartegen we het RL-beleid afzetten (opdracht 3b).
"""
    ),
    code(
        r"""
# 4.1 Baseline-toernooi: 4x random vs. 4x rule-based.
#     run_tournament gebruikt standaard frame-skip (=4), een max_steps-limiet en
#     toont voortgang. Daardoor zijn deze toernooien snel en blijft de Colab-sessie
#     'levend' (geen lange, stille cel die tot een disconnect leidt).
random_policies = {a: RandomPolicy(seed=SEED) for a in AGENT_ORDER}
rule_policies = {a: RuleBasedPolicy(corner=a) for a in AGENT_ORDER}

print("== 4x random ==")
summary_random = evaluate.run_tournament(make_parallel_env, random_policies, n_games=10)
print("win-rate:", summary_random["win_rate"], "\n")

print("== 4x rule-based ==")
summary_rule = evaluate.run_tournament(make_parallel_env, rule_policies, n_games=10)
print("win-rate:", summary_rule["win_rate"])
"""
    ),
    # ---------------------------------------------------------------- 5. methode
    md(
        r"""
## 5. Methode: van multi-agent naar trainbaar single-agent

Stable-Baselines3 traint één beleid in een standaard Gymnasium-omgeving. Onze
`WarlordsSingleAgentEnv` (in `warlords_marl/env_wrapper.py`) overbrugt dit: SB3
bestuurt precies één hoek, terwijl de andere drie hoeken door *opponent*-policies
worden gespeeld. Belangrijke ontwerpkeuzes:

- **Observatie-normalisatie:** ruwe RAM-bytes (0–255) worden geschaald naar
  `[0, 1]`, wat het leren met een MLP stabiliseert.
- **Frame-skip (action repeat ×4):** elke gekozen actie wordt **4 emulator-frames**
  herhaald in plaats van elke frame opnieuw te beslissen. Dit is een standaard-truc
  voor Atari (Mnih et al., 2015) met drie voordelen: (1) de agent beslist op een
  *grovere tijdschaal*, wat het leerprobleem makkelijker maakt; (2) het aantal (dure)
  policy-evaluaties en Python-iteraties daalt met ~4×, wat training én evaluatie
  versnelt; (3) één agent-stap dekt 4 frames, dus je krijgt veel meer complete
  episodes per stap. We gebruiken frame-skip **consistent** in de training
  (`WarlordsSingleAgentEnv`), in de evaluatie (`play_match`) én in de toernooi-agenten
  (die hun actie 4 frames vasthouden), zodat het gedrag overal hetzelfde is. Let op:
  omdat één stap nu 4 frames telt, verandert de *schaal* van de survival-beloning
  (lager getal, maar relatief gezien dezelfde trend).
- **Reward shaping:** de ruwe beloning is spaarzaam (alleen ±1 op het eind). We
  voegen een kleine `survival_bonus` per overleefde stap toe, zodat "langer
  overleven" beloond wordt — dat correleert sterk met winnen en versnelt het leren
  (vgl. reward shaping, Ng et al., 1999). De *evaluatie* gebruikt altijd de pure
  win/verlies-uitkomst, niet de shaping.
- **Independent learners:** elke hoek krijgt een eigen PPO-model; tegenstanders
  worden per generatie bevroren (`train.train_independent`).
- **Robuust op gratis Colab:** toernooien hebben een `max_steps`-limiet en tonen
  voortgang (geen lange, stille cellen → minder kans op een time-out), en de modellen
  worden naar Google Drive geback-upt (§2.5) zodat een disconnect geen werk kost.

De boekhouding van deze wrapper (actie-dict, doorgeven van beloning/terminatie,
frame-skip-accumulatie, afhandelen van geëlimineerde agenten) is lokaal geverifieerd
met unit-tests (`tests/test_env_wrapper.py`).
"""
    ),
    # ---------------------------------------------------------------- 6. training
    md(
        r"""
## 6. Training (IPPO)

> **Tijd vs. kwaliteit.** Hieronder staan bescheiden waarden voor een snelle, veilige
> demonstratie op gratis Colab. Voor sterkere agenten verhoog je `steps_per_agent` en
> `generations` — maar doe dat in stappen en gebruik de Drive-backup (§2.5), zodat een
> disconnect je werk niet wist. Met `frame_skip=4` dekt elke stap 4 frames, dus je
> krijgt veel episodes per stap. Op een Colab-GPU met RAM-observaties is de emulator
> (CPU) de bottleneck, dus meer `n_envs` helpt meer dan een zwaardere GPU.
"""
    ),
    code(
        r"""
# 6.1 Train vier independent PPO-policies (één per hoek).
#     Demo-instelling; schaal op voor betere resultaten.
models = train.train_independent(
    generations=2,
    steps_per_agent=150_000,
    n_envs=8,
    survival_bonus=0.01,
    frame_skip=4,
    seed=SEED,
    save_dir="tournament/models",   # Agent3/Agent4 lezen hieruit
    device="auto",
    progress_bar=True,
)
backup_to_drive()   # bewaar direct op Drive (overleeft een disconnect)
print("Getrainde hoeken:", list(models.keys()))
"""
    ),
    # ---------------------------------------------------------------- 7. resultaten
    md(
        r"""
## 7. Resultaten & Discussie

### 7.0 Modellen terugzetten na een disconnect (optioneel)
Liep je Colab-sessie eruit ná het trainen? Draai dan de setup-cellen (§2, incl.
§2.5 Drive) opnieuw en daarna de cel hieronder om de modellen van Drive terug te
halen — zo hoef je niet opnieuw te trainen.
"""
    ),
    code(
        r"""
# 7.0 Optioneel: haal eerder getrainde modellen terug van Drive.
restore_from_drive()
have = sorted(os.listdir("tournament/models")) if os.path.isdir("tournament/models") else []
print("Aanwezige modellen:", [f for f in have if f.startswith("ppo_")] or "geen")
"""
    ),
    md(
        r"""
### 7.1 Leercurves
De trainingsbeloning per hoek (geglad over episodes), samengevoegd over generaties.
Een stijgende curve duidt op leren; let ook op de **stabiliteit** (ruis/variantie).
Het smoothing-window schaalt mee met het aantal episodes, dus ook een korte run
oogt niet onnodig rommelig.
"""
    ),
    code(
        r"""
# 7.1 Leercurves uit de monitor-logs.
fig = evaluate.plot_learning_curves(
    monitor_root="tournament/models/monitor",
    save_path="attachments/leercurves.png",
)
plt.show()
"""
    ),
    md(
        r"""
### 7.2 RL vs. baseline
We laten de getrainde policies een toernooi spelen en vergelijken het
win-percentage met de baselines. Ter controle laten we ook één PPO-hoek tegen drie
random-tegenstanders spelen: een eerlijke "leert RL daadwerkelijk iets?"-test.
"""
    ),
    code(
        r"""
# 7.2 Bouw policies uit de getrainde modellen en evalueer.
from stable_baselines3 import PPO

ppo_models = {a: PPO.load(f"tournament/models/ppo_{a}.zip", device="cpu")
              for a in AGENT_ORDER}
ppo_policies = {a: PolicyOpponent(ppo_models[a], deterministic=True)
                for a in AGENT_ORDER}

# 4x PPO (independent learners tegen elkaar)
print("== 4x PPO ==")
summary_ppo = evaluate.run_tournament(make_parallel_env, ppo_policies, n_games=10)

# 1x PPO (first_0) vs 3x random -> isoleert het effect van RL
print("\n== PPO (first_0) vs 3x random ==")
mixed = {a: ppo_policies[a] if a == "first_0" else RandomPolicy(seed=SEED)
         for a in AGENT_ORDER}
summary_mixed = evaluate.run_tournament(make_parallel_env, mixed, n_games=10)

print("\nPPO first_0 win-rate vs 3x random:", summary_mixed["win_rate"]["first_0"])
"""
    ),
    code(
        r"""
# 7.3 Vergelijk win-percentages over configuraties.
fig = evaluate.plot_winrates(
    {
        "4x random": summary_random,
        "4x rule-based": summary_rule,
        "4x PPO": summary_ppo,
        "PPO vs 3x random": summary_mixed,
    },
    save_path="attachments/winrates.png",
)
plt.show()
"""
    ),
    md(
        r"""
### 7.4 Een wedstrijd opnemen (optioneel)
Leg een wedstrijd vast als video om het gedrag kwalitatief te beoordelen.
"""
    ),
    code(
        r"""
# 7.4 Neem één PPO-wedstrijd op als mp4.
import imageio
env = make_parallel_env(obs_type="ram", render_mode="rgb_array")
result = evaluate.play_match(env, ppo_policies, seed=123, record_frames=True)
os.makedirs("warlords_videos", exist_ok=True)
imageio.mimsave("warlords_videos/ppo_match.mp4", result["frames"], fps=15)
print("Winnaar:", result["winner"], "| stappen:", result["steps"])
"""
    ),
    # ---------------------------------------------------------------- 8. hyperparams
    md(
        r"""
## 8. Hyperparameter-experimenten (opdracht 3a)

We variëren één hyperparameter tegelijk en houden de rest vast, om het effect te
isoleren. Hieronder een sjabloon voor de **learning rate**; herhaal analoog voor
bijvoorbeeld `ent_coef` (exploratie), `survival_bonus` (reward shaping) of de
netwerkgrootte. Documenteer telkens de win-rate en de stabiliteit van de leercurve.

> Houd het aantal stappen klein per run als je meerdere waarden vergelijkt, of
> draai deze sectie als aparte (langere) experimenten en vat de resultaten samen
> in een tabel.
"""
    ),
    code(
        r"""
# 8.1 Sjabloon: vergelijk learning rates (klein gehouden voor snelheid).
results_lr = {}
for lr in [1e-4, 2.5e-4, 5e-4]:
    tag = f"lr={lr}"
    print(f"\n=== {tag} ===")
    train.train_independent(
        generations=1, steps_per_agent=20_000, n_envs=8,
        survival_bonus=0.01, seed=SEED,
        save_dir=f"experiments/{tag}",
        progress_bar=False,
        ppo_overrides=dict(learning_rate=lr),
    )
    m = {a: PPO.load(f"experiments/{tag}/ppo_{a}.zip", device="cpu") for a in AGENT_ORDER}
    pol = {a: PolicyOpponent(m[a], deterministic=True) for a in AGENT_ORDER}
    mixed = {a: pol[a] if a == "first_0" else RandomPolicy(seed=SEED) for a in AGENT_ORDER}
    s = evaluate.run_tournament(make_parallel_env, mixed, n_games=15, verbose=False)
    results_lr[tag] = s["win_rate"]["first_0"]

print("\nWin-rate (PPO first_0 vs 3x random) per learning rate:")
for tag, wr in results_lr.items():
    print(f"  {tag}: {wr:.2f}")
"""
    ),
    # ---------------------------------------------------------------- 9. tournament-agent
    md(
        r"""
## 9. Toernooi-agent (bonus)

Voor het klassentoernooi wordt onze agent in een **willekeurige** hoek geplaatst en
krijgt hij alleen de globale RAM (geen hoek-indicator). Eén beleid kan dan niet voor
elke hoek tegelijk optimaal zijn (zie §1.3). We trainen daarom een **hoek-robuust**
beleid dat afwisselend alle hoeken bestuurt; `Agent3`/`Agent4` vallen hierop terug
als er geen hoek-specifiek model is.
"""
    ),
    code(
        r"""
# 9.1 (optioneel) Train een hoek-robuust beleid voor het toernooi.
robust = train.train_corner_robust(
    total_timesteps=100_000, n_envs=8, survival_bonus=0.01, frame_skip=4,
    seed=SEED, save_dir="tournament/models", progress_bar=True,
)
backup_to_drive()
print("Hoek-robuust model opgeslagen als tournament/models/ppo_corner_robust.zip")
"""
    ),
    md(
        r"""
Draai vervolgens `tournament/warlords_tournament_ram_mode.ipynb` (met de vier
`agentN.py`-bestanden) om Agent1 (random) en Agent2 (rule-based) tegen Agent3/Agent4
(PPO) te laten spelen — een directe demonstratie van de meerwaarde van RL.
"""
    ),
    # ---------------------------------------------------------------- 10. conclusie
    md(
        r"""
## 10. Conclusie & Reflectie

**Samenvatting.** We hebben een MARL-systeem voor *Warlords* gebouwd: vier
independent PPO-policies, getraind met generatie-gewijs bevroren tegenstanders,
bovenop een single-agent wrapper rond de PettingZoo parallel-omgeving. De
prestaties zijn afgezet tegen een random- en een rule-based baseline.

**Wat biedt RL hier?** *(Vul in met je eigen resultaten.)* In tegenstelling tot de
vaste baselines past het RL-beleid zich aan de bal- en tegenstanderdynamiek aan en
verbetert het meetbaar gedurende de training (zie de leercurves en het hogere
win-percentage in §7).

**Beperkingen.**
- De beloning is zeer spaarzaam; zonder de survival-shaping leert PPO traag.
- Omdat alle hoeken dezelfde globale RAM zien zonder hoek-indicator, kan één
  gedeeld beleid de hoeken niet onderscheiden — dit beperkt een hoek-agnostische
  toernooi-agent fundamenteel.
- Generatie-gewijs bevriezen benadert echte gelijktijdige independent learning, maar
  is niet identiek (vgl. de Witt et al., 2020).

**Mogelijke uitbreidingen.**
- Een hoek-indicator toevoegen aan de observatie (bijv. one-hot), zodat één gedeeld
  beleid wél per hoek kan specialiseren (parameter sharing met agent-ID).
- Self-play met een *pool* van eerdere snapshots (PSRO / league-training) i.p.v. één
  generatie, voor robuustere strategieën.
- RAM-bytes kalibreren (§3.2) en in de reward gebruiken (bijv. balcontrole belonen).
- Vergelijken met een waarde-gebaseerde methode (DQN) of met pixel-observaties +
  CNN.
"""
    ),
    # ---------------------------------------------------------------- 11. genai
    md(
        r"""
## 11. Verantwoording Generative AI

> Volgens de cursusregels moet elk gebruik van Generative AI (GenAI) worden
> verantwoord met promptnummer, titel en link, plus een niet-GenAI-bron die de
> juistheid bevestigt. Vul hieronder je eigen prompts in (verwijder dit blok niet).

**Voorbeeld (format):**
- [ChatGPT, 2026. Prompt 1: PPO voor multi-agent Warlords](https://chat.openai.com/share/…) —
  bevestigd door Schulman et al. (2017) en Raffin et al. (2021).

In code-cellen verwijs je in een comment naar de prompt, bijv.:
`# Prompt 1: PPO voor multi-agent Warlords`.
"""
    ),
    # ---------------------------------------------------------------- 12. refs
    md(
        r"""
## 12. Referentielijst (APA)

- Bellemare, M. G., Naddaf, Y., Veness, J., & Bowling, M. (2013). The Arcade
  Learning Environment: An evaluation platform for general agents. *Journal of
  Artificial Intelligence Research, 47*, 253–279.
- Busoniu, L., Babuška, R., & De Schutter, B. (2008). A comprehensive survey of
  multiagent reinforcement learning. *IEEE Transactions on Systems, Man, and
  Cybernetics, 38*(2), 156–172.
- de Witt, C. S., Gupta, T., Makoviichuk, D., Makoviychuk, V., Torr, P. H. S.,
  Sun, M., & Whiteson, S. (2020). *Is independent learning all you need in the
  StarCraft multi-agent challenge?* arXiv:2011.09533.
- Heinrich, J., & Silver, D. (2016). *Deep reinforcement learning from self-play in
  imperfect-information games.* arXiv:1603.01121.
- Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017).
  Multi-agent actor-critic for mixed cooperative-competitive environments.
  *Advances in Neural Information Processing Systems, 30*.
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through
  deep reinforcement learning. *Nature, 518*(7540), 529–533.
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward
  transformations: Theory and application to reward shaping. *ICML*, 278–287.
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N.
  (2021). Stable-Baselines3: Reliable reinforcement learning implementations.
  *Journal of Machine Learning Research, 22*(268), 1–8.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  *Proximal policy optimization algorithms.* arXiv:1707.06347.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*
  (2nd ed.). MIT Press.
- Terry, J. K., Black, B., Grammel, N., et al. (2021). PettingZoo: Gym for
  multi-agent reinforcement learning. *Advances in Neural Information Processing
  Systems, 34*.
"""
    ),
]


def build_notebook() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build_notebook()
    with open("Portfolio3.ipynb", "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    print(f"Portfolio3.ipynb geschreven met {len(CELLS)} cellen.")
