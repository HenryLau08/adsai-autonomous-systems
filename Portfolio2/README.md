# Portfolio 2 Deep Q-Network (Stargunner)

Een implementatie van een Deep Q-Network (DQN) agent die het Atari-spel Stargunner speelt via de Gymnasium-omgeving.

## Inhoud

- [stargunner_dqn.py](stargunner_dqn.py) – DQN implementatie en trainingslogica
- [Portfolio2.ipynb](Portfolio2.ipynb) – Notebook met uitleg, experimenten en resultaten
- [attachments/](attachments/) – Grafieken van de trainingresultaten

## Vereisten

```
gymnasium[atari]
ale-py
autorom[accept-rom-license]
torch
matplotlib
numpy
```

## Gebruik

```bash
python stargunner_dqn.py
```

## Resultaten

De agent is getraind met verschillende netwerkconfiguraties (1, 2 en 3 verborgen lagen). De resultaten zijn te vinden in de `attachments/` map en in de Notebook.
