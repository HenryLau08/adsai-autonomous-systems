# Vier op een Rij (PettingZoo + Rule-based)

Een implementatie van Vier op een Rij op basis van de PettingZoo-omgeving, met ondersteuning voor menselijke spelers, meerdere AI-strategieën en een volledig visuele spelinterface.

## Functies

- Menselijke speler tegen AI, of AI tegen AI
- Meerdere regelgebaseerde strategieën met oplopende complexiteit
- Volledig bord renderen in de terminal én als grafische weergave
- Zet-validatie en windetectie via de PettingZoo-omgeving
- Modulaire, goed gedocumenteerde Python-code

---

## Projectstructuur
```
├── Board               # Bordweergave en UI-rendering
├── strategies          # Alle AI-strategieën en de menselijke spelerstrategie
├── helpers             # Helpers functies voor de strategiën
├── Portfolio1.ipynb    # Rule-based portfolio voor Connect 4 (game kan je hierin spelen)
└── README.md
```

---

## AI-strategieën

Dit project implementeert vijf strategieën, van een eenvoudige basislijn tot geavanceerdere regelgebaseerde agents. Het doel is hun prestaties te vergelijken, sterke en zwakke punten te begrijpen en spelers de mogelijkheid te geven elke strategie uit te dagen.

| Strategie       | Omschrijving                                                                 |
|-----------------|------------------------------------------------------------------------------|
| **Random**      | Kiest een willekeurige geldige kolom. Dient als basislijn.                  |
| **Defensive**   | Blokkeert dreigingen van de tegenstander zodra die worden herkend.          |
| **Smart**       | Combineert aanvallend spel met verdediging op basis van vaste regels.       |
| **Geef Niet Op**| Past zijn strategie aan op de stand van het spel; geeft niet snel op.      |
| **One to Five** | Doorzoekt meerdere zetten vooruit aan de hand van een puntensysteem.        |

---

## Basislijnstrategie

De basislijn is bewust eenvoudig gehouden:

- Kiest een willekeurige geldige kolom.
- Heeft geen vooruitkijk, geen heuristieken en geen tactisch bewustzijn.
- Dient als referentiepunt om de meerwaarde van geavanceerdere strategieën te meten.

Elke strategie die de basislijn consistent verslaat, laat zien hoeveel winst regelgebaseerde logica oplevert ten opzichte van puur willekeurig spel.

---

## Toernooimodus

Alle AI-strategieën kunnen automatisch tegen elkaar worden gespeeld via de toernooimodule. Elke combinatie speelt een instelbaar aantal rondes; de resultaten worden weergegeven als kruistabel en ranglijst.
```python
results, names = run_tournament(AI_STRATEGIES, rounds=100)
print_tournament_results(results, names, rounds=100)
visualize_tournament(results, names, rounds=100)
```

---

## Installatie
```bash
pip install -r Portfolio1/requirements.txt
```

---

## Gebruik

Open Portfolio1 Jupyter-notebook. Selecteer via de dropdowns welke strategieën als speler 0 en speler 1 optreden en klik op **Start Game**.
```python
# Voorbeeld: menselijke speler tegen Smart
player0_dropdown.value = "Human"
player1_dropdown.value = "Smart"
```

---

## Afhankelijkheden

- [PettingZoo](https://pettingzoo.farama.org/) — multi-agent spelomgeving
- [Matplotlib](https://matplotlib.org/) — grafische bordweergave
- [NumPy](https://numpy.org/) — bordrepresentatie en berekeningen
- [ipywidgets](https://ipywidgets.readthedocs.io/) — interactieve Jupyter-interface