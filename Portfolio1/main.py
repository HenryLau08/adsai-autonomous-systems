from Strategies import random_strategy, smart_strategy, defensive_strategy
from pettingzoo.classic import connect_four_v3
from Board import convert_observation, print_board, clear_output, create_game_ui, print_text_board
import matplotlib.pyplot as plt
import time



# # -----------------------------
# # strategies
# # -----------------------------
# STRATEGIES = {
#     "Random":    random_strategy,
#     "Defensive": defensive_strategy,
#     # "Smart":   smart_strategy,
#     # "Strategy4": strategy4,
#     # "Strategy5": strategy5,
# }

# ROUNDS = 10


# # -----------------------------
# # run game
# # -----------------------------
# def run_game(strategy_p0, strategy_p1, seed=None):
#     """
#     Runs one game between two strategies.
#     Returns: "p0", "p1", or "draw"
#     """
#     env = connect_four_v3.env()
#     env.reset(seed=seed)

#     result = "draw"

#     for agent in env.agent_iter():
#         obs, reward, term, trunc, _ = env.last()

#         if term or trunc:
#             if reward == 1:
#                 result = "p0" if agent == "player_0" else "p1"
#             env.step(None)
#         else:
#             board = convert_observation(obs["observation"], agent)
#             observation = {"board": board, "action_mask": obs["action_mask"]}

#             if agent == "player_0":
#                 action = strategy_p0(observation, agent)
#             else:
#                 action = strategy_p1(observation, agent)

#             env.step(action)

#     env.close()
#     return result


# # -----------------------------
# # toernooi
# # -----------------------------
# def run_tournament(strategies, rounds=10):
#     names = list(strategies.keys())
#     # results[a][b] = {"wins": x, "losses": y, "draws": z} for a as player_0 vs b as player_1
#     results = {a: {b: {"wins": 0, "losses": 0, "draws": 0} for b in names} for a in names}

#     for i, name_a in enumerate(names):
#         for j, name_b in enumerate(names):
#             if name_a == name_b:
#                 continue
#             for r in range(rounds):
#                 outcome = run_game(strategies[name_a], strategies[name_b], seed=r)
#                 if outcome == "p0":
#                     results[name_a][name_b]["wins"] += 1
#                 elif outcome == "p1":
#                     results[name_a][name_b]["losses"] += 1
#                 else:
#                     results[name_a][name_b]["draws"] += 1

#     return results, names


# def print_tournament_results(results, names, rounds):
#     col_w = 22
#     name_w = 12

#     matchups = [(a, b) for a in names for b in names if a != b]
#     headers = [f"{a} vs {b}" for a, b in matchups]

#     # Header row
#     print("\n" + "=" * (name_w + col_w * len(matchups)))
#     print(f"{'Strategy':<{name_w}}" + "".join(f"{h:^{col_w}}" for h in headers))
#     print("-" * (name_w + col_w * len(matchups)))

#     for name in names:
#         row = f"{name:<{name_w}}"
#         for a, b in matchups:
#             if name == a:
#                 r = results[a][b]
#                 cell = f"W{r['wins']} L{r['losses']} D{r['draws']}"
#             elif name == b:
#                 r = results[a][b]
#                 cell = f"W{r['losses']} L{r['wins']} D{r['draws']}"
#             else:
#                 cell = "-"
#             row += f"{cell:^{col_w}}"
#         print(row)

#     print("=" * (name_w + col_w * len(matchups)))

#     # Total wins summary
#     print("\nTotal wins across all matchups:")
#     totals = []
#     for name in names:
#         total_wins = sum(results[name][b]["wins"] for b in names if b != name)
#         totals.append((name, total_wins))
#     for name, w in sorted(totals, key=lambda x: -x[1]):
#         print(f"  {name}: {w} wins / {(len(names)-1)*rounds} games")


# # -----------------------------
# # Run
# # -----------------------------
# results, names = run_tournament(STRATEGIES, rounds=ROUNDS)
# print_tournament_results(results, names, rounds=ROUNDS)




# Uit mohamed's branch
# ==================== GAME LOOP ====================
from pettingzoo.classic import connect_four_v3

env = connect_four_v3.env(render_mode="rgb_array")
env.reset(seed=42)

for agent in env.agent_iter():
    raw_observation, reward, termination, truncation, info = env.last()
    board = convert_observation(raw_observation["observation"])
    mask = raw_observation["action_mask"]
    
    # UI Update
    clear_output(wait=True)
    current_player = 1 if agent == "player_0" else 2
    
    if termination or truncation:
        if reward == 1:
            status = "GEWONNEN!" if agent == "player_0" else "ADS & AI WINT!"
        elif reward == -1:
            status = "ADS & AI WINT!" if agent == "player_0" else "GEWONNEN!"
        else:
            status = "GELIJKSPEL!"
            
        fig = create_game_ui(board, current_player, status)
        plt.show()
        print_text_board(board, mask)
        break
    
    fig = create_game_ui(board, current_player)
    plt.show()
    print_text_board(board, mask)
    
    if agent == "player_0":
        while True:
            try:
                action = int(input("\nVoer een getal in (0-6): "))
                if 0 <= action < 7 and mask[action]: break
                print("Ongeldige zet.")
            except: print("Voer a.u.b. een getal in.")
    else:
        print("\nADS & AI denkt na...")
        time.sleep(1)
        action = smart_strategy(board, mask)
    
    env.step(action)

env.close()