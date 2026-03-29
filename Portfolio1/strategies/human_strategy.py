import numpy as np

def human_strategy_console(board, mask, player):
    valid_cols = np.where(mask)[0]
    valid_list = ", ".join(str(c) for c in valid_cols)
    while True:
        try:
            col = int(input(f"Choose a column {valid_list}: "))
            if col in valid_cols:
                return col
            print("Invalid column, try again.")
        except ValueError:
            print("Please enter a number.")