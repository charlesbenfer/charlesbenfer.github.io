import tkinter as tk
import random


# Initialize game state
class GameState:
    def __init__(self):
        self.inning = 1
        self.outs = 0
        self.score = [0, 0]  # [Team 1, Team 2]
        self.bases = [0, 0, 0]  # 1st, 2nd, 3rd
        self.current_team = 0  # 0 for Team 1, 1 for Team 2
        self.current_event = ""

    def reset_inning(self):
        self.outs = 0
        self.bases = [0, 0, 0]


# Event dictionary for dice roll combinations
EVENTS = {
    (1, 1): "Home Run",
    (1, 2): "Double",
    (2, 1): "Double",
    (1, 3): "Fly Out",
    (3, 1): "Fly Out",
    (1, 4): "Walk",
    (4, 1): "Walk",
    (1, 5): "Pop Out",
    (5, 1): "Pop Out",
    (6, 1): "Single",
    (1, 6): "Single",
    (2, 2): "Double Play",
    (2, 3): "Ground Out",
    (3, 2): "Ground Out",
    (2, 4): "Strike Out",
    (4, 2): "Strike Out",
    (2, 5): "Single",
    (5, 2): "Single",
    (2, 6): "Strike Out",
    (6, 2): "Strike Out",
    (3, 3): "Walk",
    (3, 4): "Triple",
    (4, 3): "Triple",
    (3, 5): "Ground Out",
    (5, 3): "Ground Out",
    (3, 6): "Fly Out",
    (6, 3): "Fly Out",
    (4, 4): "Walk",
    (4, 5): "Pop Out",
    (5, 4): "Pop Out",
    (4, 6): "Strike Out",
    (6, 4): "Strike Out",
    (5, 5): "Double",
    (5, 6): "Sacrifice Fly",
    (6, 5): "Sacrifice Fly",
    (6, 6): "Home Run"
}


def roll_dice():
    return random.randint(1, 6), random.randint(1, 6)


def resolve_event(roll, game_state):
    event = EVENTS.get(roll)
    bases = game_state.bases
    score = game_state.score
    outs = game_state.outs

    if event == "Home Run":
        score[game_state.current_team] += sum(bases) + 1
        bases[:] = [0, 0, 0]
    elif event == "Single":
        bases.insert(0, 1)
        score[game_state.current_team] += bases.pop()
    elif event == "Double":
        bases.insert(0, 0)
        bases.insert(0, 1)
        score[game_state.current_team] += bases.pop()
        score[game_state.current_team] += bases.pop()
    elif event == "Triple":
        bases.insert(0, 0)
        bases.insert(0, 0)
        bases.insert(0, 1)
        score[game_state.current_team] += sum(bases[3:])
        bases[:] = [0, 0, 0]
    elif event == "Walk":
        if 0 in bases:
            bases[bases.index(0)] = 1
        else:
            score[game_state.current_team] += bases.pop()
            bases.insert(0, 1)
    elif event == "Sacrifice Fly":
        if bases[2] == 1:
            outs += 1
            score[game_state.current_team] += 1
            bases[2] = 0
        else:
            outs += 1
    elif event == "Double Play":
        if sum(bases) > 0:
            outs += 2
            if bases[0] == 1:  # Runner on 1st
                bases[0] = 0
            elif bases[1] == 1:  # Runner on 2nd
                bases[1] = 0
        else:
            outs += 1
    elif event == "Fly Out" or event == "Pop Out" or event == "Ground Out":
        outs += 1
    elif event == "Strike Out":
        outs += 1

    return event, score, outs


# GUI setup
def baseball_game_gui():
    game_state = GameState()

    def roll_action():
        roll = roll_dice()
        event, score, game_state.outs = resolve_event(roll, game_state)
        game_state.current_event = event
        update_display()
        if game_state.outs >= 3:
            end_inning()

    def end_inning():
        game_state.reset_inning()
        game_state.current_team = 1 - game_state.current_team
        if game_state.current_team == 0:
            game_state.inning += 1
        update_display()
        if game_state.inning > 9:
            end_game()

    def end_game():
        result = f"Game Over! Final Score:\nTeam 1: {game_state.score[0]} - Team 2: {game_state.score[1]}"
        result_label.config(text=result)
        roll_button.config(state="disabled")

    def update_display():
        inning_label.config(text=f"Inning: {game_state.inning}")
        outs_label.config(text=f"Outs: {game_state.outs}")
        bases_label.config(text=f"Bases: {game_state.bases}")
        score_label.config(text=f"Score: Team 1: {game_state.score[0]} - Team 2: {game_state.score[1]}")
        event_label.config(text=f"Last Event: {game_state.current_event}")

    # Tkinter GUI
    root = tk.Tk()
    root.title("Baseball Dice Game")

    # Labels
    inning_label = tk.Label(root, text=f"Inning: {game_state.inning}")
    inning_label.pack()

    outs_label = tk.Label(root, text=f"Outs: {game_state.outs}")
    outs_label.pack()

    bases_label = tk.Label(root, text=f"Bases: {game_state.bases}")
    bases_label.pack()

    score_label = tk.Label(root, text=f"Score: Team 1: {game_state.score[0]} - Team 2: {game_state.score[1]}")
    score_label.pack()

    event_label = tk.Label(root, text="Last Event: ")
    event_label.pack()

    result_label = tk.Label(root, text="")
    result_label.pack()

    # Buttons
    roll_button = tk.Button(root, text="Roll Dice", command=roll_action)
    roll_button.pack()

    quit_button = tk.Button(root, text="Quit", command=root.quit)
    quit_button.pack()

    root.mainloop()


# Run the GUI
baseball_game_gui()
