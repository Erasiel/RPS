from copy import deepcopy

from ai_player.utils import get_winning_move, get_random_move


default_chances = {
    'rock': 0,
    'paper': 0,
    'scissors': 0
}

last_two_pick_chances = {
    'rock-rock': deepcopy(default_chances),
    'rock-paper': deepcopy(default_chances),
    'rock-scissors': deepcopy(default_chances),
    'paper-rock': deepcopy(default_chances),
    'paper-paper': deepcopy(default_chances),
    'paper-scissors': deepcopy(default_chances),
    'scissors-rock': deepcopy(default_chances),
    'scissors-paper': deepcopy(default_chances),
    'scissors-scissors': deepcopy(default_chances)
}

last_pick = None
pre_last_pick = None


def update_player1_chances(enemy_move: str) -> None:
    global last_two_pick_chances
    global last_pick
    global pre_last_pick

    if enemy_move == "none":
        return

    if last_pick is not None and pre_last_pick is not None:
        last_two_picks = f"{pre_last_pick}-{last_pick}"

        last_two_pick_chances[last_two_picks][enemy_move] += 1

    pre_last_pick = last_pick
    last_pick = enemy_move
    last_two_picks = f"{pre_last_pick}-{last_pick}"


def get_normal_action(enemy_move: str) -> str:
    """Returns the action of the normal difficulty AI player

    Normal difficulty AI player always acts based on the last two actions the
    human player took and tries to best the human player by its knowledge of 
    the human's picking pattern. This function updates the chances according 
    to the human player's pick, but does not take their current action into
    consideration.

    Returns:
        String with the AI player's pick.
    """

    global last_two_pick_chances
    global last_pick
    global pre_last_pick

    if enemy_move == "none":
        return get_random_move()

    ai_player_pick = ''

    # Get AI player's move
    if pre_last_pick is None or last_pick is None:
        ai_player_pick = 'rock' # TODO: random

    else:
        last_two_picks = f"{pre_last_pick}-{last_pick}"
        chances = last_two_pick_chances[last_two_picks]

        human_player_pick = "rock" # pick rock by default
        for action, chance in chances.items():
            if chance > chances[human_player_pick]:
                human_player_pick = action

        ai_player_pick = get_winning_move(human_player_pick)

    update_player1_chances(enemy_move)

    return ai_player_pick
