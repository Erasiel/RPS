from ai_player.utils import get_random_move, get_winning_move

def get_hard_action(enemy_action: str):
    winning_move = get_winning_move(enemy_action)

    if winning_move:
        return winning_move

    return get_random_move()
