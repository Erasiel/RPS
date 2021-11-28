def get_winning_move(enemy_move: str):
    if enemy_move == "rock":
        return "paper"
    elif enemy_move == "paper":
        return "scissors"
    elif enemy_move == "scissors":
        return "rock"
    else:
        return None

def get_winner(player1_move, player2_move):
    if "none" in [player1_move, player2_move]:
        return "Can't determine the winner"
    
    if player1_move == player2_move:
        return "Draw, try harder next time"

    if get_winning_move(player1_move) == player2_move:
        return "Player 1 wins!"
    else:
        return "Player 2 wins!"
