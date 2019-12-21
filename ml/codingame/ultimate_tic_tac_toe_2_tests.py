from ml.codingame.ultimate_tic_tac_toe_2 import *


"""
Test game
"""


def test_initial_moves():
    board = Board.empty()
    available_moves = board.available_moves()
    assert len(available_moves) == 9 * 9


def test_row():
    sub_board = 0
    sub_board = Board._sub_play(sub_board, PLAYER, (0, 0))
    sub_board = Board._sub_play(sub_board, PLAYER, (1, 1))
    sub_board = Board._sub_play(sub_board, PLAYER, (2, 2))
    assert Board._sub_winner(sub_board, PLAYER)

    sub_board = 0
    sub_board = Board._sub_play(sub_board, PLAYER, (0, 0))
    sub_board = Board._sub_play(sub_board, PLAYER, (1, 1))
    board = Board.empty()
    board.sub_boards[(0, 0)] = sub_board
    board = board.play(PLAYER, (2, 2))
    assert PLAYER == board.sub_winners[(0, 0)]


def test_game_over():
    """
    x o x
    x o x
    o x o
    """
    sub_board = 0
    sub_board = Board._sub_play(sub_board, PLAYER, (0, 0))
    sub_board = Board._sub_play(sub_board, PLAYER, (0, 2))
    sub_board = Board._sub_play(sub_board, PLAYER, (1, 0))
    sub_board = Board._sub_play(sub_board, PLAYER, (1, 2))
    sub_board = Board._sub_play(sub_board, PLAYER, (2, 1))
    sub_board = Board._sub_play(sub_board, OPPONENT, (0, 1))
    sub_board = Board._sub_play(sub_board, OPPONENT, (1, 1))
    sub_board = Board._sub_play(sub_board, OPPONENT, (2, 0))
    sub_board = Board._sub_play(sub_board, OPPONENT, (2, 2))
    assert not Board._sub_available_moves((0, 0), sub_board)


def tests_game():
    test_initial_moves()
    test_row()
    test_game_over()


tests_game()


"""
Test AI
"""


def test_ia(agent1, agent2):
    chrono = Chronometer()
    chrono.start()
    move_count = 0
    board = Board.empty()

    while not board.is_game_over():
        move = agent1.get_action(board)
        board = board.play(PLAYER, move)
        move_count += 1
        if board.available_moves():
            move = agent2.get_action(board)
            board = board.play(OPPONENT, move)
            move_count += 1
    time_spent = chrono.spent()

    print("time spent:", time_spent)
    print("move count:", move_count)
    print("time per move:", time_spent / move_count)
    print(board.is_winner(PLAYER))
    print(board.is_winner(OPPONENT))
    print(board)


test_ia(agent1=MinimaxAgent(player=PLAYER, max_depth=2), agent2=MinimaxAgent(player=OPPONENT, max_depth=3))
# test_ia(agent1=MinimaxAgent(player=PLAYER, max_depth=3), agent2=MCTSAgent(player=OPPONENT, exploration_factor=1.0))

