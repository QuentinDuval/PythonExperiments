from ml.codingame.ultimate_tic_tac_toe_2 import *


"""
Test game
"""


def test_initial_moves():
    board = Board.empty()
    available_moves = board.available_moves
    assert len(available_moves) == 9 * 9


def test_row():
    sub_boards = np.zeros(shape=(9, 9))
    sub_boards[(0, 0)] = PLAYER
    sub_boards[(1, 1)] = PLAYER
    sub_boards[(2, 2)] = PLAYER
    assert PLAYER == Board._sub_winner(sub_boards, (0, 0))

    board = Board.empty()
    board = board.play(PLAYER, (0, 0))
    board = board.play(PLAYER, (1, 1))
    board = board.play(PLAYER, (2, 2))
    assert PLAYER == board.sub_winners[(0, 0)]


def test_game_over():
    """
    x o x
    x o x
    o x o
    """
    sub_boards = np.zeros(shape=(9, 9))
    for pos in [(0, 0), (0, 2), (1, 0), (1, 2), (2, 1)]:
        sub_boards[pos] = PLAYER
    for pos in [(0, 1), (1, 1), (2, 0), (2, 2)]:
        sub_boards[pos] = OPPONENT
    assert not Board._sub_available_moves(sub_boards, (0, 0))


def tests_game():
    test_initial_moves()
    test_row()
    test_game_over()


tests_game()


"""
Test evaluation functions
"""


def test_price_map():
    eval = PriceMapEvaluation()
    print(eval.weights)


test_price_map()


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
        if not board.is_game_over():
            move = agent2.get_action(board)
            board = board.play(OPPONENT, move)
            move_count += 1
    time_spent = chrono.spent()

    print("time spent:", time_spent)
    print("move count:", move_count)
    print("time per move:", time_spent / move_count)
    print("winner:", board.winner)
    # print(board)


test_ia(agent1=MinimaxAgent(player=PLAYER, max_depth=3, eval_fct=PriceMapEvaluation()),
        agent2=MinimaxAgent(player=OPPONENT, max_depth=3, eval_fct=CountOwnedEvaluation()))

test_ia(agent1=MinimaxAgent(player=PLAYER, max_depth=3, eval_fct=CountOwnedEvaluation()),
        agent2=MinimaxAgent(player=OPPONENT, max_depth=3, eval_fct=PriceMapEvaluation()))

# test_ia(agent1=MinimaxAgent(player=PLAYER, max_depth=3), agent2=MCTSAgent(player=OPPONENT, exploration_factor=1.0))

