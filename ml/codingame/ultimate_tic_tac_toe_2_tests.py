from ml.codingame.ultimate_tic_tac_toe_2 import *


"""
Test game
"""


def test_initial_moves():
    board = Board.empty()
    available_moves = board.available_moves
    assert len(available_moves) == 9 * 9


def test_row():
    board = Board.empty()
    board = board.play(CROSS, (0, 0))
    board = board.play(CROSS, (1, 1))
    board = board.play(CROSS, (2, 2))
    assert CROSS == board._sub_winner((0, 0))
    assert CROSS == board.sub_winners[(0, 0)]


def test_sub_winner():
    board = Board.empty()
    board.grid = np.array([
        [0, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, -1, -1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [-1, 1, 0, -1, -1, -1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, 0],
        [-1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, -1, 0, 0, 0, 0]])
    assert EMPTY == board._sub_winner((0, 0))
    assert CIRCLE == board._sub_winner((0, 1))
    assert CIRCLE == board._sub_winner((1, 1))
    assert CROSS == board._sub_winner((2, 0))


def test_winner():
    for player in CROSS, CIRCLE:
        for combi in COMBINATIONS:
            board = Board.empty()
            for pos in combi:
                board.sub_winners[pos] = player
            assert player == board._winner()


def test_game_over():
    """
    x o x
    x o x
    o x o
    """
    board = Board.empty()
    for pos in [(0, 0), (0, 2), (1, 0), (1, 2), (2, 1)]:
        board.play_(CROSS, pos)
    for pos in [(0, 1), (1, 1), (2, 0), (2, 2)]:
        board.play_(CIRCLE, pos)
    assert not board._sub_available_moves((0, 0))


def tests_game():
    test_initial_moves()
    test_row()
    test_sub_winner()
    test_winner()
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

    while not board.is_over():
        move = agent1.get_action(board)
        board = board.play_debug(CROSS, move)
        move_count += 1
        if not board.is_over():
            move = agent2.get_action(board)
            board = board.play_debug(CIRCLE, move)
            move_count += 1
    time_spent = chrono.spent()

    print("time spent:", time_spent)
    print("move count:", move_count)
    print("time per move:", time_spent / move_count)
    print("winner:", board.winner)
    # print(board)


test_ia(agent1=MinimaxAgent(player=CROSS, max_depth=3, eval_fct=PriceMapEvaluation()),
        agent2=MinimaxAgent(player=CIRCLE, max_depth=3, eval_fct=CountOwnedEvaluation()))

test_ia(agent1=MinimaxAgent(player=CROSS, max_depth=3, eval_fct=CountOwnedEvaluation()),
        agent2=MinimaxAgent(player=CIRCLE, max_depth=3, eval_fct=PriceMapEvaluation()))

test_ia(agent1=MCTSAgent(player=CROSS, exploration_factor=1.0),
        agent2=MinimaxAgent(player=CIRCLE, max_depth=3, eval_fct=PriceMapEvaluation()))

test_ia(agent1=MinimaxAgent(player=CIRCLE, max_depth=3, eval_fct=PriceMapEvaluation()),
        agent2=MCTSAgent(player=CROSS, exploration_factor=1.0))

