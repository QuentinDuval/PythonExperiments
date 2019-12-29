from ml.codingame.ultimate_tic_tac_toe_2 import *


"""
Test game
"""


def test_agent_prototypes():
    random_agent = RandomAgent()
    random_agent.on_end_episode()


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
    test_agent_prototypes()
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
    assert 9*9 == eval.weights.size


test_price_map()


"""
Test AI
"""


def test_ia(agent1: Agent, agent2: Agent):
    chrono = Chronometer()
    chrono.start()
    move_count = 0
    board = Board.empty()

    while not board.is_over():
        move = agent1.get_action(board, CROSS)
        board = board.play_debug(CROSS, move)
        move_count += 1
        if not board.is_over():
            move = agent2.get_action(board, CIRCLE)
            board = board.play_debug(CIRCLE, move)
            move_count += 1

    agent1.on_end_episode()
    agent2.on_end_episode()
    time_spent = chrono.spent()

    print("time spent:", time_spent)
    print("move count:", move_count)
    print("time per move:", time_spent / move_count)
    print("winner:", board.winner)
    # print(board)


print("\nNaive, depth 4 (minimax vs negamax)")
print("-" * 50)


for depth in 3, 4:
    test_ia(agent1=MinimaxAgent(max_depth=depth, eval_fct=CountOwnedEvaluation()),
            agent2=NegamaxAgent(max_depth=depth, eval_fct=CountOwnedEvaluation()))

    test_ia(agent1=NegamaxAgent(max_depth=depth, eval_fct=CountOwnedEvaluation()),
            agent2=MinimaxAgent(max_depth=depth, eval_fct=CountOwnedEvaluation()))


print("\nNaive, depth 4 (minimax)")
print("-" * 50)

test_ia(agent1=MinimaxAgent(max_depth=4, eval_fct=CountOwnedEvaluation()),
        agent2=MinimaxAgent(max_depth=4, eval_fct=CountOwnedEvaluation()))

test_ia(agent1=MinimaxAgent(max_depth=3, eval_fct=CountOwnedEvaluation()),
        agent2=MinimaxAgent(max_depth=4, eval_fct=CountOwnedEvaluation()))

test_ia(agent1=MinimaxAgent(max_depth=4, eval_fct=CountOwnedEvaluation()),
        agent2=MinimaxAgent(max_depth=3, eval_fct=CountOwnedEvaluation()))


print("\nNaive, depth 4 (negamax)")
print("-" * 50)

test_ia(agent1=NegamaxAgent(max_depth=4, eval_fct=CountOwnedEvaluation()),
        agent2=NegamaxAgent(max_depth=4, eval_fct=CountOwnedEvaluation()))

test_ia(agent1=NegamaxAgent(max_depth=3, eval_fct=CountOwnedEvaluation()),
        agent2=NegamaxAgent(max_depth=4, eval_fct=CountOwnedEvaluation()))

test_ia(agent1=NegamaxAgent(max_depth=4, eval_fct=CountOwnedEvaluation()),
        agent2=NegamaxAgent(max_depth=3, eval_fct=CountOwnedEvaluation()))


print("\nNaive VS better eval")
print("-" * 50)

test_ia(agent1=MinimaxAgent(max_depth=3, eval_fct=PriceMapEvaluation()),
        agent2=MinimaxAgent(max_depth=3, eval_fct=CountOwnedEvaluation()))

test_ia(agent1=MinimaxAgent(max_depth=3, eval_fct=CountOwnedEvaluation()),
        agent2=MinimaxAgent(max_depth=3, eval_fct=PriceMapEvaluation()))

print("\nMCTS vs Minimax")
print("-" * 50)

test_ia(agent1=MCTSAgent(exploration_factor=1.0),
        agent2=MinimaxAgent(max_depth=3, eval_fct=PriceMapEvaluation()))

test_ia(agent1=MinimaxAgent(max_depth=3, eval_fct=PriceMapEvaluation()),
        agent2=MCTSAgent(exploration_factor=1.0))

