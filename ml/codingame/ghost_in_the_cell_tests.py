from ml.codingame.ghost_in_the_cell import *


def test_topology_next_hops():
    """
    0 - 1 - 2
    """

    g = {0: {1: 4, 2: 7},
         1: {0: 4, 2: 4},
         2: {0: 7, 1: 4}}

    game_state = GameState(turn_nb=1)
    game_state.factories[0] = Factory(owner=1, cyborg_count=10, production=10)
    game_state.factories[1] = Factory(owner=1, cyborg_count=10, production=10)
    game_state.factories[2] = Factory(owner=-1, cyborg_count=10, production=10)

    topology = Topology.from_graph(g)
    topology.compute_paths(game_state)

    assert topology.next_move_hop(0, 1) == 1
    assert topology.next_move_hop(0, 2) == 1

    """
        1       4
    0       3       6
        2       5
    """
    edges = [
        [0, 1, 4],
        [0, 2, 5],
        [0, 3, 8],

        [6, 4, 4],
        [6, 5, 5],
        [6, 3, 8],

        [1, 3, 5],
        [1, 4, 5],
        [1, 6, 8],

        [2, 3, 4],
        [2, 5, 6],
        [2, 6, 9],

        [4, 3, 5],
        [5, 3, 4]
    ]

    game_state = GameState(turn_nb=1)
    for i in range(3):
        game_state.factories[i] = Factory(owner=1, cyborg_count=10, production=10)
    for i in range(4, 7):
        game_state.factories[i] = Factory(owner=-1, cyborg_count=10, production=10)
    game_state.factories[3] = Factory(owner=0, cyborg_count=10, production=10)

    topology = Topology.from_edges(edges)
    topology.compute_paths(game_state)

    assert topology.next_move_hop(0, 3) == 1
    assert topology.next_move_hop(0, 2) == 2
    assert topology.next_move_hop(0, 4) == 1
    assert topology.next_move_hop(0, 6) == 1


def test_topology_camps():
    """
        1       4
    0       3       6
        2       5
    """
    edges = [
        [0, 1, 4],
        [0, 2, 5],
        [0, 3, 8],
        [1, 3, 5],
        [2, 3, 4],
        [6, 4, 4],
        [6, 5, 5],
        [6, 3, 8],
        [4, 3, 5],
        [5, 3, 4]
    ]
    topology = Topology.from_edges(edges)

    game_state = GameState(turn_nb=1)
    game_state.factories[0] = Factory(owner=1, cyborg_count=10, production=10)
    game_state.factories[6] = Factory(owner=-1, cyborg_count=10, production=10)
    topology.compute_camps(game_state)
    assert topology.get_camp(0) == 1
    assert topology.get_camp(1) == 1
    assert topology.get_camp(2) == 1
    assert topology.get_camp(4) == -1
    assert topology.get_camp(5) == -1
    assert topology.get_camp(6) == -1
    assert topology.get_camp(3) == 0


def run_unit_tests():
    test_topology_next_hops()
    test_topology_camps()


if __name__ == '__main__':
    run_unit_tests()
