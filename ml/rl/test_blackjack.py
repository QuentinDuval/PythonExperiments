from ml.rl.blackjack import *


def test_hand():
    hand = Hand(cards=[1, 10])
    assert hand.first_card == 1
    assert hand.total == 21
    assert hand.usable_ace
    assert not hand.is_bust()

    hand = Hand(cards=[1, 5, 6])
    assert hand.first_card == 1
    assert hand.total == 12
    assert not hand.usable_ace
    assert not hand.is_bust()

    hand = Hand(cards=[1, 1])
    assert hand.first_card == 1
    assert hand.total == 12
    assert hand.usable_ace
    assert not hand.is_bust()

    hand = Hand(cards=[10, 1, 10])
    assert hand.first_card == 10
    assert hand.total == 21
    assert not hand.usable_ace
    assert not hand.is_bust()

    hand = Hand(cards=[10, 1, 1])
    assert hand.total == 12
    assert not hand.is_bust()


def test_states():
    assert len(VisibleState.all()) == (21 - 4 + 1) * 10 + (21 - 12 + 1) * 10
    assert len(set(VisibleState.all())) == len(VisibleState.all())


def test_blackjack(iterations: int):
    game = BlackJack()
    for _ in range(iterations):
        game.reset()
        reward = game.play(Action.HIT)
        assert (not game.is_over and reward == 0) or (game.player.is_bust() and reward == -1)
        game.reset()
        reward = game.play(Action.STICK)
        assert game.is_over
        assert game.dealer.total >= 17
        if reward > 0:
            assert game.player.total > game.dealer.total or game.dealer.is_bust()
        elif reward == 0:
            assert game.player.total == 21 == game.dealer.total
        else:
            assert game.player.total <= game.dealer.total


def test_blackjack_model():
    model = BlackJackExactModel()
    transitions = model.get_transitions(
        state=VisibleState(dealer_card=1, current_total=21, has_usable_ace=False),
        action=Action.HIT)
    assert len(transitions) == 1 and transitions[0].reward == -1 and not transitions[0].state

    transitions = model.get_transitions(
        state=VisibleState(dealer_card=1, current_total=20, has_usable_ace=False),
        action=Action.HIT)
    assert len(transitions) == 2

    for state in VisibleState.all_with_ace():
        transitions = model.get_transitions(state, Action.HIT)
        for t in transitions:
            assert t.state

    for state in VisibleState.all():
        for action in Action.all():
            transitions = model.get_transitions(state, action)
            total_prob = sum(t.probability for t in transitions)
            assert 0.99 <= total_prob <= 1.0


test_hand()
test_states()
test_blackjack(iterations=1000)
test_blackjack_model()
