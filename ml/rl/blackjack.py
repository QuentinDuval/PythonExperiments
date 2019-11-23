import abc
from collections import *
import enum
from dataclasses import *
from functools import *
from typing import *

import numpy as np


"""
Implementation of the game of Blackjack (with infinite deck of card)
"""


Reward = float


class Action(enum.Enum):
    STICK = 0
    HIT = 1

    @staticmethod
    def all():
        return [Action.HIT, Action.STICK]


@dataclass(frozen=True)
class VisibleState:
    dealer_card: int
    current_total: int
    has_usable_ace: bool

    @staticmethod
    @lru_cache(maxsize=1)
    def all():
        states = []
        states.extend(VisibleState.all_without_ace())
        states.extend(VisibleState.all_with_ace())
        return states

    @staticmethod
    @lru_cache(maxsize=1)
    def all_without_ace():
        states = []
        for player_total in range(4, 21 + 1):
            for dealer_card in range(1, 10 + 1):
                states.append(VisibleState(dealer_card=dealer_card, current_total=player_total, has_usable_ace=False))
        return states

    @staticmethod
    @lru_cache(maxsize=1)
    def all_with_ace():
        states = []
        for player_total in range(12, 21 + 1):
            for dealer_card in range(1, 10 + 1):
                states.append(VisibleState(dealer_card=dealer_card, current_total=player_total, has_usable_ace=True))
        return states


class Hand:
    DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    def __init__(self, cards):
        self.cards = list(cards)

    @classmethod
    def random(cls):
        return cls(cards=np.random.choice(cls.DECK, size=2))

    def pick_card(self):
        self.cards.append(np.random.choice(self.DECK))

    def is_bust(self):
        return self.total > 21

    @property
    def first_card(self):
        return self.cards[0]

    @property
    def total(self) -> int:
        return self.state[0]

    @property
    def usable_ace(self) -> bool:
        return self.state[1] > 0

    @property
    def state(self) -> Tuple[int, int]:
        total = 0
        usable_ace = 0
        for card in self.cards:
            if card == 1:
                total += 11
                usable_ace += 1
            else:
                total += card
        while total > 21 and usable_ace > 0:
            total -= 10
            usable_ace -= 1
        return total, usable_ace

    def __repr__(self):
        return repr(self.cards)


class BlackJack:
    def __init__(self):
        self.dealer: Hand = None
        self.player: Hand = None
        self.is_over: bool = False
        self.reset()

    def reset(self, dealer: Hand = None, player: Hand = None):
        self.dealer = dealer or Hand.random()
        self.player = player or Hand.random()
        self.is_over = False

    def get_state(self) -> VisibleState:
        total, usable_ace = self.player.state
        return VisibleState(
            dealer_card=self.dealer.first_card,
            current_total=total,
            has_usable_ace=usable_ace > 0)

    def get_actions(self) -> List[Action]:
        return Action.all()

    def is_done(self) -> bool:
        return self.is_over

    def play(self, action) -> Reward:
        if action == Action.HIT:
            self.player.pick_card()
            if self.player.total > 21:
                self.is_over = True
                return -1
            else:
                return 0
        elif action == Action.STICK:
            self.is_over = True
            self._dealer_move()
            if self.dealer.is_bust():
                return 1
            elif self.dealer.total == 21 and self.player.total == 21:
                return 0
            elif self.player.total > self.dealer.total:
                return 1
            else:
                return -1
        else:
            raise Exception("Invalid action: you loose")

    def _dealer_move(self):
        while self.dealer.total < 17:
            self.dealer.pick_card()


"""
Model of a BlackJack game (stochastic model)
"""


@dataclass(frozen=True)
class BlackJackTransition:
    probability: float
    reward: float
    state: VisibleState  # None if we reached the end of the game


class BlackJackModel(abc.ABC):

    @abc.abstractmethod
    def get_transitions(self, state: VisibleState, action: Action) -> List[BlackJackTransition]:
        pass


class BlackJackExactModel(BlackJackModel):
    def __init__(self):
        self.card_probs = OrderedDict()
        card_prod = 1 / len(Hand.DECK)
        for card in Hand.DECK:
            self.card_probs[card] = self.card_probs.get(card, 0) + card_prod

    @lru_cache(maxsize=None)
    def get_transitions(self, state: VisibleState, action: Action) -> List[BlackJackTransition]:
        transitions = self._on_hit(state) if action == Action.HIT else self._on_stick(state)
        return self._group_by_target_state(transitions)

    def _group_by_target_state(self, transitions: Generator[BlackJackTransition, None, None]):
        grouped_by_target_state = {}
        for transition in transitions:
            accumulated = grouped_by_target_state.get(transition.state, None)
            if accumulated is None:
                grouped_by_target_state[transition.state] = transition
            else:
                total_probability = transition.probability + accumulated.probability
                expected_reward = (accumulated.probability * accumulated.reward
                                   + transition.probability * transition.reward) / total_probability
                merged = BlackJackTransition(probability=total_probability, state=transition.state, reward=expected_reward)
                grouped_by_target_state[transition.state] = merged
        return list(grouped_by_target_state.values())

    def _on_hit(self, state: VisibleState) -> Generator[BlackJackTransition, None, None]:
        for card, prob in self.card_probs.items():
            hand = self._to_hand(state)
            hand.cards.append(card)
            new_state = VisibleState(dealer_card=state.dealer_card,
                                     current_total=hand.total,
                                     has_usable_ace=hand.usable_ace)
            if hand.total > 21:
                yield BlackJackTransition(probability=prob, reward=-1, state=None)
            else:
                yield BlackJackTransition(probability=prob, reward=0, state=new_state)

    def _to_hand(self, state: VisibleState) -> Hand:
        if not state.has_usable_ace:
            return Hand(cards=[state.current_total])
        else:
            return Hand(cards=[state.current_total - 11, 1])

    def _on_stick(self, state: VisibleState) -> Generator[BlackJackTransition, None, None]:
        for card, prob in self.card_probs.items():
            dealer_hand = Hand(cards=[state.dealer_card, card])
            yield from self._on_dealer_turn(state, dealer_hand, prob)

    def _on_dealer_turn(self, state: VisibleState, dealer_hand: Hand, prob: float) -> Generator[
        BlackJackTransition, None, None]:
        if dealer_hand.total > 21:
            yield BlackJackTransition(probability=prob, reward=1, state=None)
        elif dealer_hand.total == 21 and state.current_total == 21:
            yield BlackJackTransition(probability=prob, reward=0, state=None)
        elif dealer_hand.total >= 17:
            reward = 1 if state.current_total > dealer_hand.total else -1
            yield BlackJackTransition(probability=prob, reward=reward, state=None)
        else:
            for card, card_prob in self.card_probs.items():
                new_dealer_hand = Hand(dealer_hand.cards)
                new_dealer_hand.cards.append(card)
                yield from self._on_dealer_turn(state, new_dealer_hand, prob=prob * card_prob)
