"""
Solving basic probabilities problems based on Bayes rule
"""
from typing import Dict, Generic, TypeVar, List

Hypothesis = TypeVar('P')
Observation = TypeVar('O')

T = TypeVar('T')
ProbMassFunction = Dict[T, float]


def uniform(possibles: List[T]) -> ProbMassFunction[T]:
    n = len(possibles)
    return {possible: 1/n  for possible in possibles}


class BayesRule(Generic[Hypothesis, Observation]):
    """
    Takes as input
    - a list of hypothesis with prior probabilities
    - a list of conditional probabilities for each of these hypothesis
    - a list of observations
    And return the posterior probabilities
    """

    def __init__(self,
                 priors: ProbMassFunction[Hypothesis],
                 conditions: Dict[Hypothesis, ProbMassFunction[Observation]]):
        self.priors = priors
        self.conditions = conditions

    def observation(self, point: Observation):
        for hypothesis, posterior in self.priors.items():
            self.priors[hypothesis] = posterior * self.conditions[hypothesis].get(point, 0.)

    def posteriors(self) -> ProbMassFunction[Hypothesis]:
        self._normalize()
        return self.priors

    def posterior_of(self, hypothesis: Hypothesis) -> float:
        self._normalize()
        return self.priors[hypothesis]

    def _normalize(self):
        total = sum(self.priors.values())
        for hypothesis, posterior in self.priors.items():
            self.priors[hypothesis] = posterior / total


"""
4 hypothesis, for 4 different dices
- showing some rolls
- deducing the posterior
"""


bayes = BayesRule(
    priors=uniform(["6", "8", "12", "20"]),
    conditions={
        "6": uniform(list(range(1, 7))),
        "8": uniform(list(range(1, 9))),
        "12": uniform(list(range(1, 13))),
        "20": uniform(list(range(1, 21)))
    }
)

print(bayes.posteriors())
bayes.observation(4)
print(bayes.posteriors())
bayes.observation(6)
print(bayes.posteriors())
bayes.observation(7)
print(bayes.posteriors())
bayes.observation(6)
print(bayes.posteriors())


"""
Diagnosing a disease:
- two tests, A and B, with some false positive and false negative
- get the probability of being infected from observations
"""

p_sick = 0.001
bayes = BayesRule(
    priors={"infected": p_sick, "healthy": (1-p_sick)},
    conditions={
        "infected": {
            ("A", True): 0.99,
            ("A", False): 0.01,
            ("B", True): 0.9,
            ("B", False): 0.1
        },
        "healthy": {
            ("A", True): 0.02,
            ("A", False): 0.98,
            ("B", True): 0.03,
            ("B", False): 0.97
        }
    }
)

bayes.observation(("A", True))
print(bayes.posterior_of("infected"))
bayes.observation(("B", True))
print(bayes.posterior_of("infected"))


"""
Diagnosing a disease:
- imagine that all tests are independents and all have the same probabilities of false positive and false negative
- check how many iteration of observations you need before the priors do not matter much anymore
"""

# TODO
