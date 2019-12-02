import abc


class Agent(abc.ABC):
    """
    Basic interface for any agent acting on an environment, in a given state
    """

    @abc.abstractmethod
    def get_action(self, env, state):
        pass

