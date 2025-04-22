# rl-hw2/agents/base_agent.py

class BaseAgent:
    def select_action(self, state):
        raise NotImplementedError

    def train(self, experience):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError