"""Agent4 - getrainde PPO-agent voor de vierde hoek (fourth_0).

Hergebruikt de :class:`PPOAgent` uit ``agent3`` (beide bestanden worden samen
ingeleverd). Door een andere hoek te laden speelt Agent4 een ander, onafhankelijk
getraind beleid dan Agent3 -- precies het idee van *independent learners*.
"""

from agent3 import PPOAgent


class Agent4(PPOAgent):
    """PPO-agent voor de vierde hoek (fourth_0) in het toernooi."""

    def __init__(self, corner="fourth_0", **kwargs):
        super().__init__(corner=corner, **kwargs)
