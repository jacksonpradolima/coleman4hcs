"""Policies for multi-armed bandit and contextual bandit action selection.

This module provides a collection of policies that are designed to operate
with multi-armed bandits and contextual bandits. Each policy dictates how an
agent will select its actions based on prior knowledge, current context, or
exploration strategies.

Classes
-------
Policy
    Basic policy class that prescribes actions based on the memory of an agent.
EpsilonGreedyPolicy
    Chooses either the best apparent action or a random one based on a probability epsilon.
GreedyPolicy
    Always chooses the best apparent action.
RandomPolicy
    Always chooses a random action.
UCBPolicyBase
    Base class for Upper Confidence Bound policies.
UCB1Policy
    Implementation of the UCB1 algorithm.
UCBPolicy
    A variation of the UCB algorithm with a scaling factor.
FRRMABPolicy
    Fitness-Rate-Rank based Multi-Armed Bandit policy.
SlMABPolicy
    Sliding window-based Multi-Armed Bandit policy.
LinUCBPolicy
    Contextual bandit policy using linear upper confidence bounds.
SWLinUCBPolicy
    Variation of LinUCBPolicy using a sliding window approach.

Notes
-----
- UCB (Upper Confidence Bound) policies are designed to balance exploration and exploitation by
  considering both the estimated reward of an action and the uncertainty around that reward.
- EpsilonGreedy and its variations (Greedy, Random) are simpler strategies that either exploit
  the best-known action or explore random actions based on a fixed probability.
- LinUCB and SWLinUCB are contextual bandits. They choose actions not just based on past rewards,
  but also considering the current context. SWLinUCB adds a sliding window mechanism to LinUCB,
  giving more weight to recent actions.

References
----------
.. [1] Lihong Li, et al. "A Contextual-Bandit Approach to Personalized News Article
   Recommendation." In Proceedings of the 19th International Conference on World Wide
   Web (WWW), 2010.
.. [2] Nicolas Gutowski, Tassadit Amghar, Olivier Camp, and Fabien Chhel. "Global Versus
   Individual Accuracy in Contextual Multi-Armed Bandit." In Proceedings of the 34th
   ACM/SIGAPP Symposium on Applied Computing (SAC '19), April 8-12, 2019, Limassol, Cyprus.
"""

from .base import Policy, _rng
from .contextual import LinUCBPolicy, SWLinUCBPolicy
from .greedy import EpsilonGreedyPolicy, GreedyPolicy
from .mab import FRRMABPolicy, SlMABPolicy
from .random import RandomPolicy
from .ucb import UCB1Policy, UCBPolicy, UCBPolicyBase

__all__ = [
    "_rng",
    "Policy",
    "EpsilonGreedyPolicy",
    "GreedyPolicy",
    "RandomPolicy",
    "UCBPolicyBase",
    "UCB1Policy",
    "UCBPolicy",
    "FRRMABPolicy",
    "SlMABPolicy",
    "LinUCBPolicy",
    "SWLinUCBPolicy",
]
