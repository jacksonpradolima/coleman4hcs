"""Reward functions for bandit-based test case prioritization.

Defines reward functions for agents in a multi-armed bandit framework in the context of
software testing. These reward functions help agents to prioritize software test cases
based on various strategies.

The module provides an abstract base class `Reward` that serves as a blueprint for all
reward functions. Derived classes implement specific reward strategies based on the number
of failures and the order of test cases.

Classes
-------
Reward
    An abstract base class that defines the structure and interface of a reward function.
TimeRankReward
    A reward function that considers the order of test cases and the number of failures.
RNFailReward
    A reward function that rewards based on the number of failures associated with test cases.

Notes
-----
Reward functions are essential components of the bandit-based test case prioritization
framework. They guide agents to make better decisions about which test cases to prioritize.
Ensure that the evaluation metric provides necessary details like detection ranks for the
reward functions to work correctly.
"""

from .base import Reward
from .rnfail import RNFailReward
from .timerank import TimeRankReward

__all__ = ["Reward", "TimeRankReward", "RNFailReward"]
