"""
coleman4hcs.exceptions
----------------------

Defines custom exceptions used in the Coleman4HCS framework.

Classes
-------
QException
    Exception raised for Q-value computation errors in contextual
    multi-armed bandit (CMAB) policies.
"""

class QException(Exception):
    """Exception raised for Q-value computation errors in CMAB policies."""
