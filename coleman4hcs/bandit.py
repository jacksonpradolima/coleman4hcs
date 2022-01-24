from typing import Dict, List

import pandas as pd
from coleman4hcs.evaluation import EvaluationMetric


class Bandit(object):
    """
    A Bandit
    """

    def __init__(self, arms: List[Dict]):
        """
        Init a Bandit with the its arms
        :param arms: The arms of the bandit (Test Cases record). Required columns are `self.tc_fieldnames`
        """
        # ColName | Description
        # Name | Unique numeric identifier of the test case
        # Duration | Approximated runtime of the test case
        # CalcPrio | Priority of the test case, calculated by the prioritization algorithm(output column, initially 0)
        # LastRun | Previous last execution of the test case as date - time - string(Format: `YYYY - MM - DD HH: ii`)
        # NumRan | Test runs
        # NumErrors | Test errors revealed
        # LastResults | List of previous test results(Failed: 1, Passed: 0), ordered by ascending age
        self.tc_fieldnames = ['Name',
                              'Duration',
                              'CalcPrio',
                              'LastRun',
                              'NumRan',
                              'NumErrors',
                              'Verdict',
                              'LastResults']

        self.reset()

        self.add_arms(arms)

        # Convert columns
        self.arms = self.arms.infer_objects()

    def reset(self):
        self.arms = pd.DataFrame(columns=self.tc_fieldnames)

    def get_arms(self):
        return self.arms['Name'].tolist()

    def add_arm(self, arm: Dict):
        """
        Add an arm in the bandit
        :param arm:
        :return:
        """
        self.arms = self.arms.append(pd.DataFrame([arm], columns=self.tc_fieldnames), ignore_index=True)

    def add_arms(self, arms: List[Dict]):
        """
        Add an arm in the bandit
        :param arms:
        :return:
        """
        self.arms = self.arms.append(pd.DataFrame(arms, columns=self.tc_fieldnames), ignore_index=True)

    def pull(self, action):
        return NotImplementedError('You must to implemented this function')

    def update_priority(self, action):
        """
        We update the Priority column with the priorities
        :param action: List of test cases in order of prioritization
        :return:
        """
        self.arms['CalcPrio'] = self.arms['Name'].apply(lambda x: action.index(x) + 1)


class DynamicBandit(Bandit):
    """
    Dynamic bandit is a kind of Bandit that allows us to manage its arms
    """
    def __init__(self, arms: List[Dict]):
        """
        :param arms: The arms of the bandit (Test Cases record)
        """
        super().__init__(arms)

    def update_arms(self, arms: List[Dict]):
        # I can replace all arms because the bandit don't need to maintain a "history"
        # The agent needs to maintain the "history"
        self.reset()

        # Add new arms
        self.add_arms(arms)

        # Convert columns
        self.arms = self.arms.infer_objects()

    def update_priority(self, action):
        super().update_priority(action)


class EvaluationMetricBandit(DynamicBandit):
    """
     Evaluation Metric Bandit is a kind of Dynamic Bandit that provide
     feedback for a pull (action) based on an evaluation metric, such as RNFail and TimeRank
    """
    def __init__(self, arms: List[Dict], evaluation_metric: EvaluationMetric):
        """
        :param arms: The arms of the bandit (Test Cases record)
        :param evaluation_metric: Evaluation Metric
        """
        super().__init__(arms)
        self.evaluation_metric = evaluation_metric

    def __str__(self):
        return str(self.evaluation_metric)

    def pull(self, action):
        """
        Submit prioritized test set for evaluation step the environment and get new measurements
        :param action: The Prioritized Test Suite List
        :return: The result ("state") of an evaluation by Evaluation Metric
        """
        super().update_priority(action)

        # After, we must to order the test cases based on the priorities
        # Sort tc by Prio ASC (for backwards scheduling)
        self.arms = self.arms.sort_values(by=['CalcPrio'])

        self.evaluation_metric.evaluate(self.arms.to_dict('records'))

        return self.evaluation_metric
