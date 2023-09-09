"""
`coleman4hcs.utils.monitor` - Monitor Utilities

This module provides tools for monitoring and collecting data during experiments related to
the Coleman4HCS framework. The primary functionality revolves around the `MonitorCollector` class, which
facilitates data collection during an experiment and provides methods for saving the collected data
to a CSV file.
"""
import csv
import os

import pandas as pd

from coleman4hcs.evaluation import EvaluationMetric


class MonitorCollector:
    """
    Collects data during an experiment.
    """

    def __init__(self):
        """
        Initializes the monitor collector with predefined column names and an empty dataframe.
        """

        # experiment: Experiment number
        # step: Part number (Build) from scenario that is been analyzed
        # policy: Policy name that is evaluating a part of the scenario
        # reward_function: Reward function used by the agent to observe the environment
        # sched_time: Percentage of time available (i.e., 50% of total for the Build)
        # sched_time_duration: The time in number obtained from percentage.
        # total_build_duration: Build Duration
        # prioritization_time: Prioritization Time
        # detected: Failures detected
        # missed: Failures missed
        # tests_ran: Number of tests executed
        # tests_ran_time: Time spent by the test cases executed
        # tests_not_ran: Number of tests not executed
        # ttf: Rank of the Time to Fail (Order of the first test case which failed)
        # time_reduction: Time Reduction (Total Time for the Build - Time spent until the first test case fail)
        # fitness: Evaluation metric result (example, NAPFD)
        # cost: Evaluation metric that considers cost, for instance, APFDc
        # rewards: AVG Reward from the prioritized test set
        # recall: How much test cases we found (detected/total)
        # avg_precision: 1 - We found all failures, 123 - We did not found all failures
        # prioritization_order: prioritized test set
        self.col_names = ['experiment',
                          'step',
                          'policy',
                          'reward_function',
                          'sched_time',
                          'sched_time_duration',
                          'total_build_duration',
                          'prioritization_time',
                          'detected',
                          'missed',
                          'tests_ran',
                          # 'tests_ran_time',
                          'tests_not_ran',
                          'ttf',
                          'ttf_duration',
                          'time_reduction',
                          'fitness',
                          'cost',
                          'rewards',
                          # 'recall',
                          'avg_precision',
                          'prioritization_order']

        self.df = pd.DataFrame(columns=self.col_names)

        # the temp is used when we have more than 1000 records. This is used to improve the performance
        self.temp_df = pd.DataFrame(columns=self.col_names)

    def collect_from_temp(self):
        """
        Transfers data from the temporary dataframe to the main dataframe and clears the temporary dataframe.
        """
        # Pass the temp dataframe to the original dataframe
        self.df = pd.concat([self.df, self.temp_df])
        # Empty the temp dataframe
        self.temp_df = pd.DataFrame(columns=self.col_names)
        # This can boost our performance by around 10 times

    def collect(self,
                scenario_provider,
                available_time,
                experiment,
                t,
                policy,
                reward_function,
                metric: EvaluationMetric,
                total_build_duration,
                prioritization_time,
                rewards,
                prioritization_order):
        """
        This function collects the feedback of an analysis and stores in a dataframe.
        In this way, i.e., I can export a BIG experiment to CSV
        :param scenario_provider: Scenario in analysis
        :param available_time:
        :param experiment: Experiment number
        :param t: Part number (Build) from scenario that is been analyzed.
        :param policy: Policy name that is evaluating a part (sc) of the scenario
        :param reward_function: Reward function used by the agent to observe the environment
        :param metric: The result (metric) of the analysis
        :param total_build_duration: Build Duration
        :param prioritization_time: Prioritization time
        :param rewards: AVG Reward from the prioritized test set
        :param prioritization_order: prioritized test set
        :return:
        """
        if len(self.temp_df) > 1000:
            self.collect_from_temp()

        records = {
            'experiment': experiment,
            'step': t,
            'policy': policy,
            'reward_function': reward_function,
            'sched_time': scenario_provider.avail_time_ratio,
            'sched_time_duration': available_time,
            'total_build_duration': total_build_duration,
            'prioritization_time': prioritization_time,
            'detected': metric.detected_failures,
            'missed': metric.undetected_failures,
            'tests_ran': len(metric.scheduled_testcases),
            'tests_not_ran': len(metric.unscheduled_testcases),
            'ttf': metric.ttf,
            'ttf_duration': metric.ttf_duration,
            'time_reduction': total_build_duration - metric.ttf_duration,
            'fitness': metric.fitness,
            'cost': metric.cost,
            'rewards': rewards,
            'avg_precision': metric.avg_precision,
            'prioritization_order': prioritization_order
        }

        self.temp_df.loc[len(self.temp_df)] = records

    def create_file(self, name):
        """
        Creates a CSV file with the column headers if it doesn't exist.
        """
        # if the file not exist, we need to create the header
        if not os.path.isfile(name):
            with open(name, 'w', encoding='utf-8') as f:
                f.write(";".join(self.col_names) + "\n")

    def save(self, name):
        """
        Saves the collected data to a CSV file.
        """
        # Collect data remain
        if len(self.temp_df) > 0:
            self.collect_from_temp()

        self.df.to_csv(name, mode='a+', sep=';', na_rep='[]',
                       header=False,
                       columns=self.col_names,
                       index=False,
                       quoting=csv.QUOTE_NONE)
