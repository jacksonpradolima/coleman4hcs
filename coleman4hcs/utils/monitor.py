"""
`coleman4hcs.utils.monitor` - Monitor Utilities

This module provides tools for monitoring and collecting data during experiments related to
the Coleman4HCS framework. The primary functionality revolves around the `MonitorCollector` class, which
facilitates data collection during an experiment and provides methods for saving the collected data
to a CSV file.
"""
import csv
import os

import polars as pl

from coleman4hcs.utils.monitor_params import CollectParams


class MonitorCollector:
    """
    Collects data during an experiment.
    """

    def __init__(self):
        """
        Initializes the monitor collector with predefined column names and an empty dataframe.
        """

        # scenario: Experiment name (system under test)
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
        self.col_names = ['scenario',
                          'experiment',
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

        # Define schema for the DataFrame
        schema = {
            'scenario': pl.Utf8,
            'experiment': pl.Int64,
            'step': pl.Int64,
            'policy': pl.Utf8,
            'reward_function': pl.Utf8,
            'sched_time': pl.Float64,
            'sched_time_duration': pl.Float64,
            'total_build_duration': pl.Float64,
            'prioritization_time': pl.Float64,
            'detected': pl.Int64,
            'missed': pl.Int64,
            'tests_ran': pl.Int64,
            'tests_not_ran': pl.Int64,
            'ttf': pl.Float64,
            'ttf_duration': pl.Float64,
            'time_reduction': pl.Float64,
            'fitness': pl.Float64,
            'cost': pl.Float64,
            'rewards': pl.Float64,
            'avg_precision': pl.Float64,
            'prioritization_order': pl.Utf8
        }
        self.df = pl.DataFrame(schema=schema)

        # the temp is used when we have more than 1000 records. This is used to improve the performance
        self.temp_rows = []
        self.temp_limit = 1000  # Limit for batching temp data collection

    def collect_from_temp(self):
        """
        Transfers data from the temporary rows to the main dataframe and clears the temporary rows.
        This can boost our performance by around 10 to 170 times
        """
        if self.temp_rows:
            batch_df = pl.DataFrame(self.temp_rows, schema=self.df.schema)
            # Explicitly check if batch_df contains valid rows
            if batch_df.height > 0 and batch_df.null_count().sum_horizontal()[0] < (batch_df.height * batch_df.width):
                if self.df.height == 0:
                    self.df = batch_df
                else:
                    self.df = pl.concat([self.df, batch_df], how="vertical")
            # Clear temp_rows regardless of batch_df state
            self.temp_rows = []

    def collect(self, params: CollectParams):
        """
        This function collects the feedback of an analysis and stores in a dataframe.
        In this way, i.e., I can export a BIG experiment to CSV
        :param params: CollectParams object containing all input data.
        :return:
        """
        # Trigger flush when temp_limit is reached
        if len(self.temp_rows) >= self.temp_limit:
            self.collect_from_temp()

        records = {
            'scenario': params.scenario_provider.name,
            'experiment': params.experiment,
            'step': params.t,
            'policy': params.policy,
            'reward_function': params.reward_function,
            'sched_time': params.scenario_provider.avail_time_ratio,
            'sched_time_duration': params.available_time,
            'total_build_duration': params.total_build_duration,
            'prioritization_time': params.prioritization_time,
            'detected': params.metric.detected_failures,
            'missed': params.metric.undetected_failures,
            'tests_ran': len(params.metric.scheduled_testcases),
            'tests_not_ran': len(params.metric.unscheduled_testcases),
            'ttf': params.metric.ttf,
            'ttf_duration': params.metric.ttf_duration,
            'time_reduction': params.total_build_duration - params.metric.ttf_duration,
            'fitness': params.metric.fitness,
            'cost': params.metric.cost,
            'rewards': params.rewards,
            'avg_precision': params.metric.avg_precision,
            'prioritization_order': params.prioritization_order
        }

        self.temp_rows.append(records)

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
        if self.temp_rows:
            self.collect_from_temp()

        # Determine if the file already exists
        write_header = not os.path.exists(name) or os.stat(name).st_size == 0  # Empty file means headers are missing

        # Polars write_csv doesn't have mode parameter, so we need to handle appending manually
        if write_header or not os.path.exists(name):
            self.df.write_csv(name, separator=';', null_value='[]')
        else:
            # For appending, we write to temp and then append manually
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
                tmp_name = tmp.name
                self.df.write_csv(tmp_name, separator=';', null_value='[]', include_header=False)
            
            # Append the content
            with open(tmp_name, 'r') as tmp_file:
                content = tmp_file.read()
            with open(name, 'a') as f:
                f.write(content)
            os.unlink(tmp_name)

    def clear(self):
        """
        Clears the dataframe.
        """
        self.df = pl.DataFrame(schema=self.df.schema)
        self.temp_rows = []
