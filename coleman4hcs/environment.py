"""
coleman4hcs.environment
~~~~~~~~~~~~~~~~~~~~~~~

This module contains the `Environment` class which simulates the agent's interactions and collects results
during the reinforcement learning process. The environment provides a platform where agents learn from interactions
with the bandit to make better decisions over time.

The module also contains the following features:

- Mechanism to reset the environment and agent's memories.
- Ability to run a single experiment or multiple experiments.
- Support for both simple and contextual agents.
- Periodic saving of experiments to handle long-running experiments.
- Facilities for creating and storing results obtained during experiments.
- Support for scenarios with variants, commonly found in Heterogeneous Computing Systems (HCS).
- Helper methods for loading and saving experiment states for recovery purposes.

Classes:
    - Environment: Represents the learning environment where agents interact with bandits.
"""
import logging
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from coleman4hcs.agent import ContextualAgent, SlidingWindowContextualAgent
from coleman4hcs.bandit import EvaluationMetricBandit
from coleman4hcs.scenarios import VirtualHCSScenario, IndustrialDatasetHCSScenarioProvider
from coleman4hcs.utils.monitor import MonitorCollector
from coleman4hcs.utils.monitor_params import CollectParams

Path("backup").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

warnings.simplefilter(action='ignore', category=FutureWarning)


class Environment:
    """
    The environment class that simulates the agent's interactions and collects results.
    """

    def __init__(self, agents, scenario_provider, evaluation_metric):
        self.agents = agents
        self.scenario_provider = scenario_provider
        self.evaluation_metric = evaluation_metric
        self.monitor = None
        self.variant_monitors = {}
        self.reset()

    def reset(self):
        """
        Reset the environment for a new simulation.
        """
        self.reset_agents_memory()
        # Monitor saves the feedback during the process
        self.monitor = MonitorCollector()
        self.variant_monitors = {}

        if isinstance(self.scenario_provider, IndustrialDatasetHCSScenarioProvider) and \
            self.scenario_provider.get_total_variants() > 0:
            for variant in self.scenario_provider.get_all_variants():
                self.variant_monitors[variant] = MonitorCollector()

    def reset_agents_memory(self):
        """
        Resets the agent's memory to an initial state.
        """
        for agent in self.agents:
            agent.reset()

    def run_single(self,
                   experiment,
                   trials=100,
                   bandit_type=EvaluationMetricBandit,
                   restore=True):
        """
        Execute a single simulation experiment

        :param experiment: Current Experiment
        :param trials: The max number of scenarios that will be analyzed
        :param bandit_type:
        :param restore: restore the experiment if fail (i.e., energy down)
        """
        # The agent need to learn from the beginning for each experiment
        self.reset_agents_memory()

        # create a bandit (initially is None to know that is the first bandit
        bandit = None

        # restore to step
        restore_step = 1

        if restore:
            # Restore the experiment from the last backup
            # This is useful for long experiments
            restore_step, self.agents, self.monitor, self.variant_monitors, bandit = self.load_experiment(experiment)
            self.scenario_provider.last_build(restore_step)
            restore_step += 1  # start 1 step after the last build

        # Test Budget percentage
        avail_time_ratio = self.scenario_provider.get_avail_time_ratio()

        # For each "virtual scenario" I must analyse it and evaluate it
        for t, virtual_scenario in enumerate(self.scenario_provider, start=restore_step):
            # The max number of scenarios that will be analyzed
            if t > trials:
                break

            # Time Budget
            available_time = virtual_scenario.get_available_time()
            self.evaluation_metric.update_available_time(available_time)

            # Compute time
            start_bandit = time.time()

            if bandit is None:
                bandit = bandit_type(virtual_scenario.get_testcases(), self.evaluation_metric)
            else:
                # Update bandit (the arms = TC available) for all agents
                bandit.update_arms(virtual_scenario.get_testcases())

            end_bandit = time.time()

            bandit_duration = end_bandit - start_bandit

            # we can analyse the same "moment/scenario t" for "i agents"
            for _, agent in enumerate(self.agents):
                # Update again because the variants
                # each variant has its own test budget size, that is, different from the whole system
                self.evaluation_metric.update_available_time(available_time)

                action, end, exp_name, start = self.run_prioritization(agent,
                                                                       bandit,
                                                                       bandit_duration,
                                                                       experiment,
                                                                       t,
                                                                       virtual_scenario)

                # If we are working with HCS scenario and there are variants
                if isinstance(virtual_scenario, VirtualHCSScenario) and len(virtual_scenario.get_variants()) > 0:
                    self.run_prioritization_hcs(agent,
                                                action,
                                                avail_time_ratio,
                                                bandit_duration,
                                                end,
                                                exp_name,
                                                experiment,
                                                start,
                                                t,
                                                virtual_scenario)

            self.save_periodically(restore, t, experiment, bandit)

    def run_prioritization(self, agent, bandit, bandit_duration, experiment, t, virtual_scenario):
        """
        Run the prioritization process for a given agent and scenario.

        :param agent: The agent that is being used for the prioritization.
        :param bandit: The bandit mechanism used for choosing actions.
        :param bandit_duration: Time taken by the bandit process.
        :param experiment: The current experiment number.
        :param t: The current step or iteration of the simulation.
        :param virtual_scenario: The virtual scenario being considered.
        :return: tuple containing the chosen action by the agent, the ending time of the process,
                 the name of the experiment, and the starting time of the process.
        """
        # MAB or CMAB
        if type(agent) in [ContextualAgent, SlidingWindowContextualAgent]:
            # For a Contextual agent, we first update the current context information
            agent.update_context(virtual_scenario.get_context_features())
            agent.update_features(virtual_scenario.get_features())

            exp_name = f"{str(agent)}_{virtual_scenario.get_feature_group()}"
        else:
            exp_name = str(agent)

        # Update the bandit inside the agent.
        # This loop also update the actions available for policy choose
        agent.update_bandit(bandit)

        # Compute time
        start = time.time()

        # Choose action (Prioritized Test Suite List) from agent (from current Q estimate)
        action = agent.choose()

        # Pick up reward from bandit for chosen action
        # Submit prioritized test cases for evaluation step the environment and get new measurements
        metric = agent.bandit.pull(action)

        # Update Q action-value estimates (Reward Functions)
        agent.observe(metric)

        # Compute end time
        end = time.time()

        logger.debug(f"Exp: {experiment} - Ep: {t} - Name: {exp_name} ({str(agent.get_reward_function())}) - " +
                     f"NAPFD/APFDc: {metric.fitness:.4f}/{metric.cost:.4f}")

        # Collect the data during the experiment
        params = CollectParams(
            scenario_provider=self.scenario_provider,
            available_time=virtual_scenario.get_available_time(),
            experiment=experiment,
            t=t,
            policy=exp_name,
            reward_function=str(agent.get_reward_function()),
            metric=metric,
            total_build_duration=self.scenario_provider.total_build_duration,
            prioritization_time=(end - start) + bandit_duration,
            rewards=np.mean(agent.last_reward),
            prioritization_order=action,
        )

        self.monitor.collect(params)

        return action, end, exp_name, start

    def run_prioritization_hcs(self,
                               agent,
                               action,
                               avail_time_ratio,
                               bandit_duration,
                               end,
                               exp_name,
                               experiment,
                               start,
                               t,
                               virtual_scenario):

        """
        Run the prioritization process for a given agent and HCS scenario.

        :param agent: The agent that is being used for the prioritization.
        :param action: The chosen action by the agent.
        :param avail_time_ratio: The available time ratio for the experiment.
        :param bandit_duration: Time taken by the bandit process.
        :param end: The ending time of the process.
        :param exp_name: The name of the experiment.
        :param experiment: The current experiment number.
        :param start: The starting time of the process.
        :param t: The current step or iteration of the simulation.
        :param virtual_scenario: The virtual HCS scenario being considered.
        """

        # Get the variants that exist in the current commit
        variants = virtual_scenario.get_variants()

        # For each variant I will evaluate the impact of the main prioritization
        for variant in variants['Variant'].unique():
            # Get the data from current variant
            df = variants.filter(pl.col('Variant') == variant)

            # Order by the test cases according to the main prioritization
            action_map = {name: idx + 1 for idx, name in enumerate(action)}
            df = df.with_columns([
                pl.col('Name').replace(action_map, default=0).alias('CalcPrio')
            ])
            df = df.sort('CalcPrio')

            total_build_duration = df['Duration'].sum()
            total_time = total_build_duration * avail_time_ratio

            # Update the available time concerning the variant build duration
            self.evaluation_metric.update_available_time(total_time)

            # Submit prioritized test cases for evaluation step and get new measurements
            self.evaluation_metric.evaluate(df.to_dicts())

            # Save the information (collect the data)
            params = CollectParams(
                scenario_provider=self.scenario_provider,
                available_time=total_time,
                experiment=experiment,
                t=t,
                policy=exp_name,
                reward_function=str(agent.get_reward_function()),
                metric=self.evaluation_metric,
                total_build_duration=total_build_duration,
                prioritization_time=(end - start) + bandit_duration,
                rewards=0,
                prioritization_order=df['Name'].to_list(),
            )

            self.variant_monitors[variant].collect(params)

    def save_periodically(self, restore, t, experiment, bandit, interval=50000):
        """
       Save the experiment periodically based on a predefined interval.

       :param restore: Flag to indicate if the experiment should be restored.
       :param  t: The current step or iteration of the simulation.
       :param  experiment: The current experiment number.
       :param  bandit: The current bandit being used in the simulation.
       :param  interval: The interval at which the experiment should be saved.
       """
        # Save experiment each X builds
        if restore and t % interval == 0:
            self.save_experiment(experiment, t, bandit)

    def run(self, experiments=1, trials=100, bandit_type=EvaluationMetricBandit,
            restore=True):
        """
        Execute a simulation

        :param experiments: Number of experiments
        :param trials: The max number of scenarios that will be analyzed
        :param bandit_type:
        :param restore: restore the experiment if fail (i.e., energy down)
        :return:
        """
        self.reset()

        for exp in range(experiments):
            self.run_single(exp, trials, bandit_type, restore)

    def create_file(self, name):
        """
        Create a file to store the results obtained during the experiment

        :param name: The name of the file to store the results.
        """
        self.monitor.create_file(name)

        # If we are working with HCS scenario, we create a file for each variant in a specific directory
        if isinstance(self.scenario_provider, IndustrialDatasetHCSScenarioProvider):
            if self.scenario_provider.get_total_variants() > 0:
                # Ignore the extension
                name = name.split(".csv")[0]
                name = f"{name}_variants"

                Path(name).mkdir(parents=True, exist_ok=True)

                for variant in self.scenario_provider.get_all_variants():
                    self.variant_monitors[variant].create_file(
                        f"{name}/{name.split('/')[-1].split('@')[0]}@{variant.replace('/', '-')}.csv")

    def store_experiment(self, csv_file_name):
        """
        Save the results obtained during the experiment

        :param csv_file_name: The name of the file to store the results.
        """
        # Collect from temp and save a file (backup and easy sharing/auditing)
        self.monitor.save(csv_file_name)

        if isinstance(self.scenario_provider, IndustrialDatasetHCSScenarioProvider):
            if self.scenario_provider.get_total_variants() > 0:
                # Ignore the extension
                name2 = csv_file_name.split(".csv")[0]
                name2 = f"{name2}_variants"

                Path(name2).mkdir(parents=True, exist_ok=True)

                for variant in self.scenario_provider.get_all_variants():
                    # Collect from temp and save a file (backup and easy sharing/auditing)
                    self.variant_monitors[variant].save(
                        f"{name2}/{csv_file_name.split('/')[-1].split('@')[0]}@{variant.replace('/', '-')}.csv")

        # Clear experiment
        self.monitor.clear()

    def load_experiment(self, experiment):
        """
        Load the backup

        :param experiment: The current experiment number.
        """
        filename = f'backup/{str(self.scenario_provider)}_ex_{experiment}.p'

        if not os.path.exists(filename):
            return 0, self.agents, self.monitor, self.variant_monitors, None

        with open(filename, "rb") as file:
            return pickle.load(file)

    def save_experiment(self, experiment, t, bandit):
        """
        Save a backup for the experiment

        :param experiment: The current experiment number.
        :param t: The current step or iteration of the simulation.
        :param bandit: The current bandit being used in the simulation.
        """
        try:
            filename = f'backup/{str(self.scenario_provider)}_ex_{experiment}.p'
            with open(filename, "wb") as f:
                pickle.dump([t, self.agents, self.monitor, self.variant_monitors, bandit], f)
        except Exception as e:
            logger.error(f"Error saving the experiment: {e}")
            raise e
