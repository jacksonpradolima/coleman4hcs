import numpy as np
import os
import pickle
import time
from pathlib import Path

from coleman4hcs.bandit import DynamicBandit, EvaluationMetricBandit
from coleman4hcs.scenarios import VirtualHCSScenario, IndustrialDatasetHCSScenarioProvider
from coleman4hcs.utils.monitor import MonitorCollector
from coleman4hcs.agent import ContextualAgent, SlidingWindowContextualAgent

Path("backup").mkdir(parents=True, exist_ok=True)


class Environment(object):
    """    
    The environment class that simulates the agent's interactions and collects results.
    """

    def __init__(self, agents, scenario_provider, evaluation_metric):
        self.agents = agents
        self.scenario_provider = scenario_provider
        self.evaluation_metric = evaluation_metric
        self.reset()

    def reset(self):
        """
        Reset the environment for a new simulation.
        """
        self.reset_agents_memory()
        # Monitor saves the feedback during the process
        self.monitor = MonitorCollector()

        self.variant_montitors = {}

        if isinstance(self.scenario_provider, IndustrialDatasetHCSScenarioProvider) and \
            self.scenario_provider.get_total_variants() > 0:
            for variant in self.scenario_provider.get_all_variants():
                self.variant_montitors[variant] = MonitorCollector()

    def reset_agents_memory(self):
        """
        # Resets the agent's memory to an initial state.
        """
        for agent in self.agents:
            agent.reset()

    def run_single(self,
                   experiment,
                   trials=100,
                   print_log=False,
                   bandit_type: DynamicBandit = EvaluationMetricBandit,
                   restore=True):
        """
        Execute a single simulation experiment
        :param experiment: Current Experiment
        :param trials: The max number of scenarios that will be analyzed
        :param print_log:
        :param bandit_type:
        :param restore: restore the experiment if fail (i.e., energy down)
        """
        # The agent need to learn from the beginning for each experiment
        self.reset_agents_memory()

        # create a bandit (initially is None to know that is the first bandit
        bandit = None

        # restore to step
        r_t = 1

        if restore:
            # Restore the experiment from the last backup
            # This is useful for long experiments
            r_t, self.agents, self.monitor, self.variant_montitors, bandit = self.load_experiment(experiment)
            self.scenario_provider.last_build(r_t)
            r_t += 1  # start 1 step after the last build

        # Test Budget percentage
        avail_time_ratio = self.scenario_provider.get_avail_time_ratio()

        # For each "virtual scenario (vsc)" I must analyse it and evaluate it
        for t, vsc in enumerate(self.scenario_provider, start=r_t):
            # The max number of scenarios that will be analyzed
            if t > trials:
                break

            # Time Budget
            available_time = vsc.get_available_time()
            self.evaluation_metric.update_available_time(available_time)

            # Compute time
            start_bandit = time.time()

            if bandit is None:
                bandit = bandit_type(vsc.get_testcases(), self.evaluation_metric)
            else:
                # Update bandit (the arms = TC available) for all agents
                bandit.update_arms(vsc.get_testcases())

            end_bandit = time.time()

            bandit_duration = end_bandit - start_bandit

            # we can analyse the same "moment/scenario t" for "i agents"
            for i, agent in enumerate(self.agents):
                # Update again because the variants
                # each variant has its own test budget size, that is, different from the whole system
                self.evaluation_metric.update_available_time(available_time)

                action, end, exp_name, start = self.run_prioritization(agent,
                                                                       bandit,
                                                                       bandit_duration,
                                                                       experiment,
                                                                       print_log,
                                                                       t,
                                                                       vsc)

                # If we are working with HCS scenario and there are variants
                if type(vsc) == VirtualHCSScenario and len(vsc.get_variants()) > 0:
                    self.run_prioritization_hcs(agent,
                                                action,
                                                avail_time_ratio,
                                                bandit_duration,
                                                end,
                                                exp_name,
                                                experiment,
                                                start,
                                                t,
                                                vsc)

            # Save experiment each 50000 builds
            if restore and t % 50000 == 0:
                self.save_experiment(experiment, t, bandit)

    def run_prioritization(self, agent, bandit, bandit_duration, experiment, print_log, t, vsc):
        # MAB or CMAB                
        if type(agent) in [ContextualAgent, SlidingWindowContextualAgent]:
            # For a Contextual agent, we first update the current context information
            agent.update_context(vsc.get_context_features())
            agent.update_features(vsc.get_features())

            exp_name = f"{str(agent)}_{vsc.get_feature_group()}"
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

        # if we want to see the performance during the experiments
        if print_log:
            print(
                f"Exp: {experiment} - Ep: {t} - Name: {exp_name} ({str(agent.get_reward_function())}) - " +
                f"NAPFD/APFDc: {metric.fitness:.4f}/{metric.cost:.4f}")

        # Collect the data during the experiment
        self.monitor.collect(self.scenario_provider,
                             vsc.get_available_time(),
                             experiment,
                             t,
                             exp_name,
                             str(agent.get_reward_function()),
                             metric,
                             self.scenario_provider.total_build_duration,
                             (end - start) + bandit_duration,
                             np.mean(agent.last_reward),
                             action)
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
                               vsc):

        # Get the variants that exist in the current commit
        variants = vsc.get_variants()

        # For each variant I will evaluate the impact of the main prioritization
        for variant in variants['Variant'].unique():
            # Get the data from current variant
            df = variants[variants.Variant == variant]

            # Order by the test cases according to the main prioritization
            df['CalcPrio'] = df['Name'].apply(lambda x: action.index(x) + 1)
            df.sort_values(by=['CalcPrio'], inplace=True)

            total_build_duration = df['Duration'].sum()
            total_time = total_build_duration * avail_time_ratio

            # Update the available time concerning the variant build duration
            self.evaluation_metric.update_available_time(total_time)

            # Submit prioritized test cases for evaluation step and get new measurements
            self.evaluation_metric.evaluate(df.to_dict(orient='records'))

            # Save the information (collect the data)
            self.variant_montitors[variant].collect(self.scenario_provider,
                                                    total_time,
                                                    experiment,
                                                    t,
                                                    exp_name,
                                                    str(agent.get_reward_function()),
                                                    self.evaluation_metric,
                                                    total_build_duration,
                                                    (end - start) + bandit_duration,
                                                    0,
                                                    df['Name'].tolist())

    def run(self, experiments=1, trials=100, print_log=False, bandit_type: DynamicBandit = EvaluationMetricBandit,
            restore=True):
        """
        Execute a simulation
        :param experiments: Number of experiments
        :param trials: The max number of scenarios that will be analyzed
        :param print_log:
        :param bandit_type:
        :param restore: restore the experiment if fail (i.e., energy down)
        :return:
        """
        self.reset()

        for exp in range(experiments):
            self.run_single(exp, trials, print_log, bandit_type, restore)

    def create_file(self, name):
        """
        Create a file to store the results obtained during the experiment
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
                    self.variant_montitors[variant].create_file(
                        f"{name}/{name.split('/')[-1].split('@')[0]}@{variant.replace('/', '-')}.csv")

    def store_experiment(self, name):
        """
        Save the results obtained during the experiment
        """
        self.monitor.save(name)

        if isinstance(self.scenario_provider, IndustrialDatasetHCSScenarioProvider):
            if self.scenario_provider.get_total_variants() > 0:
                # Ignore the extension
                name2 = name.split(".csv")[0]
                name2 = f"{name2}_variants"

                Path(name2).mkdir(parents=True, exist_ok=True)

                for variant in self.scenario_provider.get_all_variants():
                    self.variant_montitors[variant].save(
                        f"{name2}/{name.split('/')[-1].split('@')[0]}@{variant.replace('/', '-')}.csv")

    def load_experiment(self, experiment):
        """
        Load the backup
        """
        filename = f'backup/{str(self.scenario_provider)}_ex_{experiment}.p'

        if os.path.exists(filename):
            return pickle.load(open(filename, "rb"))

        return 0, self.agents, self.monitor, self.variant_montitors, None

    def save_experiment(self, experiment, t, bandit):
        """
        Save a backup for the experiment
        """
        filename = f'backup/{str(self.scenario_provider)}_ex_{experiment}.p'
        pickle.dump([t, self.agents, self.monitor,
                     self.variant_montitors, bandit], open(filename, "wb"))
