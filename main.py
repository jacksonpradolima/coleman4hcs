import os
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

from dotenv import load_dotenv

import coleman4hcs.policy
import coleman4hcs.reward
from coleman4hcs.agent import (
    RewardAgent,
    RewardSlidingWindowAgent,
    ContextualAgent,
    SlidingWindowContextualAgent
)
from coleman4hcs.environment import Environment
from coleman4hcs.evaluation import NAPFDVerdictMetric
from coleman4hcs.policy import FRRMABPolicy, SWLinUCBPolicy, LinUCBPolicy
from coleman4hcs.scenarios import (
    IndustrialDatasetHCSScenarioProvider,
    IndustrialDatasetScenarioProvider,
    IndustrialDatasetContextScenarioProvider
)
from config.config import get_config

warnings.filterwarnings("ignore")


def exp_run_industrial_dataset(iteration, trials, env: Environment, experiment_directory):
    """
    Execute a single run of the industrial dataset experiment.

    :param iteration: The current iteration of the experiment.
    :type iteration: int
    :param trials: The total number of trials to be executed.
    :type trials: int
    :param env: An instance of the environment where the experiment is run.
    :type env: Environment
    :param experiment_directory: The directory where experiment results will be stored.
    :type experiment_directory: str
    :return: None
    """

    env.run_single(iteration, trials, print_log=True)
    env.store_experiment(f"{experiment_directory}{str(env.scenario_provider)}.csv")


def load_class_from_module(module, class_name: str):
    """
    Dynamically loads a class from a given module.

    :param module: The Python module from which to load the class.
    :type module: module
    :param class_name: The name of the class to be loaded.
    :type class_name: str
    :return: The loaded class.
    :rtype: class
    :raises ValueError: If the class is not found in the provided module.
    """

    if hasattr(module, class_name):
        return getattr(module, class_name)
    else:
        raise ValueError(f"Class '{class_name}' not found in {module.__name__}!")


def create_agents(policy, rew_fun, window_sizes):
    """
    Create agent instances based on the policy type.

    :param policy: The policy instance.
    :type policy: Policy
    :param rew_fun: The reward function instance.
    :type rew_fun: RewardFunction
    :param window_sizes: List of window sizes (only relevant for policies
                         that use Sliding Window such as FRRMABPolicy).
    :type window_sizes: list

    :return: A list of agent instances.
    :rtype: list
    """

    if isinstance(policy, FRRMABPolicy):
        return [RewardSlidingWindowAgent(policy, rew_fun, w) for w in window_sizes]
    elif isinstance(policy, SWLinUCBPolicy):
        return [SlidingWindowContextualAgent(policy, rew_fun, w) for w in window_sizes]
    elif isinstance(policy, LinUCBPolicy):
        return [ContextualAgent(policy, rew_fun)]
    return [RewardAgent(policy, rew_fun)]


def get_scenario_provider(datasets_dir,
                          dataset,
                          sched_time_ratio,
                          use_hcs,
                          use_context,
                          context_config,
                          feature_groups):
    """
    Return the appropriate scenario provider based on the given configuration.

    The function selects the scenario provider based on whether the
    HCS (Highly-Configurable System) configuration is used.
    It constructs the appropriate paths for the
    dataset files and initializes the scenario provider with these paths.

    :param datasets_dir: The directory where datasets are stored.
    :type datasets_dir: str
    :param dataset: The specific dataset to be used.
    :type dataset: str
    :param sched_time_ratio: The ratio of scheduled time to be used in the scenario.
    :type sched_time_ratio: float
    :param use_hcs: If True, returns an `IndustrialDatasetHCSScenarioProvider` instance.
                    Otherwise, returns an `IndustrialDatasetScenarioProvider` instance.
    :type use_hcs: bool

    :return: An instance of the scenario provider based on the given configuration.
    :rtype: IndustrialDatasetScenarioProvider or IndustrialDatasetHCSScenarioProvider
    """
    base_args = [f"{datasets_dir}/{dataset}/features-engineered.csv", sched_time_ratio]

    if use_hcs and not use_context:
        scenario_cls = IndustrialDatasetHCSScenarioProvider
        base_args.insert(1, f"{datasets_dir}/{dataset}/data-variants.csv")
    elif not use_hcs and use_context:
        scenario_cls = IndustrialDatasetContextScenarioProvider
        base_args[0] = f"{datasets_dir}/{dataset}/features-engineered-contextual.csv"
        base_args.insert(1, feature_groups['feature_group_name'])
        base_args.insert(2, feature_groups['feature_group_values'])
        base_args.insert(3, context_config['previous_build'])
    elif use_hcs and use_context:
        raise NotImplementedError
    else:
        scenario_cls = IndustrialDatasetScenarioProvider

    return scenario_cls(*base_args)


if __name__ == '__main__':
    load_dotenv()
    config = get_config()

    # Execution configuration
    (
        parallel_pool_size,
        independent_executions
    ) = map(config['execution'].get, [
        'parallel_pool_size',
        'independent_executions'
    ])

    # Experiment Configuration
    (
        sched_time_ratio,
        datasets_dir,
        datasets,
        experiment_dir,
        rewards_names,
        policy_names
    ) = map(config['experiment'].get, [
        'scheduled_time_ratio',
        'datasets_dir',
        'datasets',
        'experiment_dir',
        'rewards',
        'policies'
    ])

    # Algorithms Configuration
    algorithm_configs = config['algorithm']

    # HCS Configuration
    use_hcs = config.get('hcs_configuration', False).get('wts_strategy', False)

    # Contextual Information
    (
        context_config,
        feature_groups
    ) = map(config['contextual_information'].get, [
        'config',
        'feature_group'
    ])

    # Load policy objects along with the target reward functions
    policies = {
        policy_name: {
            reward_name: load_class_from_module(coleman4hcs.policy, policy_name + "Policy")(
                **algorithm_configs.get(policy_name.lower(), {}).get(reward_name.lower(), {})
            )
            for reward_name in rewards_names
        }
        for policy_name in policy_names
    }

    # Generate agents based on the policies and reward functions
    agents = [
        agent
        for policy_name, reward_policies in policies.items()
        for reward_name, policy in reward_policies.items()
        for agent in create_agents(
            policy,
            load_class_from_module(coleman4hcs.reward, reward_name + "Reward")(),
            algorithm_configs.get(policy_name.lower(), {}).get('window_sizes', [])
        )
    ]

    # Check if there's an agent of type SlidingWindowContextualAgent or ContextualAgent
    has_sliding_window_contextual_agent = any(isinstance(agent, SlidingWindowContextualAgent) for agent in agents)
    has_contextual_agent = any(isinstance(agent, ContextualAgent) for agent in agents)

    use_context = has_contextual_agent or has_sliding_window_contextual_agent

    evaluation_metric = NAPFDVerdictMetric()

    for tr in sched_time_ratio:
        experiment_directory = os.path.join(experiment_dir, f"time_ratio_{int(tr * 100)}/")

        Path(experiment_directory).mkdir(parents=True, exist_ok=True)

        for dataset in datasets:
            scenario = get_scenario_provider(datasets_dir, dataset, tr, use_hcs, use_context, context_config,
                                             feature_groups)

            # Stop conditional
            trials = scenario.max_builds

            # Prepare the experiment
            env = Environment(agents, scenario, evaluation_metric)

            parameters = [(i + 1, trials, env, experiment_directory) for i in range(independent_executions)]

            # create a file with a unique header for the scenario (workaround)
            env.create_file(f"{experiment_directory}{str(env.scenario_provider)}.csv")

            # Compute time
            start = time.time()

            if parallel_pool_size > 1:
                with Pool(parallel_pool_size) as p:
                    p.starmap(exp_run_industrial_dataset, parameters)
            else:
                for param in parameters:
                    exp_run_industrial_dataset(*param)

            end = time.time()

            print(f"Time expend to run the experiments: {end - start}\n\n")
