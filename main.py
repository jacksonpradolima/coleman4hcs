"""
`main` - Entry Point for Coleman4HCS

This module provides the capabilities to run experiments for the Coleman4HCS framework.
It facilitates the execution of various scenarios and evaluations based on predefined configurations.

The module offers:
- Experiment setups using configuration files.
- Ability to use various policies and reward functions.
- Parallel processing capabilities for experiment runs.
- Tools for dataset processing and scenario generation.
- Dynamic class loading for policies, reward functions, and agents.
- Logging and storage functionalities for experiment results.

Usage:
    python main.py

Preconditions:
- Configuration files should be correctly set up.
- Required dependencies should be installed.
- Ensure all datasets are accessible and in the specified format.

Environment Variables:
This module uses environment variables, loaded through `dotenv`, to obtain specific configuration details.

Author:
    Jackson Antonio do Prado Lima - jacksonpradolima at gmail.com

"""
import csv
import logging
import os
import time
import warnings
import polars as pl
from multiprocessing import Pool
from pathlib import Path

from dotenv import load_dotenv

import duckdb

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
import sys

warnings.filterwarnings("ignore")

# taken from https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python
def create_logger(level):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # this bit will make sure you won't have 
    # duplicated messages in the output
    if not len(logger.handlers): 
        logger.addHandler(handler)
    return logger


def exp_run_industrial_dataset(iteration, trials, env: Environment, experiment_directory, level):
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
    csv_file_name = f"{experiment_directory}{str(env.scenario_provider)}_{iteration}.csv"
    env.logger = create_logger(level)
    env.create_file(csv_file_name)
    env.run_single(iteration, trials)
    env.store_experiment(csv_file_name)


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

    if isinstance(policy, SWLinUCBPolicy):
        return [SlidingWindowContextualAgent(policy, rew_fun, w) for w in window_sizes]

    if isinstance(policy, LinUCBPolicy):
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

def merge_csv(files, output_file):
    """
    Merges multiple CSV files into a single CSV file.

    This function reads a list of CSV files, concatenates their contents into a single
    pandas DataFrame, and then writes the combined data to a new CSV file. It is designed
    to be used after parallel processing tasks where each task writes its output to a
    separate file. This function also optionally cleans up the temporary files after
    merging.

    Parameters:
    - files (list of str): A list of file paths to the CSV files to be merged.
    - output_file (str): The file path for the output CSV file where the merged data
      will be saved.

    Returns:
    - None: This function does not return a value but writes the merged data to a CSV file.

    Example:
    >>> temp_files = ["data_part1.csv", "data_part2.csv", "data_part3.csv"]
    >>> merge_csv(temp_files, "merged_data.csv")
    """
    # Merge all CSV files into one DataFrame
    dfs = [pl.read_csv(file, separator=';') for file in files]
    df = pl.concat(dfs, how="vertical")

    # Save the merged DataFrame to CSV
    df.write_csv(output_file, separator=';')

    # Optionally, clean up temporary files
    for file in files:
        os.remove(file)

def store_experiments(csv_file, scenario):
    """
    Stores experiment results from a CSV file into a DuckDB database.

    This function reads experiment results from a given CSV file and inserts them into
    a DuckDB database table named 'experiments'. If the table does not exist, it is created
    with a predefined schema. This function is designed to work with both standard experiment
    results and variant-specific results when used with a scenario that supports variants,
    such as an IndustrialDatasetHCSScenarioProvider instance.

    Parameters:
    - csv_file (str): The path to the CSV file containing experiment results.
    - scenario (IndustrialDatasetHCSScenarioProvider or similar): The scenario object
      associated with the experiment results. This object is used to determine if variant-specific
      handling is needed.

    The function checks if the scenario has variants and, if so, creates separate directories
    for each variant's results. It then reads each variant-specific CSV file and inserts its
    contents into the 'experiments' table in the database.

    Note:
    - The database connection is hardcoded to 'experiments.db'. This function will create or
      open this database file in the current working directory.
    - The CSV file is expected to have a header row and use ';' as the delimiter.

    Example usage:
    >>> store_experiments("experiment_results.csv", my_scenario)
    """
    # Create/Open a database to store the results
    conn = duckdb.connect('experiments.db')

    # Ensure the tables exist with the appropriate schema
    conn.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        scenario VARCHAR,
        experiment_id INTEGER,
        step INTEGER,
        policy VARCHAR,
        reward_function VARCHAR,
        sched_time FLOAT,
        sched_time_duration FLOAT,
        total_build_duration FLOAT,
        prioritization_time FLOAT,
        detected INTEGER,
        missed INTEGER,
        tests_ran INTEGER,
        tests_not_ran INTEGER,
        ttf FLOAT,
        ttf_duration FLOAT,
        time_reduction FLOAT,
        fitness FLOAT,
        cost FLOAT,
        rewards FLOAT,
        avg_precision FLOAT,
        prioritization_order VARCHAR
    );
    """)

    df = conn.read_csv(csv_file, delimiter=';', quotechar='"', header=True)

    # Insert the DataFrame into the 'experiments' table
    conn.execute("INSERT INTO experiments SELECT * FROM df;")

    if isinstance(scenario, IndustrialDatasetHCSScenarioProvider):
        if scenario.get_total_variants() > 0:
            # Ignore the extension
            name2 = csv_file.split(".csv")[0]
            name2 = f"{name2}_variants"

            Path(name2).mkdir(parents=True, exist_ok=True)

            for variant in scenario.get_all_variants():
                csv_file_variant = f"{name2}/{csv_file.split('/')[-1].split('@')[0]}@{variant.replace('/', '-')}.csv"
                conn.execute( f"COPY experiments FROM '{csv_file_variant}' (HEADER);")
                df = conn.read_csv(csv_file, delimiter=';', quotechar='"', header=True)

                # Insert the DataFrame into the 'experiments' table
                conn.execute("INSERT INTO experiments SELECT * FROM df;")



if __name__ == '__main__':
    load_dotenv()
    config = get_config()

    # Execution configuration
    (
        parallel_pool_size,
        independent_executions,
        verbose
    ) = map(config['execution'].get, [
        'parallel_pool_size',
        'independent_executions',
        'verbose'
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

    logger = None
    level = logging.NOTSET

    if verbose:
        level = logging.DEBUG
        logger = create_logger(level)
    else:
        level = logging.INFO
        logger = create_logger(level)

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

            parameters = [(i + 1, trials, env, experiment_directory, level) for i in range(independent_executions)]

            # Compute time
            start = time.time()

            if parallel_pool_size > 1:
                with Pool(parallel_pool_size) as p:
                    p.starmap(exp_run_industrial_dataset, parameters)
            else:
                for param in parameters:
                    exp_run_industrial_dataset(*param)

            end = time.time()

            # Read and merge the independent executions
            csv_file_names = [f"{experiment_directory}{str(env.scenario_provider)}_{i+1}.csv" for i in range(independent_executions)]
            csv_file = f"{experiment_directory}{str(env.scenario_provider)}.csv"
            merge_csv(csv_file_names, csv_file)

            # Store the results in the duckdb database
            store_experiments(csv_file, scenario)

            logging.info(f"Time expend to run the experiments: {end - start}\n\n")
