"""
main - Entry Point for Coleman4HCS.

This module provides the capabilities to run experiments for the Coleman4HCS framework.
It facilitates the execution of various scenarios and evaluations based on predefined
configurations.

The module offers:

- Experiment setups using configuration files.
- Ability to use various policies and reward functions.
- Parallel processing capabilities for experiment runs.
- Tools for dataset processing and scenario generation.
- Dynamic class loading for policies, reward functions, and agents.
- Logging and storage functionalities for experiment results.

Notes
-----
- Configuration files should be correctly set up.
- Required dependencies should be installed.
- Ensure all datasets are accessible and in the specified format.
- This module uses environment variables, loaded through ``dotenv``, to obtain
  specific configuration details.
"""
import logging
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import polars as pl
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

warnings.filterwarnings("ignore")


# taken from https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python
def create_logger(level):
    """Create and configure a logger for multiprocessing-safe logging.

    Parameters
    ----------
    level : int
        The logging level (e.g., ``logging.DEBUG``, ``logging.INFO``).

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # this bit will make sure you won't have duplicated messages in the output
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def exp_run_industrial_dataset(iteration, trials, env: Environment, experiment_directory, level):
    """Execute a single run of the industrial dataset experiment.

    Parameters
    ----------
    iteration : int
        The current iteration of the experiment.
    trials : int
        The total number of trials to be executed.
    env : Environment
        An instance of the environment where the experiment is run.
    experiment_directory : str
        The directory where experiment results will be stored.
    level : int
        The logging level.
    """
    csv_file_name = f"{experiment_directory}{str(env.scenario_provider)}_{iteration}.csv"
    env.logger = create_logger(level)
    env.create_file(csv_file_name)
    env.run_single(iteration, trials)
    env.store_experiment(csv_file_name)


def load_class_from_module(module, class_name: str):
    """Dynamically load a class from a given module.

    Parameters
    ----------
    module : module
        The Python module from which to load the class.
    class_name : str
        The name of the class to be loaded.

    Returns
    -------
    type
        The loaded class.

    Raises
    ------
    ValueError
        If the class is not found in the provided module.
    """

    if hasattr(module, class_name):
        return getattr(module, class_name)
    raise ValueError(f"Class '{class_name}' not found in {module.__name__}!")


def create_agents(policy, rew_fun, window_sizes):
    """Create agent instances based on the policy type.

    Parameters
    ----------
    policy : object
        The policy instance.
    rew_fun : object
        The reward function instance.
    window_sizes : list
        List of window sizes (only relevant for policies that use Sliding
        Window such as FRRMABPolicy).

    Returns
    -------
    list
        A list of agent instances.
    """

    if isinstance(policy, FRRMABPolicy):
        return [RewardSlidingWindowAgent(policy, rew_fun, w) for w in window_sizes]

    if isinstance(policy, SWLinUCBPolicy):
        return [SlidingWindowContextualAgent(policy, rew_fun, w) for w in window_sizes]

    if isinstance(policy, LinUCBPolicy):
        return [ContextualAgent(policy, rew_fun)]

    return [RewardAgent(policy, rew_fun)]


def get_scenario_provider(  # pylint: disable=too-many-positional-arguments
        datasets_dir,
        dataset,
        sched_time_ratio,
        use_hcs,
        use_context,
        context_config,
        feature_groups):
    """Return the appropriate scenario provider based on the given configuration.

    The function selects the scenario provider based on whether the
    HCS (Highly-Configurable System) configuration is used. It constructs the
    appropriate paths for the dataset files and initializes the scenario
    provider with these paths.

    Parameters
    ----------
    datasets_dir : str
        The directory where datasets are stored.
    dataset : str
        The specific dataset to be used.
    sched_time_ratio : float
        The ratio of scheduled time to be used in the scenario.
    use_hcs : bool
        If True, returns an ``IndustrialDatasetHCSScenarioProvider`` instance.
    use_context : bool
        If True, returns an ``IndustrialDatasetContextScenarioProvider`` instance.
    context_config : dict
        Configuration for contextual information.
    feature_groups : dict
        Feature group configuration.

    Returns
    -------
    IndustrialDatasetScenarioProvider or IndustrialDatasetHCSScenarioProvider or IndustrialDatasetContextScenarioProvider
        An instance of the scenario provider based on the given configuration.
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
    """Merge multiple CSV files into a single CSV file.

    Reads a list of CSV files, concatenates their contents into a single
    DataFrame, and writes the combined data to a new CSV file. Also cleans up
    the temporary files after merging.

    Parameters
    ----------
    files : list of str
        A list of file paths to the CSV files to be merged.
    output_file : str
        The file path for the output CSV file where the merged data will be
        saved.
    """
    # Merge all CSV files into one DataFrame
    dfs = [pl.read_csv(file, separator=';') for file in files]
    df = pl.concat(dfs, how="vertical")

    # Save the merged DataFrame to CSV
    df.write_csv(output_file, separator=';', quote_style="never")

    # Optionally, clean up temporary files
    for file in files:
        os.remove(file)

def store_experiments(csv_file, scenario):
    """Store experiment results from a CSV file into a DuckDB database.

    Reads experiment results from a given CSV file and inserts them into a
    DuckDB database table named 'experiments'. If the table does not exist, it
    is created with a predefined schema.

    Parameters
    ----------
    csv_file : str
        The path to the CSV file containing experiment results.
    scenario : object
        The scenario object associated with the experiment results. Used to
        determine if variant-specific handling is needed.

    Notes
    -----
    The database connection is hardcoded to 'experiments.db'. This function
    will create or open this database file in the current working directory.
    The CSV file is expected to have a header row and use ';' as the
    delimiter.
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

    df = conn.read_csv(csv_file, delimiter=';', quotechar='"', header=True)  # pylint: disable=unused-variable

    # Insert the DataFrame into the 'experiments' table
    conn.execute("INSERT INTO experiments SELECT * FROM df;")

    if isinstance(scenario, IndustrialDatasetHCSScenarioProvider):
        if scenario.get_total_variants() > 0:
            # Ignore the extension
            name2 = csv_file.split(".csv")[0]
            name2 = f"{name2}_variants"

            Path(name2).mkdir(parents=True, exist_ok=True)

            for variant in scenario.get_all_variants():
                csv_file_variant = (
                    f"{name2}/{csv_file.split('/')[-1].split('@')[0]}"
                    f"@{variant.replace('/', '-')}.csv"
                )
                df = conn.read_csv(csv_file_variant, delimiter=';', quotechar='"', header=True)

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
            csv_file_names = [
                f"{experiment_directory}{str(env.scenario_provider)}_{i+1}.csv"
                for i in range(independent_executions)
            ]
            csv_file = f"{experiment_directory}{str(env.scenario_provider)}.csv"
            merge_csv(csv_file_names, csv_file)

            # Store the results in the duckdb database
            store_experiments(csv_file, scenario)

            logging.info("Time expend to run the experiments: %s\n\n", end - start)
