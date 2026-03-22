"""
main - Entry Point for Coleman4HCS.

This module provides the capabilities to run experiments for the Coleman4HCS framework.
It facilitates the execution of various scenarios and evaluations based on predefined
configurations.

The module offers:

- Experiment setups using configuration files.
- Ability to use various policies and reward functions.
- Parallel processing capabilities for experiment runs.
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
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing import TimeoutError, get_context
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

import coleman4hcs.policy
import coleman4hcs.reward
from coleman4hcs.agent import ContextualAgent, RewardAgent, RewardSlidingWindowAgent, SlidingWindowContextualAgent
from coleman4hcs.environment import Environment
from coleman4hcs.evaluation import NAPFDVerdictMetric
from coleman4hcs.policy import FRRMABPolicy, LinUCBPolicy, SWLinUCBPolicy
from coleman4hcs.scenarios import (
    IndustrialDatasetContextScenarioProvider,
    IndustrialDatasetHCSScenarioProvider,
    IndustrialDatasetScenarioProvider,
)
from config.config import get_config

warnings.filterwarnings("ignore")


def to_builtin_types(value: Any) -> Any:
    """Recursively convert mapping/sequence wrappers into builtin serializable types."""
    if isinstance(value, Mapping):
        return {str(key): to_builtin_types(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_builtin_types(item) for item in value]

    if isinstance(value, tuple):
        return [to_builtin_types(item) for item in value]

    return value


@dataclass(frozen=True)
class ExecutionPlan:
    """Serializable worker plan for one independent execution."""

    iteration: int
    trials: int
    level: int
    execution_id: str
    worker_id: str
    parallel_mode: str


@dataclass(frozen=True)
class EnvironmentBuildConfig:
    """Serializable configuration required to build an isolated Environment."""

    datasets_dir: str
    dataset: str
    sched_time_ratio: float
    use_hcs: bool
    use_context: bool
    context_config: dict[str, Any]
    feature_groups: dict[str, Any]
    results_config: dict[str, Any]
    checkpoint_config: dict[str, Any]
    telemetry_config: dict[str, Any]
    algorithm_configs: dict[str, Any]
    rewards_names: list[str]
    policy_names: list[str]


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
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # this bit will make sure you won't have duplicated messages in the output
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def exp_run_industrial_dataset(
    iteration: int,
    trials: int,
    env: Environment,
    level: int,
    runtime_metadata: dict[str, str] | None = None,
) -> None:
    """Execute a single run of the industrial dataset experiment.

    Parameters
    ----------
    iteration : int
        The current iteration of the experiment.
    trials : int
        The total number of trials to be executed.
    env : Environment
        An instance of the environment where the experiment is run.
    level : int
        The logging level.
    runtime_metadata : dict[str, str] or None
        Execution-scoped metadata attached to telemetry and persisted results.
    """
    # Initialize logging for worker processes without mutating Environment state.
    create_logger(level)
    env.set_runtime_metadata(runtime_metadata)
    env.run_single(iteration, trials)
    env.store_experiment()


def build_agents_from_config(
    algorithm_configs: dict[str, Any], policy_names: list[str], rewards_names: list[str]
) -> list[RewardAgent | RewardSlidingWindowAgent | ContextualAgent | SlidingWindowContextualAgent]:
    """Build all agents from config values in a process-local way."""
    policies = {
        policy_name: {
            reward_name: load_class_from_module(coleman4hcs.policy, policy_name + "Policy")(
                **algorithm_configs.get(policy_name.lower(), {}).get(reward_name.lower(), {})
            )
            for reward_name in rewards_names
        }
        for policy_name in policy_names
    }

    return [
        agent
        for policy_name, reward_policies in policies.items()
        for reward_name, policy in reward_policies.items()
        for agent in create_agents(
            policy,
            load_class_from_module(coleman4hcs.reward, reward_name + "Reward")(),
            algorithm_configs.get(policy_name.lower(), {}).get("window_sizes", []),
        )
    ]


def build_environment(
    build_config: EnvironmentBuildConfig, runtime_metadata: dict[str, str]
) -> tuple[Environment, int]:
    """Create a fresh environment for one execution."""
    agents = build_agents_from_config(
        build_config.algorithm_configs,
        build_config.policy_names,
        build_config.rewards_names,
    )
    scenario = get_scenario_provider(
        build_config.datasets_dir,
        build_config.dataset,
        build_config.sched_time_ratio,
        build_config.use_hcs,
        build_config.use_context,
        build_config.context_config,
        build_config.feature_groups,
    )
    env = Environment(
        agents,
        scenario,
        NAPFDVerdictMetric(),
        results_config=build_config.results_config,
        checkpoint_config=build_config.checkpoint_config,
        telemetry_config=build_config.telemetry_config,
        runtime_metadata=runtime_metadata,
    )
    return env, scenario.max_builds


def exp_run_industrial_dataset_isolated(build_config: EnvironmentBuildConfig, plan: ExecutionPlan) -> None:
    """Execute one run by constructing an isolated Environment inside the worker process."""
    runtime_metadata = {
        "execution_id": plan.execution_id,
        "worker_id": plan.worker_id,
        "parallel_mode": plan.parallel_mode,
    }
    env, _ = build_environment(build_config, runtime_metadata)
    exp_run_industrial_dataset(plan.iteration, plan.trials, env, plan.level, runtime_metadata)


def run_parallel_executions(
    parallel_pool_size: int,
    build_config: EnvironmentBuildConfig,
    execution_plans: list[ExecutionPlan],
) -> None:
    """Run worker executions with responsive Ctrl+C handling.

    Parameters
    ----------
    parallel_pool_size : int
        Number of worker processes.
    build_config : EnvironmentBuildConfig
        Serializable configuration used to build isolated environments.
    execution_plans : list[ExecutionPlan]
        Task parameters for worker execution.
    """
    with get_context("spawn").Pool(parallel_pool_size) as pool:
        async_result = pool.starmap_async(
            exp_run_industrial_dataset_isolated,
            [(build_config, execution_plan) for execution_plan in execution_plans],
        )

        try:
            while True:
                try:
                    async_result.get(timeout=1)
                    break
                except TimeoutError:
                    continue
        except KeyboardInterrupt:
            logging.warning("Interrupt received. Terminating worker pool...")
            pool.terminate()
            pool.join()
            raise SystemExit(130) from None


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


def build_runtime_metadata(dataset: str, sched_time_ratio: float, iteration: int, parallel_mode: str) -> dict[str, str]:
    """Build stable execution metadata for telemetry and persisted results."""
    execution_id = f"{dataset}|tr={sched_time_ratio:.2f}|exp={iteration}|{uuid4().hex[:8]}"
    return {
        "execution_id": execution_id,
        "worker_id": str(iteration),
        "parallel_mode": parallel_mode,
    }


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
    datasets_dir: str,
    dataset: str,
    sched_time_ratio: float,
    use_hcs: bool,
    use_context: bool,
    context_config: dict[str, Any],
    feature_groups: dict[str, Any],
) -> (
    IndustrialDatasetScenarioProvider | IndustrialDatasetHCSScenarioProvider | IndustrialDatasetContextScenarioProvider
):
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
    IndustrialDatasetScenarioProvider or IndustrialDatasetHCSScenarioProvider \
or IndustrialDatasetContextScenarioProvider
        An instance of the scenario provider based on the given configuration.
    """
    base_tcfile = f"{datasets_dir}/{dataset}/features-engineered.csv"

    if use_hcs and not use_context:
        variants_file = f"{datasets_dir}/{dataset}/data-variants.csv"
        return IndustrialDatasetHCSScenarioProvider(base_tcfile, variants_file, sched_time_ratio)

    if use_hcs and use_context:
        raise NotImplementedError

    if use_context:
        contextual_tcfile = f"{datasets_dir}/{dataset}/features-engineered-contextual.csv"
        feature_group_name = str(feature_groups["feature_group_name"])

        feature_group_values_raw = feature_groups["feature_group_values"]
        previous_build_raw = context_config["previous_build"]

        if not isinstance(feature_group_values_raw, list) or not all(
            isinstance(value, str) for value in feature_group_values_raw
        ):
            raise TypeError("feature_group_values must be a list[str]")

        if not isinstance(previous_build_raw, list) or not all(isinstance(value, str) for value in previous_build_raw):
            raise TypeError("previous_build must be a list[str]")

        return IndustrialDatasetContextScenarioProvider(
            contextual_tcfile,
            feature_group_name,
            feature_group_values_raw,
            previous_build_raw,
            sched_time_ratio,
        )

    return IndustrialDatasetScenarioProvider(base_tcfile, sched_time_ratio)


if __name__ == "__main__":
    load_dotenv()
    config = to_builtin_types(get_config())

    # Execution configuration
    (parallel_pool_size, independent_executions, verbose) = map(
        config["execution"].get, ["parallel_pool_size", "independent_executions", "verbose"]
    )

    # Experiment Configuration
    (sched_time_ratio, datasets_dir, datasets, experiment_dir, rewards_names, policy_names) = map(
        config["experiment"].get,
        ["scheduled_time_ratio", "datasets_dir", "datasets", "experiment_dir", "rewards", "policies"],
    )

    # Algorithms Configuration
    algorithm_configs = config["algorithm"]

    # HCS Configuration
    use_hcs = config.get("hcs_configuration", False).get("wts_strategy", False)

    # Contextual Information
    (context_config, feature_groups) = map(config["contextual_information"].get, ["config", "feature_group"])

    # Results / Checkpoint / Telemetry configuration (framework-first defaults)
    results_config = config.get("results", {})
    checkpoint_config = config.get("checkpoint", {})
    telemetry_config = config.get("telemetry", {})

    agents = build_agents_from_config(algorithm_configs, policy_names, rewards_names)

    # Check if there's an agent of type SlidingWindowContextualAgent or ContextualAgent
    has_sliding_window_contextual_agent = any(isinstance(agent, SlidingWindowContextualAgent) for agent in agents)
    has_contextual_agent = any(isinstance(agent, ContextualAgent) for agent in agents)

    use_context = has_contextual_agent or has_sliding_window_contextual_agent

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
            scenario = get_scenario_provider(
                datasets_dir, dataset, tr, use_hcs, use_context, context_config, feature_groups
            )

            # Stop conditional
            trials = scenario.max_builds

            build_config = EnvironmentBuildConfig(
                datasets_dir=datasets_dir,
                dataset=dataset,
                sched_time_ratio=tr,
                use_hcs=use_hcs,
                use_context=use_context,
                context_config=context_config,
                feature_groups=feature_groups,
                results_config=results_config,
                checkpoint_config=checkpoint_config,
                telemetry_config=telemetry_config,
                algorithm_configs=algorithm_configs,
                rewards_names=rewards_names,
                policy_names=policy_names,
            )

            logging.info(
                "Starting dataset=%s time_ratio=%.2f executions=%s agents=%s trials=%s",
                dataset,
                tr,
                independent_executions,
                len(agents),
                trials,
            )

            parallel_mode = "process" if parallel_pool_size > 1 else "sequential"
            execution_plans = [
                ExecutionPlan(
                    iteration=i + 1,
                    trials=trials,
                    level=level,
                    execution_id=build_runtime_metadata(dataset, tr, i + 1, parallel_mode)["execution_id"],
                    worker_id=str(i + 1),
                    parallel_mode=parallel_mode,
                )
                for i in range(independent_executions)
            ]
            # Compute time
            start = time.time()

            if parallel_pool_size > 1:
                run_parallel_executions(parallel_pool_size, build_config, execution_plans)
            else:
                for execution_plan in execution_plans:
                    sequential_plan = ExecutionPlan(
                        iteration=execution_plan.iteration,
                        trials=execution_plan.trials,
                        level=execution_plan.level,
                        execution_id=execution_plan.execution_id,
                        worker_id=execution_plan.worker_id,
                        parallel_mode="sequential",
                    )
                    exp_run_industrial_dataset_isolated(build_config, sequential_plan)

            end = time.time()

            logging.info("Time expend to run the experiments: %s\n\n", end - start)
