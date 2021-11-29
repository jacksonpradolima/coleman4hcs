import argparse
import os
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

from coleman4hcs import scenarios
from coleman4hcs.agent import RewardAgent, RewardSlidingWindowAgent
from coleman4hcs.environment import Environment
from coleman4hcs.evaluation import NAPFDMetric, NAPFDVerdictMetric
from coleman4hcs.policy import EpsilonGreedyPolicy
from coleman4hcs.policy import FRRMABPolicy, GreedyPolicy, RandomPolicy, UCBPolicy
from coleman4hcs.reward import RNFailReward, TimeRankReward

warnings.filterwarnings("ignore")

ITERATIONS = 30
PARALLEL_POOL_SIZE = 15
DEFAULT_EXPERIMENT_DIR = 'results/experiments/'
EXPERIMENT_DIR = DEFAULT_EXPERIMENT_DIR
DEFAULT_SCALING_FACTOR_FRR = [0.3]
DEFAULT_SCALING_FACTOR_UCB = [0.5, 0.3]
DEFAULT_EPSILON = [0.5, 0.3]
DEFAULT_WINDOW_SIZES = [100]
DEFAULT_SCHED_TIME_RATIO = [0.1, 0.5, 0.8]


def run_experiments_with_threads(repo_path, dataset, policies, window_sizes=DEFAULT_WINDOW_SIZES,
                                 sched_time_ratio=0.5,
                                 reward_functions=[RNFailReward(), TimeRankReward()],
                                 evaluation_metric=NAPFDMetric(),
                                 considers_variants=False):
    # Define my agents
    agents = []
    for policy in policies:
        for rew_fun in reward_functions:
            if type(policy) == FRRMABPolicy:
                for w in window_sizes:
                    agents.append(RewardSlidingWindowAgent(policy, rew_fun, w))
            elif type(policy) == UCBPolicy:
                # Based on tunning settings
                if (type(rew_fun) == RRankReward or type(rew_fun) == TimeRankReward) and policy.c == 0.5:
                    agents.append(RewardAgent(policy, rew_fun))
                elif type(rew_fun) == RNFailReward and policy.c == 0.3:
                    agents.append(RewardAgent(policy, rew_fun))
            elif type(policy) == EpsilonGreedyPolicy:
                # Based on tunning settings
                if (type(rew_fun) == RRankReward or type(rew_fun) == TimeRankReward) and policy.epsilon == 0.5:
                    agents.append(RewardAgent(policy, rew_fun))
                elif type(rew_fun) == RNFailReward and policy.epsilon == 0.3:
                    agents.append(RewardAgent(policy, rew_fun))
            else:
                agents.append(RewardAgent(policy, rew_fun))

    # Get scenario
    if considers_variants:
        scenario = scenarios.IndustrialDatasetHCSScenarioProvider(
            f"{repo_path}/{dataset}/features-engineered.csv",
            f"{repo_path}/{dataset}/data-variants.csv",
            sched_time_ratio)
    else:
        scenario = scenarios.IndustrialDatasetScenarioProvider(
            f"{repo_path}/{dataset}/features-engineered.csv",
            sched_time_ratio)

    # Stop conditional
    trials = scenario.max_builds

    # Prepare the experiment
    env = Environment(agents, scenario, evaluation_metric)

    parameters = [(i + 1, trials, env) for i in range(ITERATIONS)]

    # create a file with a unique header for the scenario (workaround)
    env.create_file(f"{EXPERIMENT_DIR}{str(env.scenario_provider)}.csv")

    # Compute time
    start = time.time()

    with Pool(PARALLEL_POOL_SIZE) as p:
        p.starmap(exp_run_industrial_dataset, parameters)

    end = time.time()

    print(f"Time expend to run the experiments: {end - start}")


def exp_run_industrial_dataset(iteration, trials, env: Environment):
    env.run_single(iteration, trials, print_log=True)
    env.store_experiment(f"{EXPERIMENT_DIR}{str(env.scenario_provider)}.csv")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Main')

    ap.add_argument('--project_dir', required=True)
    ap.add_argument('--considers_variants', default=False,
                    type=lambda x: (str(x).lower() == 'true'))

    ap.add_argument('--parallel_pool_size', type=int,
                    default=PARALLEL_POOL_SIZE)

    ap.add_argument('--datasets', nargs='+', default=[], required=True,
                    help='Datasets to analyse. Ex: \'deeplearning4j@deeplearning4j\'')

    ap.add_argument('-r', '--rewards', nargs='+', default=['RNFailReward', 'TimeRankReward'],
                    help='Reward Functions available: RNFailReward and TimeRankReward')

    ap.add_argument('-p', '--policies', nargs='+', default=['Random', 'Greedy', 'EpsilonGreedy', 'UCB', 'FRR'],
                    help='Policies available: Random, Greedy, EpsilonGreedy, UCB, FRR')

    ap.add_argument('--scaling_factor_frr', type=int,
                    nargs='+', default=DEFAULT_SCALING_FACTOR_FRR)

    ap.add_argument('--scaling_factor_ucb', type=int,
                    nargs='+', default=DEFAULT_SCALING_FACTOR_UCB)

    ap.add_argument('--epsilon', type=int, nargs='+', default=DEFAULT_EPSILON)

    ap.add_argument('--window_sizes', type=int, nargs='+',
                    default=DEFAULT_WINDOW_SIZES)

    ap.add_argument('--sched_time_ratio', nargs='+',
                    default=DEFAULT_SCHED_TIME_RATIO, help='Schedule Time Ratio')

    ap.add_argument('-o', '--output_dir',
                    default=DEFAULT_EXPERIMENT_DIR,
                    const=DEFAULT_EXPERIMENT_DIR,
                    nargs='?')

    args = ap.parse_args()

    PARALLEL_POOL_SIZE = args.parallel_pool_size
    considers_variants = args.considers_variants

    reward_functions = []
    for reward in args.rewards:
        if reward == 'RNFailReward':
            reward_functions.append(RNFailReward())
        elif reward == 'TimeRankReward':
            reward_functions.append(TimeRankReward())

    policies = []
    for pol in args.policies:
        if pol == "Random":
            policies.append(RandomPolicy())
        elif pol == "Greedy":
            policies.append(GreedyPolicy())
        elif pol == "EpsilonGreedy":
            policies.extend([EpsilonGreedyPolicy(float(epsilon))
                             for epsilon in args.scaling_factor_ucb])
        elif pol == "UCB":
            policies.extend([UCBPolicy(float(scaling))
                             for scaling in args.scaling_factor_ucb])
        elif pol == "FRR":
            policies.extend([FRRMABPolicy(float(scaling))
                             for scaling in args.scaling_factor_frr])
        else:
            print(f"Policies '{pol}' not found!")

    metric = NAPFDVerdictMetric()

    for tr in args.sched_time_ratio:
        EXPERIMENT_DIR = os.path.join(args.output_dir, f"time_ratio_{int(tr * 100)}/")

        Path(EXPERIMENT_DIR).mkdir(parents=True, exist_ok=True)

        for dataset in args.datasets:
            run_experiments_with_threads(args.project_dir, dataset, policies,
                                         window_sizes=args.window_sizes,
                                         reward_functions=reward_functions,
                                         sched_time_ratio=tr,
                                         evaluation_metric=metric,
                                         considers_variants=considers_variants)
