import logging
import os
import random
import time
import unittest
import warnings
from pathlib import Path

import numpy as np

from coleman4hcs import scenarios
from coleman4hcs.agent import RewardSlidingWindowAgent, RewardAgent
from coleman4hcs.environment import Environment
from coleman4hcs.evaluation import NAPFDVerdictMetric
from coleman4hcs.policy import FRRMABPolicy, UCBPolicy
from coleman4hcs.reward import RNFailReward
from coleman4hcs.reward import TimeRankReward

warnings.filterwarnings("ignore")
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Keep the same result
random.seed(10)

if not os.path.exists("../../logs"):
    os.makedirs("../../logs")

logging.basicConfig(
    filename="../../logs/data_filtering.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
)


class RunningPolicy(unittest.TestCase):
    def setUp(self):
        # Shared settings
        self.reward_functions = [RNFailReward(), TimeRankReward()]

        # HCS
        self.output_hcs_dir = '../../results/experiments_hcs_test/'
        self.dataset_hcs_dir = "../../example/libssh@libssh-mirror"
        self.dataset_hcs = 'libssh@total'

        Path(self.output_hcs_dir).mkdir(parents=True, exist_ok=True)

        self.scenario_hcs = scenarios.IndustrialDatasetHCSScenarioProvider(
            f"{self.dataset_hcs_dir}/{self.dataset_hcs}/features-engineered.csv",
            f"{self.dataset_hcs_dir}/{self.dataset_hcs}/data-variants.csv",
        )

        # NON HCS
        self.output_dir = '../../results/experiments_test/'
        self.dataset_dir = "../../example"
        self.dataset = 'fakedata'

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.scenario = scenarios.IndustrialDatasetScenarioProvider(
            f"{self.dataset_dir}/{self.dataset}/features-engineered.csv",
        )

    @unittest.skip
    def test_FRRMAB_HCS(self):
        # Stop conditional
        trials = self.scenario_hcs.max_builds
        iteration = 1

        # Prepare the agents with same policy (untreat by default)
        agents = [RewardSlidingWindowAgent(FRRMABPolicy(0.3), rew_fun, 50) for rew_fun in self.reward_functions]

        # Prepare the experiment
        env = Environment(agents, self.scenario_hcs, NAPFDVerdictMetric())

        # create a file with a unique header for the scenario (workaround)
        env.create_file(f"{self.output_dir_hcs}{str(env.scenario_provider)}.csv")

        # Compute time
        start = time.time()

        env.run_single(iteration, trials, print_log=True)
        env.store_experiment(f"{self.output_dir_hcs}{str(env.scenario_provider)}.csv")

        end = time.time()

        print(f"Time expend to run the experiments: {end - start}")

    def test_UCB(self):
        # Stop conditional
        trials = self.scenario.max_builds
        iteration = 1

        # Prepare the agents with same policy (untreat by default)
        agents = [RewardAgent(UCBPolicy(0.3), rew_fun) for rew_fun in self.reward_functions]

        # Prepare the experiment
        env = Environment(agents, self.scenario, NAPFDVerdictMetric())

        # create a file with a unique header for the scenario (workaround)
        env.create_file(f"{self.output_dir}{str(env.scenario_provider)}.csv")

        # Compute time
        start = time.time()

        env.run_single(iteration, trials, print_log=True)
        env.store_experiment(f"{self.output_dir}{str(env.scenario_provider)}.csv")

        end = time.time()

        print(f"Time expend to run the experiments: {end - start}")


if __name__ == '__main__':
    unittest.main()
