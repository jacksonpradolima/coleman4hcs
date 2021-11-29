import os
import time
import unittest
from pathlib import Path

from coleman4hcs import scenarios
from coleman4hcs.agent import ContextualAgent
from coleman4hcs.agent import RewardSlidingWindowAgent
from coleman4hcs.environment import Environment
from coleman4hcs.evaluation import NAPFDVerdictMetric
from coleman4hcs.policy import FRRMABPolicy, LinUCBPolicy
from coleman4hcs.reward import RNFailReward
from coleman4hcs.reward import RRankReward
from coleman4hcs.reward import TimeRankReward


class RunningPolicy(unittest.TestCase):
    def setUp(self):
        self.output_dir = '../../results/experiments_test/'
        self.dataset_dir = "../../data/libssh@libssh-mirror"
        self.dataset = 'libssh@total'
        self.reward_functions = [RRankReward(), RNFailReward(), TimeRankReward()]
        self.reward_functions = [RNFailReward(), TimeRankReward()]

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Get scenario
        self.scenario = scenarios.IndustrialDatasetHCSScenarioProvider(
            f"{self.dataset_dir}/{self.dataset}/features-engineered.csv",
            f"{self.dataset_dir}/{self.dataset}/data-variants.csv",
        )

    def testFRRMAB(self):
        # Stop conditional
        trials = self.scenario.max_builds
        iteration = 1

        # Prepare the agents with same policy (untreat by default)
        agents = [RewardSlidingWindowAgent(FRRMABPolicy(0.3), rew_fun, 50) for rew_fun in self.reward_functions]

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
