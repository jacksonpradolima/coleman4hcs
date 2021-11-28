class EvaluationMetric(object):
    """
    Evaluation Metric
    """

    def __init__(self):
        self.available_time = 0
        self.reset()

    def update_available_time(self, available_time):
        self.available_time = available_time

    def reset(self):
        self.scheduled_testcases = []
        self.unscheduled_testcases = []
        self.detection_ranks = []
        self.detection_ranks_time = []
        self.detection_ranks_failures = []
        # Time to Fail (rank value)
        self.ttf = 0
        self.ttf_duration = 0
        # APFD or NAPFD value
        self.fitness = 0
        self.detected_failures = 0
        self.undetected_failures = 0
        self.recall = 0
        self.avg_precision = 0
        # APFDc (to compute at same time, for instance, with NAPFD)
        self.cost = 0

    def evaluate(self, test_suite):
        raise NotImplementedError('This method must be override')


class NAPFDMetric(EvaluationMetric):
    """
    Normalized Average Percentage of Faults Detected (NAPFD) Metric based
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'NAPFD'

    def evaluate(self, test_suite):
        super().reset()

        rank_counter = 1
        total_failure_count = 0
        scheduled_time = 0
        costs = []
        failures = 0
        self.detected_failures = 0

        # We consider the faults are different, that is, a fault is only revelead by only a test case
        # Build prefix sum of durations to find cut off point
        for row in test_suite:
            total_failure_count += row['NumErrors']
            failures += row['Verdict']
            costs.append(row['Duration'])

            # Time spent to fail
            if len(self.detection_ranks_time) == 0:
                self.ttf_duration += row['Duration']

            if scheduled_time + row['Duration'] <= self.available_time:
                # If the Verdict is "Failed"
                if row['NumErrors'] > 0:
                    self.detected_failures += row['NumErrors'] * rank_counter
                    self.detection_ranks.append(rank_counter)

                    # Individual information
                    self.detection_ranks_failures.append(row['NumErrors'])
                    self.detection_ranks_time.append(row['Duration'])

                scheduled_time += row['Duration']
                self.scheduled_testcases.append(row['Name'])
                rank_counter += 1
            else:
                self.unscheduled_testcases.append(row['Name'])
                self.undetected_failures += row['NumErrors']

        if total_failure_count > 0:
            # Time to Fail (rank value)
            self.ttf = self.detection_ranks[0] if self.detection_ranks else 0
            self.recall = sum(self.detection_ranks_failures) / total_failure_count
            self.avg_precision = 123

            p = self.recall if self.undetected_failures > 0 else 1
            no_testcases = len(test_suite)

            # NAPFD
            self.fitness = p - self.detected_failures / (total_failure_count * no_testcases) + p / (2 * no_testcases)

            # APFDc
            self.cost = sum([sum(costs[i - 1:]) - 0.5 * costs[i - 1] for i in self.detection_ranks]) / (
                    sum(costs) * failures)

        else:
            # Time to Fail (rank value)
            self.ttf = -1
            self.recall = 1
            self.avg_precision = 1

            # NAPFD
            self.fitness = 1
            # APFDc
            self.cost = 1


class NAPFDVerdictMetric(EvaluationMetric):
    """
    Normalized Average Percentage of Faults Detected (NAPFD) Metric based on Verdict
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'NAPFDVerdict'

    def evaluate(self, test_suite):
        super().reset()

        rank_counter = 1
        total_failure_count = 0
        scheduled_time = 0
        costs = []

        # Build prefix sum of durations to find cut off point
        for row in test_suite:
            total_failure_count += row['Verdict']
            costs.append(row['Duration'])

            # Time spent to fail
            if len(self.detection_ranks_time) == 0:
                self.ttf_duration += row['Duration']

            if scheduled_time + row['Duration'] <= self.available_time:
                # If the Verdict is "Failed"
                if row['Verdict']:
                    self.detection_ranks.append(rank_counter)
                    # Individual information
                    self.detection_ranks_failures.append(row['Verdict'])
                    self.detection_ranks_time.append(row['Duration'])

                scheduled_time += row['Duration']
                self.scheduled_testcases.append(row['Name'])
                rank_counter += 1
            else:
                self.unscheduled_testcases.append(row['Name'])
                self.undetected_failures += row['Verdict']

        self.detected_failures = len(self.detection_ranks)

        assert self.undetected_failures + self.detected_failures == total_failure_count

        if total_failure_count > 0:
            try:
                # Time to Fail (rank value)
                self.ttf = self.detection_ranks[0] if self.detection_ranks else 0
                self.recall = self.detected_failures / total_failure_count
                self.avg_precision = 123

                p = self.recall if self.undetected_failures > 0 else 1
                no_testcases = len(test_suite)

                # NAPFD
                self.fitness = p - sum(self.detection_ranks) / (total_failure_count * no_testcases) + p / (
                        2 * no_testcases)

                # APFDc
                self.cost = sum([sum(costs[i - 1:]) - 0.5 * costs[i - 1] for i in self.detection_ranks]) / (
                        sum(costs) * total_failure_count)
            except:
                pass
        else:
            # Time to Fail (rank value)
            self.ttf = -1
            self.recall = 1
            self.avg_precision = 1

            # NAPFD
            self.fitness = 1
            # APFDc
            self.cost = 1


class APFDMetric(EvaluationMetric):
    """
    Average Percentage of Faults Detected (APFD) Metric based on Verdict
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Average Percentage of Faults Detected (APFD)'

    def evaluate(self, test_suite):
        super().reset()

        rank_counter = 1
        scheduled_time = 0
        total_failure_count = 0
        scheduled_time = 0

        # Build prefix sum of durations to find cut off point
        for row in test_suite:
            total_failure_count += row['Verdict']
            if scheduled_time + row['Duration'] <= self.available_time:
                # If the Verdict is "Failed"
                if row['Verdict']:
                    self.detection_ranks.append(rank_counter)

                scheduled_time += row['Duration']
                self.scheduled_testcases.append(row['Name'])
                rank_counter += 1
            else:
                self.undetected_failures += row['Verdict']

        self.detected_failures = len(self.detection_ranks)

        assert self.undetected_failures + self.detected_failures == total_failure_count

        if total_failure_count > 0:
            # Time to Fail (rank value)
            self.ttf = self.detection_ranks[0] if self.detection_ranks else 0
            no_testcases = len(test_suite)

            # APFD
            self.fitness = 1 - sum(self.detection_ranks) / (no_testcases * total_failure_count) + 1 / (2 * no_testcases)
            self.recall = self.detected_failures / total_failure_count
            self.avg_precision = 123
        else:
            # Time to Fail (rank value)
            self.ttf = 0
            # APFD
            self.fitness = 1
            self.recall = 1
            self.avg_precision = 1


class APFDcMetric(EvaluationMetric):
    """
    Average Percentage of Faults Detected with cost (APFDc)
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Average Percentage of Faults Detected with cost (APFDc)'

    def evaluate(self, test_suite):
        super().reset()

        # The severity for each fault is the same for all

        rank_counter = 1
        total_failure_count = 0
        costs = []
        scheduled_time = 0

        # Build prefix sum of durations to find cut off point
        for row in test_suite:
            total_failure_count += row['Verdict']
            costs.append(row['Duration'])

            # Time spent to fail
            if len(self.detection_ranks_time) == 0:
                self.ttf_duration += row['Duration']

            if scheduled_time + row['Duration'] <= self.available_time:
                # If the Verdict is "Failed"
                if row['Verdict']:
                    self.detection_ranks.append(rank_counter)
                    # Individual information
                    self.detection_ranks_failures.append(row['Verdict'])
                    self.detection_ranks_time.append(row['Duration'])

                scheduled_time += row['Duration']
                self.scheduled_testcases.append(row['Name'])
                rank_counter += 1
            else:
                self.unscheduled_testcases.append(row['Name'])
                self.undetected_failures += row['Verdict']

        self.detected_failures = len(self.detection_ranks)

        assert self.undetected_failures + self.detected_failures == total_failure_count

        if total_failure_count > 0:
            # Time to Fail (rank value)
            self.ttf = self.detection_ranks[0] if self.detection_ranks else 0

            detection_cost = 0
            for i in self.detection_ranks:
                detection_cost += sum(costs[i - 1:]) - 0.5 * costs[i - 1]

            no_testcases = len(test_suite)
            # APFDc
            self.fitness = (detection_cost) / (sum(costs) * total_failure_count)
            self.recall = self.detected_failures / total_failure_count
            self.avg_precision = 123
        else:
            # Time to Fail (rank value)
            self.ttf = -1
            # APFDc
            self.fitness = 1
            self.recall = 1
            self.avg_precision = 1
