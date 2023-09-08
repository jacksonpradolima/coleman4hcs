class EvaluationMetric:
    """
    Evaluation Metric
    """

    def __init__(self):
        """
        Initializes the EvaluationMetric class.
        """
        self.available_time = 0
        self.reset()

    def update_available_time(self, available_time: float):
        """
        Updates the available time for the metric.

        :param available_time: Time available for the metric.
        """
        self.available_time = available_time

    def reset(self):
        """
        Resets all the attributes to their default values.
        """
        self.scheduled_testcases = []
        self.unscheduled_testcases = []
        self.detection_ranks = []
        self.detection_ranks_time = []
        self.detection_ranks_failures = []
        # Time to Fail (rank value)
        self.ttf = self.ttf_duration = 0
        # APFD or NAPFD value
        self.fitness = 0
        self.detected_failures = 0
        self.undetected_failures = 0
        self.recall = 0
        self.avg_precision = 0
        # APFDc (to compute at same time, for instance, with NAPFD)
        self.cost = 0

    def process_test_suite(self, test_suite, error_key):
        """
        Process the test suite and returns the costs and total failure count.

        :param test_suite: Test suite to process.
        :param error_key: Key to determine the error in the test suite.
        :return: Tuple containing costs and total failure count.
        """
        rank_counter = 1
        total_failure_count = 0
        total_failed_tests = 0
        scheduled_time = 0
        costs = []

        # We consider the faults are different, that is, a fault is only revelead by only a test case
        # Build prefix sum of durations to find cut off point
        for row in test_suite:
            total_failure_count += row[error_key]
            total_failed_tests += row['Verdict']
            costs.append(row['Duration'])

            # Time spent to fail
            if not self.detection_ranks_time:
                self.ttf_duration += row['Duration']

            if scheduled_time + row['Duration'] <= self.available_time:
                # If the Verdict is "Failed"
                if row[error_key]:
                    self.detected_failures += row[error_key] * rank_counter
                    self.detection_ranks.append(rank_counter)

                    # Individual information
                    self.detection_ranks_failures.append(row[error_key])
                    self.detection_ranks_time.append(row['Duration'])

                scheduled_time += row['Duration']
                self.scheduled_testcases.append(row['Name'])
                rank_counter += 1
            else:
                self.unscheduled_testcases.append(row['Name'])
                self.undetected_failures += row[error_key]

        # Update detected failures if verdict-based
        self.detected_failures = len(self.detection_ranks) if error_key == "Verdict" else self.detected_failures

        return costs, total_failure_count, total_failed_tests

    def evaluate(self, test_suite):
        """
        Evaluates the test suite. This is an abstract method and must be implemented in child classes.

        :param test_suite: Test suite to evaluate.
        :raises NotImplementedError: If not implemented in a child class.
        """
        raise NotImplementedError("This method must be overridden in child classes")


class NAPFDMetric(EvaluationMetric):
    """
    Normalized Average Percentage of Faults Detected (NAPFD) Metric based
    """

    def __str__(self):
        return 'NAPFD'

    def evaluate(self, test_suite):
        super().reset()
        costs, total_failure_count, total_failed_tests = self.process_test_suite(test_suite, 'NumErrors')

        if total_failure_count > 0:
            self.compute_metrics(costs, total_failure_count, total_failed_tests, len(test_suite))
        else:
            self.set_default_metrics()

    def compute_metrics(self, costs, total_failure_count, total_failed_tests, no_testcases):
        """
        Computes the metrics based on the provided parameters.

        :param costs: List of costs for each test case.
        :param total_failure_count: Total number of failures.
        :param total_failed_tests: Total number of tests failed.
        :param no_testcases: Total number of test cases.
        """
        # Time to Fail (rank value)
        self.ttf = self.detection_ranks[0] if self.detection_ranks else 0
        self.recall = sum(self.detection_ranks_failures) / total_failure_count
        self.avg_precision = 123

        p = self.recall if self.undetected_failures > 0 else 1

        # NAPFD
        self.fitness = p - self.detected_failures / (total_failure_count * no_testcases) + p / (2 * no_testcases)

        # APFDc
        self.cost = sum([sum(costs[i - 1:]) - 0.5 * costs[i - 1] for i in self.detection_ranks]) / (
            sum(costs) * total_failed_tests)

    def set_default_metrics(self):
        """
        Sets default values for the metrics.
        """
        # Time to Fail (rank value)
        self.ttf = -1
        self.recall = self.avg_precision = 1

        # NAPFD and APFDc
        self.fitness = self.cost = 1


class NAPFDVerdictMetric(EvaluationMetric):
    """
    Normalized Average Percentage of Faults Detected (NAPFD) Metric based on Verdict
    """

    def __str__(self):
        return 'NAPFDVerdict'

    def evaluate(self, test_suite):
        super().reset()
        costs, total_failure_count, _ = self.process_test_suite(test_suite, 'Verdict')

        assert self.undetected_failures + self.detected_failures == total_failure_count

        if total_failure_count > 0:
            self.compute_metrics(costs, total_failure_count, len(test_suite))
        else:
            self.set_default_metrics()

    def compute_metrics(self, costs, total_failure_count, no_testcases):
        # Time to Fail (rank value)
        self.ttf = self.detection_ranks[0] if self.detection_ranks else 0
        self.recall = self.detected_failures / total_failure_count
        self.avg_precision = 123  # Placeholder value, can be replaced with real calculation

        p = self.recall if self.undetected_failures > 0 else 1

        # NAPFD
        self.fitness = p - sum(self.detection_ranks) / (total_failure_count * no_testcases) + p / (
            2 * no_testcases)

        # APFDc
        self.cost = sum([sum(costs[i - 1:]) - 0.5 * costs[i - 1] for i in self.detection_ranks]) / (
            sum(costs) * total_failure_count)

    def set_default_metrics(self):
        # Time to Fail (rank value)
        self.ttf = -1
        self.recall = self.avg_precision = 1

        # NAPFD and APFDc
        self.fitness = self.cost = 1
