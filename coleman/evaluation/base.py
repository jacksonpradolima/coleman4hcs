"""Base evaluation metric class."""


class EvaluationMetric:
    """Base class for evaluation metrics.

    Attributes
    ----------
    available_time : float
        The time available for test execution.
    scheduled_testcases : list
        Test cases that were scheduled for execution.
    unscheduled_testcases : list
        Test cases that were not scheduled.
    detection_ranks : list
        Ranks at which failures were detected.
    detection_ranks_time : list
        Durations of failure-detecting test cases.
    detection_ranks_failures : list
        Failure counts at each detection rank.
    ttf : int
        Time to Fail (rank value).
    ttf_duration : float
        Time spent until the first test case fail.
    fitness : float
        APFD or NAPFD value.
    cost : float
        APFDc value.
    detected_failures : int
        Number of detected failures.
    undetected_failures : int
        Number of undetected failures.
    recall : float
        Recall metric value.
    avg_precision : float
        Average precision metric value.
    """

    def __init__(self):
        """Initialize the EvaluationMetric."""
        self.available_time = 0
        self.reset()

    def update_available_time(self, available_time: float):
        """Update the available time for the metric.

        Parameters
        ----------
        available_time : float
            Time available for the metric.
        """
        self.available_time = available_time

    def reset(self):
        """Reset all attributes to their default values."""
        self.scheduled_testcases = []
        self.unscheduled_testcases = []
        self.detection_ranks = []
        self.detection_ranks_time = []
        self.detection_ranks_failures = []
        self.ttf = self.ttf_duration = 0  # Time to Fail (rank value)
        self.fitness = 0  # APFD or NAPFD value
        self.cost = 0  # APFDc (to compute at same time, for instance, with NAPFD)
        self.detected_failures = self.undetected_failures = 0
        self.recall = self.avg_precision = 0

    def process_test_suite(self, test_suite, error_key):
        """Process the test suite and return the costs and total failure count.

        Parameters
        ----------
        test_suite : list of dict
            Test suite to process.
        error_key : str
            Key to determine the error in the test suite.

        Returns
        -------
        costs : list
            List of durations for each test case.
        total_failure_count : int
            Total number of failures detected.
        total_failed_tests : int
            Total number of test cases that failed.
        """
        rank_counter = 1
        total_failure_count = total_failed_tests = scheduled_time = 0
        costs = []

        for test_case in test_suite:
            failure_count = test_case[error_key]
            total_failure_count += failure_count
            total_failed_tests += test_case["Verdict"]
            costs.append(test_case["Duration"])

            if not self.detection_ranks_time:
                self.ttf_duration += test_case["Duration"]

            if scheduled_time + test_case["Duration"] <= self.available_time:
                if failure_count:
                    self.detected_failures += failure_count * rank_counter
                    self.detection_ranks.append(rank_counter)

                    self.detection_ranks_failures.append(failure_count)
                    self.detection_ranks_time.append(test_case["Duration"])

                scheduled_time += test_case["Duration"]
                self.scheduled_testcases.append(test_case["Name"])
                rank_counter += 1
            else:
                self.unscheduled_testcases.append(test_case["Name"])
                self.undetected_failures += failure_count

        self.detected_failures = len(self.detection_ranks) if error_key == "Verdict" else self.detected_failures

        return costs, total_failure_count, total_failed_tests

    def evaluate(self, test_suite):
        """Evaluate the test suite.

        This is an abstract method and must be implemented in child classes.

        Parameters
        ----------
        test_suite : list of dict
            Test suite to evaluate.

        Raises
        ------
        NotImplementedError
            If not implemented in a child class.
        """
        raise NotImplementedError("This method must be overridden in child classes")

    def set_default_metrics(self):
        """Set the default values for NAPFD and APFDc metrics.

        This method is called when there are no detected failures in the test
        suite, ensuring that the metric attributes are appropriately
        initialized.

        Notes
        -----
        This method updates the instance's attributes directly and does not
        return any value.
        """
        self.ttf = -1
        self.recall = self.avg_precision = 1
        self.fitness = self.cost = 1
