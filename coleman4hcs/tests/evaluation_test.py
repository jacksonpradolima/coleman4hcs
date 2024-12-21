import unittest
from coleman4hcs.evaluation import NAPFDMetric, NAPFDVerdictMetric, EvaluationMetric

# Constants for error messages
NAPFD_FITNESS_NON_NEGATIVE = "NAPFD fitness should be non-negative."
NAPFD_FITNESS_NOT_EXCEED_ONE = "NAPFD fitness should not exceed 1."
NAPFD_COST_NON_NEGATIVE = "NAPFD cost should be non-negative."

class RunningEvaluationTests(unittest.TestCase):
    def setUp(self):
        """
        Set up reusable test data and metric instances for the tests.
        """
        self.records = [
            {'Name': 8, 'Duration': 0.001, 'NumRan': 1, 'NumErrors': 3, 'Verdict': 1},
            {'Name': 9, 'Duration': 0.497, 'NumRan': 1, 'NumErrors': 1, 'Verdict': 1},
            {'Name': 4, 'Duration': 0.188, 'NumRan': 3, 'NumErrors': 2, 'Verdict': 1},
            {'Name': 6, 'Duration': 0.006, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
            {'Name': 3, 'Duration': 0.006, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
            {'Name': 1, 'Duration': 0.235, 'NumRan': 2, 'NumErrors': 0, 'Verdict': 0},
            {'Name': 2, 'Duration': 5.704, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
            {'Name': 5, 'Duration': 3.172, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
            {'Name': 7, 'Duration': 0.034, 'NumRan': 1, 'NumErrors': 5, 'Verdict': 1}
        ]

        self.available_time = sum([item['Duration'] for item in self.records])

    def test_evaluation_metric_not_implemented(self):
        """
        Test that the abstract EvaluationMetric class raises NotImplementedError.
        """
        metric = EvaluationMetric()
        with self.assertRaises(NotImplementedError):
            metric.evaluate(self.records)

    def test_evaluation_metric_str(self):
        """
        Test that the __str__ method is implemented for derived metric classes.
        """
        napfd = NAPFDMetric()
        napfd_v = NAPFDVerdictMetric()

        self.assertEqual(str(napfd), "NAPFD", "NAPFDMetric __str__ method failed.")
        self.assertEqual(str(napfd_v), "NAPFDVerdict", "NAPFDVerdictMetric __str__ method failed.")

    def test_napfd_metric(self):
        """
        Test NAPFDMetric with standard records and 50% available time.
        """
        napfd = NAPFDMetric()
        napfd.update_available_time(self.available_time * 0.5)  # 50%
        napfd.evaluate(self.records)

        # Assertions for fitness and cost
        self.assertGreaterEqual(napfd.fitness, 0, NAPFD_FITNESS_NON_NEGATIVE)
        self.assertLessEqual(napfd.fitness, 1, NAPFD_FITNESS_NOT_EXCEED_ONE)
        self.assertGreaterEqual(napfd.cost, 0, NAPFD_COST_NON_NEGATIVE)

    def test_napfd_verdict_metric(self):
        """
        Test NAPFDVerdictMetric with standard records and 50% available time.
        """
        napfd_v = NAPFDVerdictMetric()
        napfd_v.update_available_time(self.available_time * 0.5)  # 50%
        napfd_v.evaluate(self.records)

        # Assertions for fitness and cost
        self.assertGreaterEqual(napfd_v.fitness, 0, NAPFD_FITNESS_NON_NEGATIVE)
        self.assertLessEqual(napfd_v.fitness, 1, NAPFD_FITNESS_NOT_EXCEED_ONE)
        self.assertGreaterEqual(napfd_v.cost, 0, NAPFD_COST_NON_NEGATIVE)

    def test_empty_records(self):
        """
        Test metrics with empty records to ensure proper handling.
        """
        napfd = NAPFDMetric()
        napfd.update_available_time(self.available_time * 0.5)
        napfd.evaluate([])  # Empty records

        # Assertions for fitness and cost
        self.assertEqual(napfd.fitness, 1, "NAPFD fitness should be 1 for empty records.")
        self.assertEqual(napfd.cost, 1, "NAPFD cost should be 1 for empty records.")

    def test_napfd_verdict_metric_no_failures(self):
        """
        Test NAPFDVerdictMetric with no failures to ensure default metrics are set.
        """
        no_failure_records = [
            {'Name': i, 'Duration': 1, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0} for i in range(1, 10)
        ]

        napfd_v = NAPFDVerdictMetric()
        napfd_v.update_available_time(self.available_time * 0.5)
        napfd_v.evaluate(no_failure_records)

        # Assertions for default metrics
        self.assertEqual(napfd_v.fitness, 1, "NAPFD-V fitness should be 1 when no failures are present.")
        self.assertEqual(napfd_v.cost, 1, "NAPFD-V cost should be 1 when no failures are present.")

    def test_identical_durations(self):
        """
        Test metrics with records having identical durations.
        """
        identical_records = [
            {'Name': i, 'Duration': 1, 'NumRan': 1, 'NumErrors': i % 2, 'Verdict': i % 2} for i in range(1, 10)
        ]

        napfd = NAPFDMetric()
        napfd.update_available_time(self.available_time * 0.5)
        napfd.evaluate(identical_records)

        # Assertions for fitness and cost
        self.assertGreaterEqual(napfd.fitness, 0, NAPFD_FITNESS_NON_NEGATIVE)
        self.assertLessEqual(napfd.fitness, 1, NAPFD_FITNESS_NOT_EXCEED_ONE)
        self.assertGreaterEqual(napfd.cost, 0, NAPFD_COST_NON_NEGATIVE)

    def test_identical_cost_and_results(self):
        """
        Test metrics where all records have the same cost and results.
        """
        identical_cost_records = [
            {'Name': i, 'Duration': 1, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 1} for i in range(1, 10)
        ]

        napfd = NAPFDMetric()
        napfd.update_available_time(self.available_time * 0.5)
        napfd.evaluate(identical_cost_records)

        # Assertions for fitness and cost
        self.assertGreaterEqual(napfd.fitness, 0, NAPFD_FITNESS_NON_NEGATIVE)
        self.assertLessEqual(napfd.fitness, 1, NAPFD_FITNESS_NOT_EXCEED_ONE)
        self.assertGreaterEqual(napfd.cost, 0, NAPFD_COST_NON_NEGATIVE)

if __name__ == '__main__':
    unittest.main()
