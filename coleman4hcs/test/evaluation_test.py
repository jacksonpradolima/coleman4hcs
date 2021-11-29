import unittest

from coleman4hcs.evaluation import NAPFDMetric, NAPFDVerdictMetric, APFDMetric, APFDcMetric


class RunningEvaluation(unittest.TestCase):
    def testUntreat(self):
        records = [
            {'Name': 8, 'Duration': 0.001, 'NumRan': 1, 'NumErrors': 3, 'Verdict': 1},
            {'Name': 9, 'Duration': 0.497, 'NumRan': 1, 'NumErrors': 1, 'Verdict': 1},
            {'Name': 4, 'Duration': 0.188, 'NumRan': 3, 'NumErrors': 2, 'Verdict': 1},
            {'Name': 6, 'Duration': 0.006, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
            {'Name': 3, 'Duration': 0.006, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
            {'Name': 1, 'Duration': 0.235, 'NumRan': 2, 'NumErrors': 0, 'Verdict': 0},
            {'Name': 2, 'Duration': 5.704, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
            {'Name': 5, 'Duration': 3.172, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
            {'Name': 7, 'Duration': 0.034, 'NumRan': 1, 'NumErrors': 5, 'Verdict': 1}]

        # Tests with same "cost" (duration)
        # records = [
        #     {'Name': 9, 'Duration': 1, 'NumRan': 1, 'NumErrors': 1, 'Verdict': 1},
        #     {'Name': 8, 'Duration': 1, 'NumRan': 1, 'NumErrors': 1, 'Verdict': 1},
        #     {'Name': 4, 'Duration': 1, 'NumRan': 3, 'NumErrors': 0, 'Verdict': 0},
        #     {'Name': 6, 'Duration': 1, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
        #     {'Name': 3, 'Duration': 1, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
        #     {'Name': 1, 'Duration': 1, 'NumRan': 2, 'NumErrors': 0, 'Verdict': 0},
        #     {'Name': 2, 'Duration': 1, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
        #     {'Name': 5, 'Duration': 1, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
        #     {'Name': 7, 'Duration': 1, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0}]

        available_time = sum([item['Duration'] for item in records])

        napfd = NAPFDMetric()
        napfd.update_available_time(available_time * 0.5)  # 50%
        napfd.evaluate(records)
        print("NAPFD:", napfd.fitness, "- APFDc:", napfd.cost)

        napfdV = NAPFDVerdictMetric()
        napfdV.update_available_time(available_time * 0.5)  # 50%
        napfdV.evaluate(records)
        print("NAPFD-Verdict:", napfdV.fitness, "- APFDc:", napfd.cost)

        apfd = APFDMetric()
        apfd.update_available_time(available_time * 0.5)  # 50%
        apfd.evaluate(records)
        print("APFD:", apfd.fitness)

        apfdc = APFDcMetric()
        apfdc.update_available_time(available_time * 0.5)  # 50%
        apfdc.evaluate(records)
        print("APFDc:", apfdc.fitness)


if __name__ == '__main__':
    unittest.main()
