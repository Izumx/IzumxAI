import random
import unittest

from izumxai.distill.tasks import CODE_TASKS, run_tests


class AlgorithmTasks(unittest.TestCase):
    def test_references_pass_and_common_algorithm_mistakes_fail(self):
        wrong = {
            "lower_bound": "def lower_bound(a, x):\n    return len([v for v in a if v <= x])",
            "range_sums": "def range_sums(a, queries):\n    return [sum(a[l:r+1]) for l,r in queries]",
            "longest_unique": "def longest_unique(s):\n    return len(set(s))",
            "min_coins": "def min_coins(coins, amount):\n    return amount // max(coins) if coins else -1",
            "shortest_path": "def shortest_path(n, edges, start, end):\n    return 0 if start == end else 1",
        }
        by_name = {task[0]: task for task in CODE_TASKS}
        self.assertFalse(set(wrong) - by_name.keys(), "algorithm families missing from task bank")
        for name, buggy_source in wrong.items():
            with self.subTest(name=name):
                _, reference, _, cases_fn, _ = by_name[name]
                for seed in (7, 42):
                    cases = cases_fn(random.Random(seed))
                    self.assertTrue(run_tests(reference, name, cases))
                    self.assertFalse(run_tests(buggy_source, name, cases))


if __name__ == "__main__":
    unittest.main()
