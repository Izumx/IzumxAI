import unittest

from izumxai.eval import solve

PROMPT = '''def add(a, b):
    """Adds two numbers.
    >>> add(1, 2)
    3
    add(2, 2) == 4
    """
'''


class Solver(unittest.TestCase):
    def test_examples_come_only_from_statement(self):
        self.assertEqual(solve.public_examples(PROMPT, "add"), [("add(1, 2)", "3"), ("add(2, 2)", "4")])

    def test_checker_reports_wrong_and_accepts_right_code(self):
        examples = solve.public_examples(PROMPT, "add")
        self.assertEqual(solve.check_examples("", "def add(a, b):\n    return a + b", examples), (0, ""))
        fails, message = solve.check_examples("", "def add(a, b):\n    return a - b", examples)
        self.assertEqual(fails, 2)
        self.assertIn("expected 3", message)

    def test_checker_survives_infinite_loop(self):
        fails, message = solve.check_examples("", "def add(a, b):\n    while True: pass", [("add(1, 2)", "3")])
        self.assertEqual(fails, 1)
        self.assertIn("timed out", message)


if __name__ == "__main__":
    unittest.main()
