import unittest
from izumxai.distill.prepare_smel_data import verify_reference


class VerifiedReferences(unittest.TestCase):
    def test_wrong_reference_cannot_enter_training(self):
        self.assertTrue(verify_reference('def add_one(x):\n    return x+1', '', ['assert add_one(3)==4']))
        self.assertFalse(verify_reference('def add_one(x):\n    return x-1', '', ['assert add_one(3)==4']))

    def test_out_of_scope_io_code_is_filtered(self):
        self.assertFalse(verify_reference('import os\ndef f():\n    return os.getcwd()', '', []))
        self.assertFalse(verify_reference('def f():\n    return open("private.txt").read()', '', []))


if __name__=='__main__':
    unittest.main()
