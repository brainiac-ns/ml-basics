import unittest
from src.utils import foo


class TestFoo(unittest.TestCase):
    def test_foo(self):
        val = foo()

        self.assertEqual(val, True)
