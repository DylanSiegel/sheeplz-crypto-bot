# File: tests/test_utils.py

import unittest
import logging
from utils.utils import setup_logging, get_logger
import os

class TestUtils(unittest.TestCase):

    def test_setup_logging_console(self):
        setup_logging(log_level="DEBUG")
        logger = get_logger()
        self.assertEqual(logger.level, logging.DEBUG)

    def test_setup_logging_file(self):
        log_file = 'logs/test.log'
        if os.path.exists(log_file):
            os.remove(log_file)
        setup_logging(log_level="INFO", log_file=log_file)
        logger = get_logger()
        logger.info("Test log message.")
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, 'r') as f:
            content = f.read()
            self.assertIn("Test log message.", content)
        os.remove(log_file)

if __name__ == '__main__':
    unittest.main()
