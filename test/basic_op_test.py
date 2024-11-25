"""
This test file is supposed to run from parent directory (root of git repo)

"""

import unittest
import numpy as np # to verify results correctness

import tensorswift

"""
    Test of Basic Arithmatic operation using tensorswift class

    - Define two numpy arrays, data_a and data_b using uniform data from [0,1]
    - Define two swifttensor from data_a and data_b
    - Perform following operations and test for accuracy by comparing equality to numpy array results
      upto 1e-11 precision
        - Addition
        - Subtraction
        - Multiplication
        - Division 
"""
class FloatArithmaticTestCase(unittest.TestCase):
    RANDOM_SEED = 20241126
    ARRAY_LENGTH = 100
    EPSILON_DIFFERENCE = 5e-12

    def equalityCheck(data1:np.ndarray, data2:tensorswift.SwiftTensor) -> bool:
        if data1.size != data2.size():
            return False
        
        for idx, e in enumerate(data1):
            if abs(e - data2[idx]) > FloatArithmaticTestCase.EPSILON_DIFFERENCE:
                return False

        return True

    def setUp(self):
        # create two random list from random uniform ins [0,1]
        np.random.seed(self.RANDOM_SEED)
        self.data_a = np.random.uniform(0, 1, self.ARRAY_LENGTH).astype(np.float32)
        self.data_b = np.random.uniform(0, 1, self.ARRAY_LENGTH).astype(np.float32)

        self.ts_a = tensorswift.SwiftTensor(list(self.data_a), [self.ARRAY_LENGTH])
        self.ts_b = tensorswift.SwiftTensor(list(self.data_b), [self.ARRAY_LENGTH])

    def testAddition(self):
        result_np = self.data_a + self.data_b
        result_ts = self.ts_a + self.ts_b
        assert FloatArithmaticTestCase.equalityCheck(result_np, result_ts)

    def testSubtraction(self):
        result_np = self.data_a - self.data_b
        result_ts = self.ts_a - self.ts_b
        assert FloatArithmaticTestCase.equalityCheck(result_np, result_ts)

    def testMultiplication(self):
        result_np = self.data_a * self.data_b
        result_ts = self.ts_a * self.ts_b
        assert FloatArithmaticTestCase.equalityCheck(result_np, result_ts)

    def testDivision(self):
        result_np = self.data_a / self.data_b
        result_ts = self.ts_a / self.ts_b
        assert FloatArithmaticTestCase.equalityCheck(result_np, result_ts)


if __name__ == "__main__":
    unittest.main() # run all tests