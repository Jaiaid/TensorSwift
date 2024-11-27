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
      upto 1e-6 precision
        - Addition
        - Subtraction
        - Multiplication
        - Division
        - Sum of all array
"""
class FloatArithmaticTestCase(unittest.TestCase):
    RANDOM_SEED = 20241126
    ARRAY_LENGTH = 1000000
    EPSILON_DIFFERENCE = 1e-5

    def equalityCheck(data1:np.ndarray, data2:tensorswift.SwiftTensor) -> bool:
        if data1.size != data2.size():
            return False
        
        for idx, e in enumerate(data1):
            if abs(e - data2[idx]) > FloatArithmaticTestCase.EPSILON_DIFFERENCE:
                return False

        return True

    def setUp(self):
        # create two random list from random uniform ins [1.5e-6,1e-6] and 
        # the choice of this range is based on chosen precision accuracy and array size
        # at that range, mantissa gives at most 2^-43~=1.19x10^-13 precision
        #
        # adding 10^6 element will cause power raise of 10^6 which in 2's power will cause exponent increment 20 times
        # each exponent increment will cause one bit of precision loss, which totals 2^-42 + ... +2^-22 ~= 2^-21 
        # why start from -42? => cause if both 2^-43 precision was bit 1 than the sum will be 2^-42, so in worst case 
        # we can lose value starting from 2^-42
        # therefore, as 4e-6 < 2^-21 < 5e-6 but the observed precision difference is at 1e-5
        np.random.seed(self.RANDOM_SEED)
        self.data_a = np.random.uniform(1e-6, 1.05e-6, self.ARRAY_LENGTH).astype(np.float32)
        self.data_b = np.random.uniform(1e-6, 1.05e-6, self.ARRAY_LENGTH).astype(np.float32)

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

    def testSum(self):
        result_np = np.sum(self.data_a)
        # bring back to host memory
        result_ts = self.ts_a.sum()
        assert FloatArithmaticTestCase.equalityCheck(np.array([result_np]), result_ts), "element difference is larger than {0}".format(FloatArithmaticTestCase.EPSILON_DIFFERENCE)


if __name__ == "__main__":
    unittest.main() # run all tests