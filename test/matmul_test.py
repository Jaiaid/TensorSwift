"""
This test file is supposed to run from parent directory (root of git repo)

"""

import unittest
import numpy as np # to verify results correctness

import tensorswift

"""
    Test of matrix multiplication operation using tensorswift class

    - Define two numpy arrays, data_a and data_b using uniform data from [0,1]
    - Define two swifttensor from data_a and data_b
    - Perform matrix multiplication and test elementwise accuracy at 1e-5 precision
    - The performed unit tests are
        - 1d matrix multiplication
        - 2d matrix multiplication
        - Value error for same dimension but different size multiplication
        - Value error for different dimension multiplication
"""
class MatmulTestCase(unittest.TestCase):
    RANDOM_SEED = 20241126
    ARRAY_LENGTH = 100
    ARRAY_LENGTH_SQRT = 10
    EPSILON_DIFFERENCE = 5e-5

    def equalityCheck(data1:np.ndarray, data2:tensorswift.SwiftTensor) -> bool:
        if data1.size != data2.size():
            return False

        for idx, e in enumerate(data1):
            if abs(e - data2[idx]) > MatmulTestCase.EPSILON_DIFFERENCE:
                print(idx, e, data2[idx])
                return False

        return True

    def setUp(self):
        # create two random list from random uniform ins [0,1]
        np.random.seed(self.RANDOM_SEED)
        self.data_a = np.random.uniform(0, 1, self.ARRAY_LENGTH).astype(np.float32)
        self.data_b = np.random.uniform(0, 1, self.ARRAY_LENGTH).astype(np.float32)

        self.ts_a = tensorswift.SwiftTensor(list(self.data_a), [self.ARRAY_LENGTH])
        self.ts_b = tensorswift.SwiftTensor(list(self.data_b), [self.ARRAY_LENGTH])

    def test1D(self):
        result_np = np.matmul(self.data_a, self.data_b.T)
        result_ts = self.ts_a.matmul(self.ts_b.T)

        assert MatmulTestCase.equalityCheck(np.array([result_np], dtype=np.float32), result_ts)

    def test2D(self):
        result_np = np.matmul(
            self.data_a.reshape(self.ARRAY_LENGTH_SQRT, self.ARRAY_LENGTH_SQRT),
            self.data_b.reshape(self.ARRAY_LENGTH_SQRT, self.ARRAY_LENGTH_SQRT)
        )
        result_ts = self.ts_a.view([self.ARRAY_LENGTH_SQRT, self.ARRAY_LENGTH_SQRT]).matmul(
            self.ts_b.view([self.ARRAY_LENGTH_SQRT, self.ARRAY_LENGTH_SQRT])
        )

        assert MatmulTestCase.equalityCheck(
            result_np.reshape(self.ARRAY_LENGTH),
            result_ts.view([self.ARRAY_LENGTH])
        )

    def testInvalidOperationDiffSizeSameDim(self):
        with self.assertRaises(ValueError):
            diff_size = tensorswift.SwiftTensor(
                list(self.data_a[:self.ARRAY_LENGTH - 1]),
                [self.ARRAY_LENGTH-1]
            )
            # following line will raise value error
            self.ts_a.matmul(diff_size)

    def testInvalidOperationDiffDimension(self):
        with self.assertRaises(ValueError):
            changed_view = self.ts_a.view([self.ARRAY_LENGTH_SQRT, self.ARRAY_LENGTH_SQRT])
            # following line will raise value error
            self.ts_a.matmul(changed_view)


if __name__ == "__main__":
    unittest.main() # run all tests