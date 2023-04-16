import tensorswift
import time
import torch

import numpy as np
import matplotlib.pyplot as plot
import tensorswift

TEST_SIZE = [100, 1000, 10000, 100000, 1000000]

if __name__=="__main__":
    # test for 1d only
    # with current representation state I think, 1d and 2d will be same
    for size in TEST_SIZE:
        data1 = list(np.random.randn(size))
        data2 = list(np.random.randn(size))

        torch_tensor1 = torch.FloatTensor(data1)
        torch_tensor2 = torch.FloatTensor(data2)

        swifttensor1 = tensorswift.SwiftTensor(data1,[size])
        swifttensor2 = tensorswift.SwiftTensor(data2,[size])

        swift_times = []
        torch_times = []

        print("Time Report for Size = {0}".format(size))
        for i in range(4):
            time1 = time.time()
            if i == 0:
                swifttensor3 = swifttensor1 + swifttensor2
            elif i==1:
                swifttensor3 = swifttensor1 - swifttensor2
            # elif i==2:
            #     swifttensor3 = swifttensor1.dot(swifttensor2)
            # elif i==3:
            #     swifttensor3 = swifttensor1 / swifttensor2

            time2 = time.time()
            swift_times.append(time2-time1)

        print("--------------------------------------------------------------")
        print("Swifttensor")
        print(swift_times)

        for i in range(4):
            time1 = time.time()
            if i == 0:
                torch_tensor3 = torch_tensor1 + torch_tensor2
            elif i==1:
                torch_tensor3 = torch_tensor1 - torch_tensor2
            # elif i==2:
            #     torch_tensor3 = torch_tensor1 * torch_tensor2
            # elif i==3:
            #     torch_tensor3 = torch_tensor1 / torch_tensor2

            time2 = time.time()
            torch_times.append(time2-time1)
        print("--------------------------------------------------------------")
        print("Torch")
        print(torch_times)

        print("\n")