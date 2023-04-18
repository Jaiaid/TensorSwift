import tensorswift
import time
import torch

import numpy as np
import matplotlib.pyplot as plot
import tensorswift

TEST_SIZE = [100, 1000, 10000, 100000, 1000000]

if __name__=="__main__":
    fig, ax = plot.subplots()
    # test for 1d only
    # with current representation state I think, 1d and 2d will be same
    swift_op_bar = []
    torch_op_bar = []
    torch_cuda_op_bar = []

    for size in TEST_SIZE:
        data1 = list(np.random.randn(size))
        data2 = list(np.random.randn(size))
        diviser = np.random.randint(2, 25)

        swift_times = []
        torch_times = []
        torch_cuda_times = []

        print("Time Report for Size = {0}".format(size))
        
        swifttensor1 = tensorswift.SwiftTensor(data1,[size])
        swifttensor2 = tensorswift.SwiftTensor(data2,[size])
        for i in range(4):
            time1 = time.time()
            if i == 0:
                swifttensor3 = swifttensor1 + swifttensor2
            elif i==1:
                swifttensor3 = swifttensor1 - swifttensor2
            elif i==2:
                swifttensor3 = swifttensor1 * swifttensor2
            elif i==3:
                swifttensor3 = swifttensor1 / diviser

            time2 = time.time()
            swift_times.append(time2-time1)
        # get the mean and append
        swift_op_bar.append(1000*sum(swift_times)/len(swift_times))

        print("--------------------------------------------------------------")
        print("Swifttensor")
        print(swift_times)

        torch_tensor1 = torch.FloatTensor(data1)
        torch_tensor2 = torch.FloatTensor(data2)

        for i in range(4):
            time1 = time.time()
            if i == 0:
                torch_tensor3 = torch_tensor1 + torch_tensor2
            elif i==1:
                torch_tensor3 = torch_tensor1 - torch_tensor2
            elif i==2:
                torch_tensor3 = torch_tensor1 * torch_tensor2
            elif i==3:
                torch_tensor3 = torch_tensor1 / diviser

            time2 = time.time()
            torch_times.append(time2-time1)

        # mean for bar plot
        torch_op_bar.append(1000*sum(torch_times)/len(torch_times))
        print("--------------------------------------------------------------")
        print("Torch")
        print(torch_times)

        torch_tensor1 = torch.Tensor(data1).cuda()
        torch_tensor2 = torch.Tensor(data2).cuda()

        for i in range(4):
            time1 = time.time()
            if i == 0:
                torch_tensor3 = torch_tensor1 + torch_tensor2
            elif i==1:
                torch_tensor3 = torch_tensor1 - torch_tensor2
            elif i==2:
                torch_tensor3 = torch_tensor1 * torch_tensor2
            elif i==3:
                torch_tensor3 = torch_tensor1 / diviser

            time2 = time.time()
            torch_cuda_times.append(time2-time1)

        # for bar plot
        torch_cuda_op_bar.append(1000*sum(torch_cuda_times)/len(torch_cuda_times))
        print("--------------------------------------------------------------")
        print("Torch CUDA")
        print(torch_cuda_times)

        print("\n")

    X = np.arange(len(TEST_SIZE))

    ax.set_ylabel("execution time (ms)")
    ax.set_xlabel("tensor size")
    ax.set_ylim([0,6])
    # to align with zero
    ax.set_xticklabels([0] + TEST_SIZE)
    ax.bar(X-0.2, swift_op_bar, width=0.2, label="SwiftTensor", hatch="//")
    ax.bar(X, torch_op_bar, width=0.2, label="Torch Tensor", hatch="oo")
    ax.bar(X+0.2, torch_cuda_op_bar, width=0.2, label="Torch cuda tensor", hatch="\\")
    ax.legend()

    fig.savefig("exec_time_compare_among_tensors.png", dpi=300)
    