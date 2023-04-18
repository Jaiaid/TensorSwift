import tensorswift as ts
import numpy as np
import sys

def main():

    npfile = sys.argv[1]
    tsfile = sys.argv[2]
    op = int(sys.argv[3])
    row = int(sys.argv[4])
    col = int(sys.argv[5])

    data1 = np.random.randint(row, size=(row, col))
    data2 = np.random.randint(row, size=(row, col))
    diviser = row

    np_tensor1 = np.array(data1)
    np_tensor2 = np.array(data2)

    tsdata1 = [element for sublist in data1.tolist() for element in sublist]
    tsdata2 = [element for sublist in data2.tolist() for element in sublist]

    swifttensor1 = ts.SwiftTensor(tsdata1,[row,col])
    swifttensor2 = ts.SwiftTensor(tsdata2,[row, col])

    if op == 1:   #add
        np_result = np_tensor1 + np_tensor2
        ts_result = swifttensor1 + swifttensor2
    elif op == 2: # minus
        np_result = np_tensor1 - np_tensor2
        ts_result = swifttensor1 - swifttensor2
    elif op == 3: # times
        np_result = np_tensor1 * np_tensor2
        ts_result = swifttensor1 * swifttensor2
    elif op == 4: # division
        np_result = np_tensor1 / diviser
        ts_result = swifttensor1 / diviser
    elif op == 5: # matmul
        np_result = np.matmul(np_tensor1, np_tensor2.T)
        ts_result = swifttensor1.matmul(swifttensor2.T)
    elif op == 6: # multiply
        np_result = np.multiply(np_tensor1, np_tensor2)
        ts_result = swifttensor1.multiply(swifttensor2)
    elif op == 7: # dot
        np_result = np.dot(np_tensor1, np_tensor2.T)
        ts_result = swifttensor1.dot(swifttensor2.T)
    elif op == 8: # transpose
        np_result = np_tensor1.T
        ts_result = swifttensor1.T

    with open(npfile, 'w') as npf:
        for i in range(np_result.shape[0]):
            for j in range(np_result.shape[1]):
                npf.write(f"{np_result[i][j]:.3f}\n")

    with open(tsfile, 'w') as tsf:
        for i in range(ts_result.shape[0]*ts_result.shape[1]):
            tsf.write(f"{ts_result[i]:.3f}\n")

if __name__=="__main__":
    main()