import tensorswift

nn = tensorswift.CGraph()
# layer = tensorswift.Linear(10,10, True)
layer = tensorswift.CGOp()

nn.add(layer)

nn.compute(tensorswift.SwiftTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
print(nn.output().parameter)