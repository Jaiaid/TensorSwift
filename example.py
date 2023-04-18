import tensorswift

# constructor
# a tensor of (2,2) shape with uninitialized entries
a=tensorswift.SwiftTensor([2,2])
# a tensor of (2,2) shape with initialized entries [[1,4][8,0]]
b=tensorswift.SwiftTensor([1,4,8,0],[2,2])
# empty contstructor, creates a tensor with empty buffer
c=tensorswift.SwiftTensor()

# item can be assigned at particular index
a[1,1]=4
# create different view
# view different but d and a will have same storage
d = a.view([4,1])

# to view shape
print(a.shape)
# to view size
print(a.size())

# operator overloading is done for operations
# return tensor with new storage
print(a+b)
print(a-b)
print(a*b)
print(a/b)

# return a new tensor with appropriate shape
print(a.matmul(b))
# sum of all element of a
print(b.sum())
