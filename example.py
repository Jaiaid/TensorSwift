import tensorswift

a=tensorswift.SwiftTensor([2,2])
a[1,1]=4
print(a)
# view different but c and a will have same storage
c = a.view([4,1])
c[0]=10
# a will also change
print(a)