# TensorSwift: High-Performance Tensor Arithmetic

A very basic tensor engine done as part of course work in cisc830

## Prerequisite
```
sudo apt-get install python3-dev
```

Check if pybind exists
```
python3 -m pybind11 --includes
```

If not exists
```
pip3 install --no-cache-dir pybind11
```

## Build
To build 

with openmp support (for cpu acceleration)
```
make
```
without openmp support
```
make OMP=1
```

To control number of thread employed for parallelization, do following modification in ts.h
```
#define SYS_PARAM_CPUCOUNT 8
```
change the number according to your system's core count

Output will be an executable file tensorswift.cpython-38-x86_64-linux-gnu.so (depends on python version)



## Example usage
From the same folder where tensorswift.cpython-38-x86_64-linux-gnu.so exists

Check if tensorswift found,
```
python3 -c "import tensorswift"
```

Currently two class exists in the module, 
 1. Data (a simple int wrapping class with basic four operation support)
 2. SwiftTensor, ongoing work, buffer contains garbage, size is in uint64_t and means number of element, arbitrary dimension should work, only floating number hold

Example import
```
a = tensorswift.Data(1)
b = tensorswift.Data(2)

print(a)
print(b)

c = a+b
```

```
# empty tensor
a = tensorswift.SwiftTensor()
# tensor with shape [1,2]
b = tensorswift.SwiftTensor([1,2])
```

example code can be found in example.py