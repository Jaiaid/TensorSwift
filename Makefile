SRC=src
SRC_FILES=${SRC}/*.cpp
PYBIND11_INCLUDE:=$(shell python3 -m pybind11 --includes)
PYTHON_SHARED_LIB_EXTENSION:=$(shell python3-config --extension-suffix)

all: Module

Module: tensorswift

tensorswift: ${SRC_FILES}
	g++ -O3 -Wall -shared -std=c++11 -fPIC ${PYBIND11_INCLUDE} ${SRC_FILES} -o tensorswift${PYTHON_SHARED_LIB_EXTENSION}
