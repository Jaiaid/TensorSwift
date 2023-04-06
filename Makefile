SRC=src
SRC_FILES=${SRC}/*.cpp
HEADER_FILES=${SRC}/*.h
PYBIND11_INCLUDE:=$(shell python3 -m pybind11 --includes)
PYTHON_SHARED_LIB_EXTENSION:=$(shell python3-config --extension-suffix)

all: Module

Module: tensorswift${PYTHON_SHARED_LIB_EXTENSION}

tensorswift${PYTHON_SHARED_LIB_EXTENSION}: ${SRC_FILES} ${HEADER_FILES}
	g++ -O3 -Wall -shared -std=c++11 -fPIC ${PYBIND11_INCLUDE} ${SRC_FILES} -o tensorswift${PYTHON_SHARED_LIB_EXTENSION}
