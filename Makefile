SRC=src
SRC_FILES=${SRC}/*.cpp
HEADER_FILES=${SRC}/*.h
OPENMP_FLAGS=
PYBIND11_INCLUDE:=$(shell python3 -m pybind11 --includes)
PYTHON_SHARED_LIB_EXTENSION:=$(shell python3-config --extension-suffix)

ifeq ($(OMP),1)
	OPENMP_FLAGS = -fopenmp -DBUILD_OPENMP
endif

all: Module

Module: tensorswift${PYTHON_SHARED_LIB_EXTENSION}

tensorswift${PYTHON_SHARED_LIB_EXTENSION}: ${SRC_FILES} ${HEADER_FILES}
	g++ -O3 -Wall -shared -std=c++11 ${OPENMP_FLAGS} -fPIC ${PYBIND11_INCLUDE} ${SRC_FILES} -o tensorswift${PYTHON_SHARED_LIB_EXTENSION}

clean:
	rm tensorswift${PYTHON_SHARED_LIB_EXTENSION}