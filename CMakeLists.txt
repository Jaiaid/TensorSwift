cmake_minimum_required(VERSION 3.15)
project(tensorswift VERSION 0.1 LANGUAGES CXX CUDA)

include(CTest)

# set CPP required and default standard
set(CMAKE_CXX_STANDARD_REQUIRED 14)
set(CMAKE_CXX_STANDARD 14)

# define the build target
add_library(${CMAKE_PROJECT_NAME} SHARED)

# set compilation flag
target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC -O3)

# add source headers for compilation
# PUBLIC inclusion
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC include)

# find available python3 version, shared library suffix will be done accordingly
# this is required to progress with build
find_package(Python3 3.8 EXACT COMPONENTS Interpreter Development REQUIRED)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".${Python3_SOABI}.so")
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
# include Python3 associated headers for compilation and libraries to link 
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${PYTHON3_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Python3::Python)


# find pybind11 for python interface creation
# this is required to progress with build
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)


# find OpenMP for multithreaded calculation
# not required for build
find_package(OpenMP)
if (OPENMP_FOUND)
	message(STATUS "module OpenMP found, multithreaded calculation will be built.")
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BUILD_OPENMP)
	target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
else()
	message(WARNING "module OpenMP not found, multithreaded calculation will be unavailable.")
endif()


# find CUDA for GPU accleration code compilation
find_package(CUDA)
if (CUDA_FOUND)
	message(STATUS "module CUDA found, GPU acceleration will be built if available.")
	# compilation flag to enable conditional compilation of cuda oprations
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BUILD_CUDA)
	# Set CUDA properties for the target 
	set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES 
		CUDA_SEPARABLE_COMPILATION ON 
		# CUDA_ARCHITECTURES "80" # Specify target architectures
	)

	target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})

	set(CUDA_SOURCES
		src/device/cuda/cuda_device.cu
		src/device/cuda/ops.cu)

	# compile the cuda files
	target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${CUDA_SOURCES})
else()
	message(WARNING "module CUDA not found, GPU acceleration will be unavailable.")
endif()

# set source files, add new file path if new source files added
set(CORE_SOURCES
	src/core/ts.cpp
	src/core/storage.cpp)

set(DEVICESPECIFIC_SOURCES
	src/device/device_builder.cpp
	src/device/cpu/cpu_device.cpp)

set(CGRAPH_SOURCES 
	src/cgraph/cgraph.cpp
	src/cgraph/cg_op.cpp
	src/cgraph/cg_node.cpp)

set(PYINTERFACE_SOURCES 
	src/python/pymodule_interface.cpp
	src/core/python/core_interface.cpp
	src/cgraph/python/cgraph_interface.cpp)


# compile the source files
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${CORE_SOURCES})
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${DEVICESPECIFIC_SOURCES})
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${CGRAPH_SOURCES})
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${PYINTERFACE_SOURCES})


# test if the build is done
add_test(NAME build_check1 COMMAND ${Python3_EXECUTABLE} -c "import ${CMAKE_PROJECT_NAME}")
