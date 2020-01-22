cmake_minimum_required(VERSION 3.0)
project(pybind_test)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories(../../3rdparty/googletest/googletest)
link_directories(../../3rdparty/googletest/googletest)
link_libraries(gtest)

file(GLOB PYBIND_TEST_SRC
        main.cpp
	test_factory_constructors.cpp
	test_embed.cpp
	test_eval.cpp
	test_pytypes.cpp
	test_modules.cpp
	test_sequences_and_iterators.cpp
	test_builtin_casters.cpp
	test_callbacks.cpp
	test_class.cpp
	test_stl.cpp
	test_kwargs_and_defaults.cpp
	test_local_bindings.cpp
	test_buffers.cpp
	test_numpy_dtypes.cpp
	test_numpy_array.cpp
	test_numpy_vectorize.cpp
	#test_constants_and_functions.cpp
)

pybind11_add_module(pybind11_tests pybind11_tests.cpp)
pybind11_add_module(pybind11_cross_module_tests pybind11_cross_module_tests.cpp)

add_executable(pybind_test ${PYBIND_TEST_SRC} )
target_link_libraries(pybind_test PRIVATE pybind11::embed)
