cmake_minimum_required(VERSION 3.0)
project(unittest)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})


include_directories(../../3rdparty/googletest/googletest)
link_directories(../../3rdparty/googletest/googletest)
link_libraries(gtest)

file(GLOB SRC
        main.cpp
	test_lang_basic.cpp
	# TODO test_lang_buffer.cpp
	# TODO test_lang_constructor.cpp
	test_lang_operator.cpp
	test_lang_container.cpp
	test_lang_reflection.cpp
	test_lang_tensor.cpp
	test_lang_data_layout.cpp
	test_lang_group.cpp
	test_lang_schedule.cpp
	test_arith_simplify.cpp
	test_arith_rewrite_simplify.cpp
	test_arith_const_int_bound.cpp
	test_arith_detect_clip_bound.cpp
	test_arith_detect_linear_equation.cpp
	test_arith_domain_touched.cpp
	test_arith_intset.cpp
	test_arith_modular_set.cpp
	test_pass_attrs_hash_equal.cpp
	test_pass_basic.cpp
	#test_pass_bound_checkers.cpp
	test_pass_decorate_device_scope.cpp
	#test_pass_equal.cpp
	test_pass_inline.cpp
	#test_pass_inject_copy_intrin.cpp
	#test_pass_inject_double_buffer.cpp
	#test_pass_inject_vthread.cpp
	# TODO test_pass_loop_partition.cpp
	# TODO test_pass_lower_warp_memory.cpp
	test_pass_makeapi.cpp
	# TODO test_pass_remove_no_op.cpp
	# TODO test_pass_rewrite_unsafe_select.cpp
	# TODO test_pass_simplify.cpp
	# TODO test_pass_split_pipeline.cpp
	# TODO test_pass_storage_flatten.cpp
	test_pass_storage_rewrite.cpp
	test_pass_storage_sync.cpp
	#test_codegen_arm.cpp
	test_codegen_x86.cpp
	test_codegen_bool.cpp
	test_runtime_ndarray.cpp
)

add_executable(unittest ${SRC})
target_link_libraries(unittest PRIVATE pybind11::embed)
target_link_libraries(unittest PUBLIC ixgraph_compiler)

#	unittest.cpp
#	test_pybind.cpp)

#add_library(unittest MODULE ${SRC})
#target_link_libraries(unittest PRIVATE pybind11::module)
#set_target_properties(unittest PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
