cmake_minimum_required(VERSION 3.0)
project(unittest)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories(../../3rdparty/googletest/googletest)
link_directories(../../3rdparty/googletest/googletest)
link_libraries(gtest)

set(TEST_EXECS "")

file(GLOB TEST_SRCS
        base_equality_test.cc
        base_simplify_test.cc
        base_interval_test.cc
        expr_test.cc
        ir*.cc
        attrs_test.cc
        tensor_test.cc
        simple_passes_test.cc
        packed_func_test.cc
        relay_pass_type_infer_test.cc
)

foreach(__srcpath ${TEST_SRCS})
    get_filename_component(__srcname ${__srcpath} NAME)
    string(REPLACE ".cc" "" __execname ${__srcname})
    add_executable(${__execname} ${__srcpath})
    list(APPEND TEST_EXECS ${__execname})
    target_link_libraries(${__execname}
      ixgraph_relay
      ixgraph_runtime
      ixgraph_compiler
      ${GTEST_LIB} pthread dl)
    target_include_directories(${__execname}
      PUBLIC "include")
    #set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_ALL 1)
    #set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
endforeach()
#add_custom_target(cpptest DEPENDS ${TEST_EXECS})



