# Additional target to perform clang-format/clang-tidy run
# Requires clang-format and clang-tidy

file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
  ${CMAKE_SOURCE_DIR}/base/*.h
  ${CMAKE_SOURCE_DIR}/base/*.cpp
  ${CMAKE_SOURCE_DIR}/compiler/*.h
  ${CMAKE_SOURCE_DIR}/compiler/*.cc
  ${CMAKE_SOURCE_DIR}/graph/*.cc
  ${CMAKE_SOURCE_DIR}/graph/*.h
  ${CMAKE_SOURCE_DIR}/include/*.h
  ${CMAKE_SOURCE_DIR}/topi/*.h
  ${CMAKE_SOURCE_DIR}/topi/*.cc
  ${CMAKE_SOURCE_DIR}/tests/*.cc
  ${CMAKE_SOURCE_DIR}/tests/*.cpp
  )

set(clang-format "clang-format")

# Adding clang-format target if executable is found
find_program(CLANG_FORMAT ${clang-format})
if(CLANG_FORMAT)
  add_custom_target(
    clang-format
    COMMAND ${clang-format}
    -i -style=file
    ${ALL_CXX_SOURCE_FILES}
    )

  add_custom_target(
    clang-format-check
    COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/scripts/clang-format.sh
  )
else()
  message(STATUS "${clang-format} was not found")
endif()

# Adding clang-tidy target if executable is found
find_program(CLANG_TIDY "clang-tidy")
if(CLANG_TIDY)
  add_custom_target(
    clang-tidy
    COMMAND /usr/bin/clang-tidy
    ${ALL_CXX_SOURCE_FILES}
    -config=''
    -checks=*
    --
    -std=c++11
    ${INCLUDE_DIRECTORIES}
    )
endif()

add_custom_target(
   cpplint
   COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/scripts/run_lints.sh
   ${ALL_CXX_SOURCE_FILES}
   )
