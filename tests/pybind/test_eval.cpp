/*
    tests/test_eval.cpp -- Usage of eval() and eval_file()

    Copyright (c) 2016 Klemens D. Morgenstern

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "gtest/gtest.h"
#include "dmlc/logging.h"
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include "pybind11_tests.h"
namespace py = pybind11;

TEST(test_eval, test_eval)
{
    py::scoped_interpreter guard{};

    auto m = py::module::import("test_eval");
    CHECK_EQ(m.attr("test_eval_statements")().cast<bool>(), true);

    CHECK_EQ(m.attr("test_eval")().cast<bool>(), true);

    CHECK_EQ(m.attr("test_eval_single_statement")().cast<bool>(), true);

    auto sys_path = py::module::import("os").attr("path");
    auto filename = sys_path.attr("join")(sys_path.attr("dirname")(__FILE__), "test_eval_call.py");
    py::print(filename);
    CHECK_EQ(m.attr("test_eval_file")(filename).cast<bool>(), true);

    CHECK_EQ(m.attr("test_eval_failure")().cast<bool>(), true);

    CHECK_EQ(m.attr("test_eval_file_failure")().cast<bool>(), true);
}



PYBIND11_EMBEDDED_MODULE(test_eval, m) {
    // test_evals

    auto global = py::dict(py::module::import("__main__").attr("__dict__"));

    m.def("test_eval_statements", [global]() {
        auto local = py::dict();
        local["call_test"] = py::cpp_function([&]() -> int {
            return 42;
        });

        // Regular string literal
        py::exec(
            "message = 'Hello World!'\n"
            "x = call_test()",
            global, local
        );

        // Multi-line raw string literal
        py::exec(R"(
            if x == 42:
                print(message)
            else:
                raise RuntimeError
            )", global, local
        );
        auto x = local["x"].cast<int>();

        return x == 42;
    });

    m.def("test_eval", [global]() {
        auto local = py::dict();
        local["x"] = py::int_(42);
        auto x = py::eval("x", global, local);
        return x.cast<int>() == 42;
    });

    m.def("test_eval_single_statement", []() {
        auto local = py::dict();
        local["call_test"] = py::cpp_function([&]() -> int {
            return 42;
        });

        auto result = py::eval<py::eval_single_statement>("x = call_test()", py::dict(), local);
        auto x = local["x"].cast<int>();
        return result.is_none() && x == 42;
    });

    m.def("test_eval_file", [global](py::str filename) {
        auto local = py::dict();
        local["y"] = py::int_(43);

        int val_out;
        local["call_test2"] = py::cpp_function([&](int value) { val_out = value; });

        auto result = py::eval_file(filename, global, local);
        return val_out == 43 && result.is_none();
    });

    m.def("test_eval_failure", []() {
        try {
            py::eval("nonsense code ...");
        } catch (py::error_already_set &) {
            return true;
        }
        return false;
    });

    m.def("test_eval_file_failure", []() {
        try {
            py::eval_file("non-existing file");
        } catch (std::exception &) {
            return true;
        }
        return false;
    });
}
