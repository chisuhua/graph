#include "gtest/gtest.h"
#include <tvm/build_module.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/packed_func_ext.h>
#include <tvm/schedule_pass.h>
#include "pybind11_tests.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

template <typename T>
class TD;

using namespace tvm;

TEST(LANG_RUNTIME, test_dltensor_compatible)
{
    py::module np = py::module::import("numpy");
    auto dtype = py::dtype("int64");

    Var n("n");
    Buffer Ab = tvm::decl_buffer({n}, dtype);

    Var i("i");

}

/*
@tvm.register_extension
class MyTensorView(object):
    _tvm_tcode = tvm.TypeCode.ARRAY_HANDLE
    def __init__(self, arr):
        self.arr = arr

    @property
    def _tvm_handle(self):
        return self.arr._tvm_handle

def test_dltensor_compatible():
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n,), dtype)
    i = tvm.var('i')
    ib = tvm.ir_builder.create()
    A = ib.buffer_ptr(Ab)
    with ib.for_range(0, n - 1, "i") as i:
        A[i + 1] = A[i] + 1
    stmt = ib.get()
    fapi = tvm.ir_pass.MakeAPI(stmt, "arange", [Ab], 0, True)
    fapi = tvm.ir_pass.LowerTVMBuiltin(fapi)
    f = tvm.codegen.build_module(fapi, "stackvm")
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    aview = MyTensorView(a)
    f(aview)
    np.testing.assert_equal(a.asnumpy(), np.arange(a.shape[0]))

*/
