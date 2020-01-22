#include "gtest/gtest.h"
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/packed_func_ext.h>
#include <tvm/schedule_pass.h>

template <typename T>
class TD;

using namespace tvm;

TEST(LANG_PASS_IR, test_ir_transform)
{
}
/*
def test_ir_transform():
    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            x = tvm.call_extern("int32", "TestA", i * 3 + j * 1)
            ib.emit(tvm.call_extern("int32", "TestB", x))
            ib.emit(tvm.call_extern("int32", "TestC", x))
    body = ib.get()

    def preorder(op):
        if op.name == "TestC":
            return tvm.const(0, "int32")
        return None

    def postorder(op):
        assert isinstance(op, tvm.expr.Call)
        if op.name == "TestA":
            return tvm.call_extern("int32", "TestB", op.args[0] + 1)
        return op
    body = tvm.ir_pass.IRTransform(body, preorder, postorder, ["Call"])
    stmt_list = tvm.make.stmt_list(body.body.body)
    assert stmt_list[0].value.args[0].name == "TestB"
    assert stmt_list[1].value.value == 0

*/
