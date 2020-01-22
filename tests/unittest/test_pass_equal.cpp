#include "gtest/gtest.h"
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>

using namespace tvm;

extern int buffer_read;
extern int buffer_write;
extern int buffer_rw;

TEST(LANG_PASS_BASIC, equal_expr)
{
    Var x("x");
    Var y("y");

    auto func1 = [&]() { return x + y + 1; };
    auto func2 = [&]() { return exp(x + y + 1) * y / 4; };

    CHECK(ir::Equal(func1(), func1()));
    CHECK(ir::Equal(func2(), func2()));
    CHECK(!ir::Equal(func2(), func1()));
}

TEST(LANG_PASS_BASIC, equal_compute)
{
    Var x("x");
    Var y("y");
    auto n = 128u;
    Tensor A = placeholder({ n, n }, "A");
    Tensor B = placeholder({ n, n }, "B");

    Var ii("ii");
    Var jj("jj");
    IterVar k = reduce_axis({ 0, n }, "k");
    // FIXME
    auto func1 = [&]() {
        return sum(A[ii][k] * B[jj][k], { k });
    };
    func1();

    //    Buffer Ab = tvm::decl_buffer({n}, "A");
    // Var n("n");

    //    // Refer to compiler/pass/inject_double_buffer loop_seq
    //    // FIXME
    //    Var i("i");
    //    Var j("j");
    //    auto func2 = [&]() {
    //        // auto Aptr = Ab.access_ptr(buffer_rw);
    //        Expr value = ir::Load::make(Ab->dtype, Ab->data, i , const_true());
    //        Stmt stmt = ir::Store::make(Ab->data, value, i, const_true());
    //        Stmt loop_stmt = ir::For::make(i, 0, n, ir::ForType::Serial, ir::DeviceAPI::Host, stmt);
    //        return loop_stmt;
    //
    //    };
    //
    //    CHECK(ir::Equal(func1(), func1()));
    //    CHECK(ir::Equal(func2(), func2()));
    //    CHECK(!ir::Equal(func2(), ir::Evaluate::make(func1())));
}

/*

def test_equal_compute():
    x = tvm.var('x')
    y = tvm.var('y')
    n = 128
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((n, n), name='B')
    ii = tvm.var('i')
    jj = tvm.var('j')

    def func1():
        k = tvm.reduce_axis((0, n), name='k')
        return tvm.sum(A[ii, k] * B[jj, k], axis=k)

    Ab = tvm.decl_buffer((n,), name='A')
    n = tvm.var("n")
    def func2():
        ib = tvm.ir_builder.create()
        A = ib.buffer_ptr(Ab)
        with ib.for_range(0, n, name="i") as i:
            A[i] = A[i] + 1
            with ib.for_range(0, 10, name="j") as j:
                A[j] = A[j] + 2
        return ib.get()

    assert tvm.ir_pass.Equal(func1(), func1())
    assert tvm.ir_pass.Equal(func2(), func2())
*/
