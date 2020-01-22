#include "gtest/gtest.h"
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>

template <typename T>
class TD;

using namespace tvm;

TEST(LANG_PASS_OPT, test_rewrite_select)
{
    auto nop = ir::Evaluate::make(1);
    Var A("A");
    Var i("i");
    Stmt a = ir::Allocate::make(A, Float(32), { 0, 100 }, const_true(1), nop);
    Expr y = ir::Select::make(i > 1, ir::Load::make(Float(32), A, i - 1, const_true(1)), 1.0);

    // Stmt yy = ir::RewriteUnsafeSelect(ir::Evaluate::make(y));
    // LOG(INFO) << "\n" << yy->type_key() << "\n" << yy;
    //
    /*
def test_rewrite_Select():
    ib = tvm.ir_builder.create()
    A = ib.allocate("float32", 100, name="A", scope="global")
    i = tvm.var("i")
    y = tvm.expr.Select(i > 1, A[i-1], 1.0)
    yy = tvm.ir_pass.RewriteUnsafeSelect(tvm.make.Evaluate(y)).value

    z = tvm.expr.Select(
        tvm.expr.Select(i > 1, A[i-1], 1.0) > 0.0, A[i], 0.1)
    zz = tvm.ir_pass.RewriteUnsafeSelect(tvm.make.Evaluate(z)).value

    a = tvm.expr.Select(i>10, y, z)
    aa = tvm.ir_pass.RewriteUnsafeSelect(tvm.make.Evaluate(a)).value
    assert yy.name == "tvm_if_then_else"
    assert zz.name == "tvm_if_then_else"
    assert isinstance(aa, tvm.expr.Select)
*/
}
