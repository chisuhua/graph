#include "gtest/gtest.h"
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>

template <typename T>
class TD;

using namespace tvm;
TEST(LANG_PASS_BASIC, simplify)
{
    Var x("x");
    Expr e1 = ir::Simplify(x + 2 + 1);
    CHECK(ir::Equal(e1, x + 3));

    Expr e2 = ir::Simplify(x * 3 + 5 * x);
    CHECK(ir::Equal(e2, x * 8));

    Expr e3 = ir::Simplify(x - x / 3 * 3);
    CHECK(ir::Equal(e3, ir::Mod::make(x, 3)));

    Expr let = ir::Let::make(x, 1, x + 3);
    Expr e4 = ir::Simplify(let);
    CHECK(ir::Equal(e4, 4));
}

TEST(LANG_PASS_BASIC, verify_ssa)
{
    Var x("x");
    Var y;
    Stmt z = ir::Evaluate::make(x + y);
    CHECK(ir::VerifySSA(z));
}

TEST(LANG_PASS_BASIC, convert_ssa)
{
    Var x("x");
    Var y;
    Expr let1 = ir::Let::make(x, 1, x + 1);
    Expr let2 = ir::Let::make(x, 1, x + y);
    Stmt z = ir::Evaluate::make(let1 + let2);
    CHECK(!ir::VerifySSA(z));
    Stmt z_ssa = ir::ConvertSSA(z);
    CHECK(ir::VerifySSA(z_ssa));
}

TEST(LANG_PASS_BASIC, expr_use_var)
{
    Var x("x");
    CHECK(ir::ExprUseVar(x + 1, x));
    CHECK(!ir::ExprUseVar(1 + 10, x));
}
