#include "gtest/gtest.h"
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
// #include <arithmetic/Simplify.h>

template <typename T>
class TD;
using namespace tvm;
using namespace tvm::ir;

TEST(LANG_ARITH, test_dectect_linear_equation_basic)
{
    Var a = Var("a");
    Var b = Var("b");
    Array<Expr> m = arith::DetectLinearEquation(a * 4 + b * 6 + 7, { a });

    CHECK_EQ(m[0].as<IntImm>()->value, 4);
    Expr r = ir::Simplify(m[1] - (b * 6 + 7));
    CHECK_EQ(r.as<IntImm>()->value, 0);

    m = arith::DetectLinearEquation(a * 4 * (a + 1) + b * 6 + 7, { a });
    CHECK_EQ(m.size(), 0);

    m = arith::DetectLinearEquation(a * 4 + (a + 1) + b * 6 + 7, { a });
    CHECK_EQ(m[0].as<IntImm>()->value, 5);
    r = ir::Simplify(m[1] - (b * 6 + 7 + 1));
    CHECK_EQ(r.as<IntImm>()->value, 0);

    m = arith::DetectLinearEquation(a * b + 7, { a });
    CHECK(m[0].same_as(b));

    m = arith::DetectLinearEquation(b * 7, { a });
    CHECK_EQ(m[0].as<IntImm>()->value, 0);

    m = arith::DetectLinearEquation(b * 7, Array<Var>());
    CHECK_EQ(m.size(), 1);
    r = ir::Simplify(m[0] - b * 7);
    CHECK_EQ(r.as<IntImm>()->value, 0);
}

TEST(LANG_ARITH, test_dectect_linear_equation_multivariate)
{
    Array<Var> v;
    v.push_back(Var("v1"));
    v.push_back(Var("v2"));
    v.push_back(Var("v3"));
    v.push_back(Var("v4"));
    Var b = Var("b");

    Array<Expr> m = arith::DetectLinearEquation(v[0] * (b + 4) + v[0] + v[1] * 8, v);
    Expr r = ir::Simplify(m[0]);
    CHECK(ir::Equal(r, b + 5));
    CHECK_EQ(m[1].as<IntImm>()->value, 8);

    m = arith::DetectLinearEquation(v[0] * (b + 4) + v[0] + v[1] * 8 * v[2], v);
    CHECK_EQ(m.size(), 0);

    m = arith::DetectLinearEquation(v[0] * (b + 4) + v[0] + v[1] * 8 * v[1] + v[3], v);
    CHECK_EQ(m.size(), 0);

    m = arith::DetectLinearEquation(((v[0] * b + v[1]) * 8 + v[2] + 1) * 2, v);
    CHECK_EQ(m[1].as<IntImm>()->value, 16);
    CHECK_EQ(m[2].as<IntImm>()->value, 2);
    CHECK_EQ(m[m.size() - 1].as<IntImm>()->value, 2);

    m = arith::DetectLinearEquation((v[0] - v[1]), { v[2] });
    CHECK_EQ(m[0].as<IntImm>()->value, 0);
    r = ir::Simplify(m[1] - (v[0] - v[1]));
    CHECK_EQ(r.as<IntImm>()->value, 0);

    m = arith::DetectLinearEquation((v[0] - v[1]), {});
    CHECK_EQ(m.size(), 1);
    r = ir::Simplify(m[0] - (v[0] - v[1]));
    CHECK_EQ(r.as<IntImm>()->value, 0);
}
