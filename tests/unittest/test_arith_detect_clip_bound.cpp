#include "gtest/gtest.h"
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
// #include <arithmetic/Simplify.h>

template <typename T>
class TD;
using namespace tvm;
using namespace tvm::ir;

TEST(LANG_ARITH, test_dectect_clip_bound)
{
    Var a = Var("a");
    Var b = Var("b");
    Var c = Var("c");
    Array<Expr> m = arith::DetectClipBound(ir::And::make(a * 1 < b * 6, a - 1 > 0), { a });

    Expr r = ir::Simplify(m[1] - (b * 6 - 1));
    CHECK_EQ(r.as<IntImm>()->value, 0);

    CHECK_EQ(m[0].as<IntImm>()->value, 2);

    m = arith::DetectClipBound(ir::And::make(a * 1 < b * 6, a - 1 > 0), { a, b });
    CHECK_EQ(m.size(), 0);

    m = arith::DetectClipBound(ir::And::make(a + 10 * c <= 20, b - 1 > 0), { a, b });
    r = ir::Simplify(m[1] - (20 - 10 * c));
    CHECK_EQ(r.as<IntImm>()->value, 0);

    r = ir::Simplify(m[2] - 2);
    CHECK_EQ(r.as<IntImm>()->value, 0);
}
