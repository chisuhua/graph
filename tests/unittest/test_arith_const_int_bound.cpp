#include "gtest/gtest.h"
#include <tvm/arithmetic.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
// #include <arithmetic/Simplify.h>

template <typename T>
class TD;
using namespace tvm;
using namespace tvm::ir;

constexpr int64_t kNegInf = arith::ConstIntBoundNode::kNegInf;
constexpr int64_t kPosInf = arith::ConstIntBoundNode::kPosInf;

TEST(LANG_ARITH, test_dtype_bound)
{
    arith::Analyzer analyzer;

    {
        Var x("x", Int(64));
        arith::ConstIntBound bd = analyzer.const_int_bound(x);
        CHECK_EQ(bd->min_value, kNegInf);
        CHECK_EQ(bd->max_value, kPosInf);
    }

    {
        Var x("x", Int(8));
        arith::ConstIntBound bd = analyzer.const_int_bound(x);
        CHECK_EQ(bd->min_value, -128);
        CHECK_EQ(bd->max_value, 127);
    }

    {
        Var x("x", UInt(8));
        arith::ConstIntBound bd = analyzer.const_int_bound(x);
        CHECK_EQ(bd->min_value, 0);
        CHECK_EQ(bd->max_value, 255);
    }
}

TEST(LANG_ARITH, test_cast_bound)
{
    arith::Analyzer analyzer;
    Expr x = Var("x", Int(8));
    Expr y = x % 3;
    {
        arith::ConstIntBound bd = analyzer.const_int_bound(cast(UInt(32), y));
        LOG(INFO) << y;
        CHECK_EQ(bd->min_value, 0);
        CHECK_EQ(bd->max_value, 2);
    }
    {
        arith::ConstIntBound bd = analyzer.const_int_bound(cast(Int(32), cast(Float(32), y)));
        LOG(INFO) << y;
        CHECK_EQ(bd->min_value, -2);
        CHECK_EQ(bd->max_value, 2);
    }
}

TEST(LANG_ARITH, test_add_sub_bound)
{
    arith::Analyzer analyzer;
    Var x = Var("x", Int(64));
    Var y = Var("y", Int(64));
    arith::ConstIntBound bd = analyzer.const_int_bound(x + y);
    CHECK_EQ(bd->min_value, kNegInf);
    CHECK_EQ(bd->max_value, kPosInf);

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(0, 4));
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(1, 10));
    bd = analyzer.const_int_bound(x + y);
    CHECK_EQ(bd->min_value, 1);
    CHECK_EQ(bd->max_value, 14);

    bd = analyzer.const_int_bound(x - y);
    CHECK_EQ(bd->min_value, -10);
    CHECK_EQ(bd->max_value, 3);

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(0, kPosInf), true);
    bd = analyzer.const_int_bound(x - y);
    CHECK_EQ(bd->min_value, -10);
    CHECK_EQ(bd->max_value, kPosInf);

    bd = analyzer.const_int_bound(1 - x);
    CHECK_EQ(bd->min_value, kNegInf);
    CHECK_EQ(bd->max_value, 1);
}

TEST(LANG_ARITH, test_mul_bound)
{
    arith::Analyzer analyzer;
    Var x = Var("x");
    Var y = Var("y");

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(-2, 4));
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(4, 10));
    arith::ConstIntBound bd = analyzer.const_int_bound(x * y + 20);
    CHECK_EQ(bd->min_value, 0);
    CHECK_EQ(bd->max_value, 60);

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(-3, 4), true);
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(-8, 2), true);
    bd = analyzer.const_int_bound(x * y);
    CHECK_EQ(bd->min_value, -32);
    CHECK_EQ(bd->max_value, 24);

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(kNegInf, 4), true);
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(-8, 2), true);
    bd = analyzer.const_int_bound(x * y);
    CHECK_EQ(bd->min_value, kNegInf);
    CHECK_EQ(bd->max_value, kPosInf);
}

TEST(LANG_ARITH, test_div_bound)
{
    arith::Analyzer analyzer;
    Var x = Var("x");
    Var y = Var("y");

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(-9, 4));
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(4, 10));
    arith::ConstIntBound bd = analyzer.const_int_bound(x / y);
    CHECK_EQ(bd->min_value, -2);

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(-9, 4), true);
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(-2, 0), true);
    bd = analyzer.const_int_bound(x / y);
    CHECK_EQ(bd->min_value, -4);
    CHECK_EQ(bd->max_value, 9);

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(kNegInf, 4), true);
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(-2, 1), true);
    bd = analyzer.const_int_bound(x / y);
    CHECK_EQ(bd->min_value, kNegInf);
    CHECK_EQ(bd->max_value, kPosInf);
}

TEST(LANG_ARITH, test_mod_bound)
{
    arith::Analyzer analyzer;
    Var x = Var("x");
    Var y = Var("y");

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(-9, 4));
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(4, 10));
    arith::ConstIntBound bd = analyzer.const_int_bound(x % y);
    CHECK_EQ(bd->min_value, -9);
    CHECK_EQ(bd->max_value, 4);

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(kNegInf, kPosInf), true);
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(4, 10), true);
    bd = analyzer.const_int_bound(x % y);
    CHECK_EQ(bd->min_value, -9);
    CHECK_EQ(bd->max_value, 9);

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(1, kPosInf), true);
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(4, 10), true);
    bd = analyzer.const_int_bound(x % y);
    CHECK_EQ(bd->min_value, 0);
    CHECK_EQ(bd->max_value, 9);
}

TEST(LANG_ARITH, test_min_max_bound)
{
    arith::Analyzer analyzer;
    Var x = Var("x");
    Var y = Var("y");

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(-9, 11));
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(4, 10));
    arith::ConstIntBound bd = analyzer.const_int_bound(min(x, y));
    CHECK_EQ(bd->min_value, -9);
    CHECK_EQ(bd->max_value, 10);

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(kNegInf, kPosInf), true);
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(4, 10), true);
    bd = analyzer.const_int_bound(min(x, y));
    CHECK_EQ(bd->min_value, kNegInf);
    CHECK_EQ(bd->max_value, 10);

    bd = analyzer.const_int_bound(max(x, y));
    CHECK_EQ(bd->min_value, 4);
    CHECK_EQ(bd->max_value, kPosInf);

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(1, kPosInf), true);
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(4, 10), true);
    bd = analyzer.const_int_bound(max(x, y));
    CHECK_EQ(bd->min_value, 4);
    CHECK_EQ(bd->max_value, kPosInf);
}

TEST(LANG_ARITH, test_select_bound)
{
    arith::Analyzer analyzer;
    Var x = Var("x");
    Var y = Var("y");

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(-9, 11));
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(4, 10));
    arith::ConstIntBound bd = analyzer.const_int_bound(
        ir::Select::make(x > 1, cast(Int(32), (y < 0)), y + 1));
    CHECK_EQ(bd->min_value, 0);
    CHECK_EQ(bd->max_value, 11);
}

TEST(LANG_ARITH, test_shift_and_bound)
{
    arith::Analyzer analyzer;
    Var x = Var("x");
    Var y = Var("y");

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(-9, 11));
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(2, 10));
    arith::ConstIntBound bd = analyzer.const_int_bound(x >> y);
    CHECK_EQ(bd->min_value, -3);
    CHECK_EQ(bd->max_value, 2);

    bd = analyzer.const_int_bound(x & y);
    CHECK_EQ(bd->min_value, 0);
    CHECK_EQ(bd->max_value, 10);

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(10, 11), true);
    bd = analyzer.const_int_bound(x & y);
    CHECK_EQ(bd->min_value, 0);
    CHECK_EQ(bd->max_value, 10);
}

TEST(LANG_ARITH, test_mix_index_bound)
{
    arith::Analyzer analyzer;
    Var x = Var("x");
    Var y = Var("y");

    analyzer.const_int_bound.Update(x, arith::ConstIntBound(0, 24 - 1));
    analyzer.const_int_bound.Update(y, arith::ConstIntBound(0, 3 - 1));

    arith::ConstIntBound bd = analyzer.const_int_bound((x % 8) + (x / 8) * 8);
    CHECK_EQ(bd->min_value, 0);
    CHECK_EQ(bd->max_value, 24 - 1);

    bd = analyzer.const_int_bound(y + x * 3);
    CHECK_EQ(bd->min_value, 0);
    CHECK_EQ(bd->max_value, 24 * 3 - 1);

    bd = analyzer.const_int_bound((x % 7) + (x / 7) * 7);
    CHECK_EQ(bd->min_value, 0);
    CHECK_EQ(bd->max_value, (23 / 7) * 7 + 6);
}
