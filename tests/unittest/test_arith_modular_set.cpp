#include "gtest/gtest.h"
#include <tvm/arithmetic.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>

using namespace tvm;
using namespace tvm::ir;

TEST(LANG_ARITH_MODULAR, cast)
{
    arith::Analyzer analyzer;
    Var x("x", Int(8));
    arith::ModularSet m = analyzer.modular_set(cast(UInt(32), x * 3));
    CHECK_EQ(m->coeff, 3);
    CHECK_EQ(m->base, 0);
    m = analyzer.modular_set(
        cast(Int(32), cast(Float(32), x * 3 + 1)));
    CHECK_EQ(m->coeff, 3);
    CHECK_EQ(m->base, 1);
}

TEST(LANG_ARITH_MODULAR, add_sub)
{
    arith::Analyzer analyzer;
    Var x("x", Int(64));
    Var y("y", Int(64));

    arith::ModularSet m = analyzer.modular_set(x * 6 + y * 4);
    CHECK_EQ(m->coeff, 2);
    CHECK_EQ(m->base, 0);

    analyzer.Bind(y, x * 4 + 1);

    m = analyzer.modular_set(1 - y);
    CHECK_EQ(m->coeff, 4);
    CHECK_EQ(m->base, 0);
}

TEST(LANG_ARITH_MODULAR, mul)
{
    arith::Analyzer analyzer;
    Var x("x");
    Var y("y");

    arith::ModularSet m = analyzer.modular_set((x * 4 + 2) * (y * 6 + 1));
    CHECK_EQ(m->coeff, 4);
    CHECK_EQ(m->base, 2);
}

TEST(LANG_ARITH_MODULAR, div_shift)
{
    arith::Analyzer analyzer;
    Var x("x");
    Var y("y");

    // # not sure if x is non-negative
    arith::ModularSet m = analyzer.modular_set((x * 4 + 2) / 2);
    CHECK_EQ(m->coeff, 1);
    CHECK_EQ(m->base, 0);

    // # right shift always round down so it is fine
    m = analyzer.modular_set((x * 4 + 2) >> 1);
    CHECK_EQ(m->coeff, 2);
    CHECK_EQ(m->base, 1);

    // # x is non-negative
    analyzer.const_int_bound.Update(x, arith::ConstIntBound(0, 100));
    m = analyzer.modular_set((x * 4 + 2) / 2);
    CHECK_EQ(m->coeff, 2);
    CHECK_EQ(m->base, 1);
}

TEST(LANG_ARITH_MODULAR, min_max_select)
{
    arith::Analyzer analyzer;
    Var x("x");
    Var y("y");

    arith::ModularSet m = analyzer.modular_set(min(x * 3, y * 9));
    CHECK_EQ(m->coeff, 3);
    CHECK_EQ(m->base, 0);

    m = analyzer.modular_set(max(x * 3 + 1, y * 9 + 4));
    CHECK_EQ(m->coeff, 3);
    CHECK_EQ(m->base, 1);

    m = analyzer.modular_set(ir::Select::make(x > 0, x * 3 + 1, y * 9 + 2));
    CHECK_EQ(m->coeff, 1);
    CHECK_EQ(m->base, 0);
}

TEST(LANG_ARITH_MODULAR, mix_index)
{
    arith::Analyzer analyzer;
    Var a("a");
    Var b("b");

    arith::ModularSet m = analyzer.modular_set(a * 4 + b * 6 + 7);
    CHECK_EQ(m->coeff, 2);
    CHECK_EQ(m->base, 1);

    m = analyzer.modular_set((a * 4 + 1) * (b * 8 + 3));
    CHECK_EQ(m->coeff, 4);
    CHECK_EQ(m->base, 3);

    m = analyzer.modular_set((a * 4 + 1) / (b * 8 + 3));
    CHECK_EQ(m->coeff, 1);
    CHECK_EQ(m->base, 0);

    m = analyzer.modular_set((a * 4 + 1) * (b * 8 / 4));
    CHECK_EQ(m->coeff, 2);
    CHECK_EQ(m->base, 0);

    m = analyzer.modular_set((a * 12 + 1) - (b * 3 * 7 + 2));
    CHECK_EQ(m->coeff, 3);
    CHECK_EQ(m->base, 2);

    m = analyzer.modular_set(a * 12 + min(b * 3 * 7, 2));
    CHECK_EQ(m->coeff, 1);
    CHECK_EQ(m->base, 0);
}

TEST(LANG_ARITH_MODULAR, constraint_scope)
{
    arith::Analyzer analyzer;
    Var a("a");
    Var b("b");
    arith::ModularSet m;
    // api_arith.cc define python api
    {
        With<arith::ConstraintContext> ctx(&analyzer, b % 4 == 2);
        m = analyzer.modular_set(b + 1);
        CHECK_EQ(m->coeff, 4);
        CHECK_EQ(m->base, 3);
        {
            With<arith::ConstraintContext> ctx2(&analyzer, a % 2 == 1);
            m = analyzer.modular_set(b + a * 2);
            CHECK_EQ(m->coeff, 4);
            CHECK_EQ(m->base, 0);
        }
        m = analyzer.modular_set(b + a * 2);
        CHECK_EQ(m->coeff, 2);
        CHECK_EQ(m->base, 0);
    }

    m = analyzer.modular_set(b + 1);
    CHECK_EQ(m->coeff, 1);
    CHECK_EQ(m->base, 0);
}

TEST(LANG_ARITH_MODULAR, intersect)
{
    arith::Analyzer analyzer;
    Var a("a");
    arith::ModularSet m;
    {
        With<arith::ConstraintContext> ctx(&analyzer, a % 4 == 1);
        {
            With<arith::ConstraintContext> ctx2(&analyzer, a % 3 == 1);
            m = analyzer.modular_set(a);
            CHECK_EQ(m->coeff, 12);
            CHECK_EQ(m->base, 1);
        }
    }

    {
        With<arith::ConstraintContext> ctx(&analyzer, a % 3 == 2);
        {
            With<arith::ConstraintContext> ctx2(&analyzer, a % 5 == 3);
            {
                With<arith::ConstraintContext> ctx3(&analyzer, a % 7 == 2);
                m = analyzer.modular_set(a);
                CHECK_EQ(m->coeff, 105);
                CHECK_EQ(m->base, 23);
            }
        }
    }
}
