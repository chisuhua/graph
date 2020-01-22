#include "gtest/gtest.h"
#include <tvm/arithmetic.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <compiler/arithmetic/const_fold.h>

using namespace tvm;
using namespace tvm::ir;

TEST(LANG_ARITH_INTSET, test_intset_basic)
{
    arith::IntSet s = arith::IntSet::interval(2, 3);
    CHECK_EQ(s.min().as<IntImm>()->value, 2);
    CHECK_EQ(s.max().as<IntImm>()->value, 3);
}

TEST(LANG_ARITH_INTSET, test_intset_vector)
{
    int base = 10;
    int stride = 3;
    int lanes = 2;
    arith::IntSet s = arith::IntSet::vector(ir::Ramp::make(base, stride, lanes));
    CHECK_EQ(s.min().as<IntImm>()->value, base);
    CHECK_EQ(s.max().as<IntImm>()->value, base + stride * lanes - 1);
}

TEST(LANG_ARITH_INTSET, test_intset_deduce)
{
    Var a("a"), b("b"), c("c"), d("d");
    arith::IntSet b_s = arith::IntSet::interval(2, 3);
    arith::IntSet c_s = arith::IntSet::interval(10, 15);
    arith::IntSet d_s = arith::IntSet::interval(-3, -1);

    Expr e0 = (-b) * a + c - d;
    arith::IntSet res0 = arith::DeduceBound(a, e0 >= 0, { { b, b_s }, { c, c_s }, { d, d_s } }, Map<Var, arith::IntSet>());
    Expr ans0 = ((d - c) / (-b));
    std::ostringstream os_res0, os_ans0;
    os_res0 << ir::Simplify(res0.max());
    os_ans0 << ans0;
    CHECK_EQ(os_res0.str(), os_ans0.str());

    e0 = d * a + c - d;
    res0 = arith::DeduceBound(a, e0 >= 0, { { b, b_s }, { c, c_s }, { d, d_s } }, Map<Var, arith::IntSet>());
    ans0 = ((0 - c) / d + 1);
    std::ostringstream o1_res0, o1_ans0;
    o1_res0 << ir::Simplify(res0.max());
    o1_ans0 << ans0;
    CHECK_EQ(o1_res0.str(), o1_ans0.str());

    Expr e1 = (a * 4 + b < c);
    arith::IntSet res1 = arith::DeduceBound(a, e1, { { b, b_s }, { c, c_s }, { d, d_s } }, Map<Var, arith::IntSet>());
    Expr ans1 = (((c - b) + -1) / 4);
    std::ostringstream o1_res1, o1_ans1;
    o1_res1 << ir::Simplify(res1.max());
    o1_ans1 << ans1;
    CHECK_EQ(o1_res1.str(), o1_ans1.str());

    Expr e2 = (max(5, a * 4) < c);
    arith::IntSet res2 = arith::DeduceBound(a, e2, { { b, b_s }, { c, c_s }, { d, d_s } }, Map<Var, arith::IntSet>());
    CHECK(res2.max().same_as(arith::neg_inf()));
    CHECK(res2.min().same_as(arith::pos_inf()));

    Expr e3 = (-b) + a * c - d;
    arith::IntSet res3 = arith::DeduceBound(a, e3 >= 0, { { b, b_s }, { c, c_s }, { d, d_s } }, { { b, b_s }, { d, d_s } });
    Expr ans3 = 2 / c + 1;
    Expr res3_s = ir::Simplify(res3.min());
    LOG(INFO) << res3;
    LOG(INFO) << res3_s;
    LOG(INFO) << ans3;
    std::ostringstream o3_res, o3_ans;
    o3_res << res3_s;
    o3_ans << ans3;
    CHECK_EQ(o3_res.str(), o3_ans.str());
}

TEST(LANG_ARITH_INTSET, test_intset_check)
{
    Var a("a"), b("b"), c("c"), d("d");
    arith::IntSet b_s = arith::IntSet::interval(2, 3);
    arith::IntSet c_s = arith::IntSet::interval(5, 7);
    arith::IntSet d_s = arith::IntSet::interval(-3, -1);

    arith::IntSet res1 = arith::DeduceBound(a, a + b, { { b, b_s } }, Map<Var, arith::IntSet>());
    CHECK(res1.is_nothing());

    // TODO arith::IntSet res2 = arith::DeduceBound(a, cast(c->dtype, ((a+b)>3))>c, {{b, b_s}, {c, c_s}}, Map<Var, arith::IntSet>());
    // CHECK(res2.is_nothing());

    arith::IntSet res3 = arith::DeduceBound(a, a * 2 - a > b, { { b, b_s } }, Map<Var, arith::IntSet>());
    CHECK(res3.is_nothing());
}

TEST(LANG_ARITH_INTSET, test_deduce_basic)
{
    auto test_basic = [](auto a1, auto a2, auto coff) {
        Var a("a"), b("b");
        arith::IntSet b_s = arith::IntSet::interval(a1, a2);
        auto e0 = b + a * coff + 3;
        Expr x, y;

        arith::IntSet res1 = arith::DeduceBound(a, e0 < 17, { { b, b_s } }, { { b, b_s } });
        if (coff > 0) {
            x = res1.max();
            y = b_s.max();
        } else {
            x = res1.min();
            y = b_s.min();
        }
        Expr r = ir::Simplify((x * coff + 3 + y) < 17);
        CHECK_EQ(r.as<UIntImm>()->value, 1);

        res1 = arith::DeduceBound(a, e0 > 17, { { b, b_s } }, { { b, b_s } });
        if (coff < 0) {
            x = res1.max();
            y = b_s.max();
        } else {
            x = res1.min();
            y = b_s.min();
        }
        r = ir::Simplify((x * coff + 3 + y) > 17);
        CHECK_EQ(r.as<UIntImm>()->value, 1);

        res1 = arith::DeduceBound(a, e0 <= 17, { { b, b_s } }, { { b, b_s } });
        if (coff > 0) {
            x = res1.max();
            y = b_s.max();
        } else {
            x = res1.min();
            y = b_s.min();
        }
        r = ir::Simplify((x * coff + 3 + y) <= 17);
        CHECK_EQ(r.as<UIntImm>()->value, 1);

        res1 = arith::DeduceBound(a, e0 >= 17, { { b, b_s } }, { { b, b_s } });
        if (coff < 0) {
            x = res1.max();
            y = b_s.max();
        } else {
            x = res1.min();
            y = b_s.min();
        }
        r = ir::Simplify((x * coff + 3 + y) >= 17);
        CHECK_EQ(r.as<UIntImm>()->value, 1);
    };
    test_basic(0, 4, 4);
    test_basic(1, 5, 5);
    test_basic(2, 6, 4);
    test_basic(0, 4, -4);
    test_basic(1, 5, -4);
    test_basic(2, 6, -4);
}

TEST(LANG_ARITH_INTSET, test_deduce_complex)
{
    auto test_complex = [](auto a1, auto a2, auto coff) {
        Var a("a"), b("b");
        arith::IntSet b_s = arith::IntSet::interval(a1, a2);
        auto e0 = (b * 3 + a * coff) * 4;
        Expr x, t;

        arith::IntSet res1 = arith::DeduceBound(a, e0 < 63, { { b, b_s } }, { { b, b_s } });
        if (coff > 0) {
            t = res1.max();
            x = b_s.max();
        } else {
            t = res1.min();
            x = b_s.min();
        }
        Expr r = ir::Simplify(((x * 3 + t * coff) * 4) < 63);
        CHECK_EQ(r.as<UIntImm>()->value, 1);

        res1 = arith::DeduceBound(a, e0 <= 63, { { b, b_s } }, { { b, b_s } });
        if (coff > 0) {
            t = res1.max();
            x = b_s.max();
        } else {
            t = res1.min();
            x = b_s.min();
        }
        r = ir::Simplify((x * 3 + t * coff) * 4 <= 63);
        CHECK_EQ(r.as<UIntImm>()->value, 1);

        res1 = arith::DeduceBound(a, e0 > 63, { { b, b_s } }, { { b, b_s } });
        if (coff < 0) {
            t = res1.max();
            x = b_s.max();
        } else {
            t = res1.min();
            x = b_s.min();
        }
        r = ir::Simplify((x * 3 + t * coff) * 4 > 63);
        CHECK_EQ(r.as<UIntImm>()->value, 1);

        res1 = arith::DeduceBound(a, e0 >= 63, { { b, b_s } }, { { b, b_s } });
        if (coff < 0) {
            t = res1.max();
            x = b_s.max();
        } else {
            t = res1.min();
            x = b_s.min();
        }
        r = ir::Simplify((x * 3 + t * coff) * 4 >= 63);
        CHECK_EQ(r.as<UIntImm>()->value, 1);
    };
    test_complex(0, 4, 4);
    test_complex(0, 4, -4);
    test_complex(2, 6, 4);
    test_complex(0, 4, -4);
    test_complex(1, 5, -4);
    test_complex(2, 6, -4);
}
