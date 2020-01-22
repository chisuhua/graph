#include "halide/arithmetic/Interval.h"
#include "halide/ir/IREquality.h"
#include "halide/ir/IROperator.h"

#include <gtest/gtest.h>

using namespace HalideIR;
using namespace HalideIR::Internal;

void check(Interval result, Interval expected, int line)
{
    internal_assert(equal(result.min, expected.min) && equal(result.max, expected.max))
        << "Interval test on line " << line << " failed\n"
        << "  Expected [" << expected.min << ", " << expected.max << "]\n"
        << "  Got      [" << result.min << ", " << result.max << "]\n";
}

TEST(BASE_INTERVAL, Basic)
{
    Interval e = Interval::everything();
    Interval n = Interval::nothing();
    Expr x = Variable::make(Int(32), "x");
    Interval xp { x, Interval::pos_inf };
    Interval xn { Interval::neg_inf, x };
    Interval xx { x, x };

    internal_assert(e.is_everything());
    internal_assert(!e.has_upper_bound());
    internal_assert(!e.has_lower_bound());
    internal_assert(!e.is_empty());
    internal_assert(!e.is_bounded());
    internal_assert(!e.is_single_point());

    internal_assert(!n.is_everything());
    internal_assert(!n.has_upper_bound());
    internal_assert(!n.has_lower_bound());
    internal_assert(n.is_empty());
    internal_assert(!n.is_bounded());
    internal_assert(!n.is_single_point());

    internal_assert(!xp.is_everything());
    internal_assert(!xp.has_upper_bound());
    internal_assert(xp.has_lower_bound());
    internal_assert(!xp.is_empty());
    internal_assert(!xp.is_bounded());
    internal_assert(!xp.is_single_point());

    internal_assert(!xn.is_everything());
    internal_assert(xn.has_upper_bound());
    internal_assert(!xn.has_lower_bound());
    internal_assert(!xn.is_empty());
    internal_assert(!xn.is_bounded());
    internal_assert(!xn.is_single_point());

    internal_assert(!xx.is_everything());
    internal_assert(xx.has_upper_bound());
    internal_assert(xx.has_lower_bound());
    internal_assert(!xx.is_empty());
    internal_assert(xx.is_bounded());
    internal_assert(xx.is_single_point());

    check(Interval::make_union(xp, xn), e, __LINE__);
    check(Interval::make_union(e, xn), e, __LINE__);
    check(Interval::make_union(xn, e), e, __LINE__);
    check(Interval::make_union(xn, n), xn, __LINE__);
    check(Interval::make_union(n, xp), xp, __LINE__);
    check(Interval::make_union(xp, xp), xp, __LINE__);

    check(Interval::make_intersection(xp, xn), Interval::single_point(x), __LINE__);
    check(Interval::make_intersection(e, xn), xn, __LINE__);
    check(Interval::make_intersection(xn, e), xn, __LINE__);
    check(Interval::make_intersection(xn, n), n, __LINE__);
    check(Interval::make_intersection(n, xp), n, __LINE__);
    check(Interval::make_intersection(xp, xp), xp, __LINE__);

    check(Interval::make_union({ 3, Interval::pos_inf }, { 5, Interval::pos_inf }), { 3, Interval::pos_inf }, __LINE__);
    check(Interval::make_intersection({ 3, Interval::pos_inf }, { 5, Interval::pos_inf }), { 5, Interval::pos_inf }, __LINE__);

    check(Interval::make_union({ Interval::neg_inf, 3 }, { Interval::neg_inf, 5 }), { Interval::neg_inf, 5 }, __LINE__);
    check(Interval::make_intersection({ Interval::neg_inf, 3 }, { Interval::neg_inf, 5 }), { Interval::neg_inf, 3 }, __LINE__);

    check(Interval::make_union({ 3, 4 }, { 9, 10 }), { 3, 10 }, __LINE__);
    check(Interval::make_intersection({ 3, 4 }, { 9, 10 }), { 9, 4 }, __LINE__);

    check(Interval::make_union({ 3, 9 }, { 4, 10 }), { 3, 10 }, __LINE__);
    check(Interval::make_intersection({ 3, 9 }, { 4, 10 }), { 4, 9 }, __LINE__);

    std::cout << "Interval test passed" << std::endl;
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    return RUN_ALL_TESTS();
}
