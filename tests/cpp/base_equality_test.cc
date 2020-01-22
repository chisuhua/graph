#include "halide/ir/IREquality.h"
#include "halide/ir/IROperator.h"
#include "halide/ir/IRVisitor.h"
#include <gtest/gtest.h>

using namespace HalideIR;
using namespace HalideIR::Internal;

using std::string;
using std::vector;

// Testing code
/*
IRComparer::CmpResult flip_result(IRComparer::CmpResult r)
{
    switch (r) {
    case IRComparer::LessThan:
        return IRComparer::GreaterThan;
    case IRComparer::Equal:
        return IRComparer::Equal;
    case IRComparer::GreaterThan:
        return IRComparer::LessThan;
    case IRComparer::Unknown:
        return IRComparer::Unknown;
    }
    return IRComparer::Unknown;
}

void check_equal(Expr a, Expr b)
{
    IRCompareCache cache(5);
    IRComparer::CmpResult r = IRComparer(&cache).compare_expr(a, b);
    internal_assert(r == IRComparer::Equal)
        << "Error in ir_equality_test: " << r
        << " instead of " << IRComparer::Equal
        << " when comparing:\n"
        << a
        << "\nand\n"
        << b << "\n";
}

void check_not_equal(Expr a, Expr b)
{
    IRCompareCache cache(5);
    IRComparer::CmpResult r1 = IRComparer(&cache).compare_expr(a, b);
    IRComparer::CmpResult r2 = IRComparer(&cache).compare_expr(b, a);
    internal_assert(r1 != IRComparer::Equal && r1 != IRComparer::Unknown && flip_result(r1) == r2)
        << "Error in ir_equality_test: " << r1
        << " is not the opposite of " << r2
        << " when comparing:\n"
        << a
        << "\nand\n"
        << b << "\n";
}
*/


TEST(IREQUALITY, Basic)
{
    ir_equality_test();
        /*
    Expr x = Variable::make(Int(32), "x");
    check_equal(Ramp::make(x, 4, 3), Ramp::make(x, 4, 3));
    check_not_equal(Ramp::make(x, 2, 3), Ramp::make(x, 4, 3));

    // FIXME it wont't equil since it is different var
    // check_equal(x, Variable::make(Int(32), "x"));
    check_not_equal(x, Variable::make(Int(32), "y"));

    // Something that will hang if IREquality has poor computational
    // complexity.
    Expr e1 = x, e2 = x;
    for (int i = 0; i < 100; i++) {
        e1 = e1 * e1 + e1;
        e2 = e2 * e2 + e2;
    }
    check_equal(e1, e2);
    // These are only discovered to be not equal way down the tree:
    e2 = e2 * e2 + e2;
    check_not_equal(e1, e2);

    debug(0) << "ir_equality_test passed\n";
    */
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    return RUN_ALL_TESTS();
}
