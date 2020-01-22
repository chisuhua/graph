#include "gtest/gtest.h"
// #include "ir/IROperator.h"
#include <arithmetic/const_fold.h>
#include <tvm/arithmetic.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include <type_traits>
#include <vector>

template <typename T>
class TD;

using namespace tvm;

TEST(LANG_OPERATOR, test_const_fold)
{
    auto check = [](auto f, int a, int b) {
        Expr x = f(make_const(Int(32), a), make_const(Int(32), b));
        int y = f(a, b);
        // std::cout << x << std::endl;
        // std::cout << y << std::endl;
        // std::cout << x->type_key() << std::endl;
        if (x->type_key() == std::string("IntImm")) {
            assert(x.as<ir::IntImm>()->value == y);
        } else if (x->type_key() == std::string("UIntImm")) {
            assert(x.as<ir::UIntImm>()->value == (unsigned int)y);
        } else {
            assert(false);
        }
    };

    check([](auto x, auto y) { return x + y; }, 3, 4);
    check([](auto x, auto y) { return x * y; }, 3, 12);
    check([](auto x, auto y) { return x * y - 10; }, 3, 12);
    check([](auto x, auto y) { return x - y % 10; }, 3, 12);
    check([](auto x, auto y) { return x % y + 10; }, 100, 12);
    check([](auto x, auto y) { return x > y; }, 112, 128);
    check([](auto x, auto y) { return x < y; }, 112, 128);
    check([](auto x, auto y) { return x <= y; }, 112, 128);
    check([](auto x, auto y) { return x >= y; }, 112, 128);
    check([](auto x, auto y) { return (x | y) ^ 10; }, 112, 128);
}

TEST(LANG_OPERATOR, test_const_fold2)
{
    Var x("x");
    assert((x + 0).same_as(x));
    assert((0 + x).same_as(x));
    assert((x - 0).same_as(x));
    assert((x % 1).as<IntImm>()->value == 0);
    assert((x * 1).same_as(x));
    assert((1 * x).same_as(x));
    std::cout << (1 / x)->type_key() << std::endl;
    assert((1 / x)->type_key() == std::string("Div"));
}

TEST(LANG_OPERATOR, test_const_fold3)
{
    auto check_throws = [](auto f) {
        try {
            f();
            assert(false);
        } catch (dmlc::Error) {
            assert(true);
        }
    };
    {
        Var x("x");

        for (auto& val : { 0, 1 }) {
            check_throws([val, x]() { ir::Or::make(make_const(UInt(1), val), x); });
            check_throws([val, x]() { ir::Or::make(x, make_const(UInt(1), val)); });
            check_throws([val, x]() { ir::And::make(make_const(UInt(1), val), x); });
            check_throws([val, x]() { ir::And::make(x, make_const(UInt(1), val)); });
        }
    }

    // TODO auto and_func = ir::And();
    //  aoto call ConstFold ?
    auto and_ = [](auto a, auto b) { return a && b; };
    for (auto& v1 : { 0, 1 }) {
        for (auto& v2 : { 0, 1 }) {
            // auto and_func_result = and_func.make(make_const(UInt(1), v1), make_const(UInt(1), v2));
            auto and_func_result = arith::TryConstFold<ir::And>(make_const(UInt(1), v1), make_const(UInt(1), v2));
            auto and_result = make_const(UInt(1), and_(v1, v2));
            std::cout << "and_func_result: " << and_func_result << std::endl;
            std::cout << "and_result: " << and_result << std::endl;
            assert(ir::Equal(and_func_result, and_result));

            auto or_fold = arith::TryConstFold<ir::Or>(make_const(UInt(1), v1), make_const(UInt(1), v2));
            auto or_result = make_const(UInt(1), v1 | v2);
            assert(ir::Equal(or_fold, or_result));
        }
    }

    auto x = Var("x", UInt(1));
    auto True = make_const(UInt(1), 1);
    auto False = make_const(UInt(1), 0);

    assert(arith::TryConstFold<ir::And>(x, True).same_as(x));
    assert(arith::TryConstFold<ir::And>(True, x).same_as(x));
    assert(arith::TryConstFold<ir::Or>(x, False).same_as(x));
    assert(arith::TryConstFold<ir::Or>(False, x).same_as(x));

    assert(arith::TryConstFold<ir::And>(x, True).same_as(x));
    assert(arith::TryConstFold<ir::And>(True, x).same_as(x));
    assert(arith::TryConstFold<ir::Or>(x, False).same_as(x));
    assert(arith::TryConstFold<ir::Or>(False, x).same_as(x));
}
