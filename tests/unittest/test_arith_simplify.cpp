#include "gtest/gtest.h"
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
// #include <arithmetic/Simplify.h>

template <typename T>
class TD;
using namespace tvm;
using namespace tvm::ir;

template <typename T>
Expr csimplify(T t)
{

    Stmt stmt = ir::Evaluate::make(t);
    const Evaluate* op = stmt.as<Evaluate>();
    // TD<decltype(ir::Evaluate::make(t).GetNodePtr().as<ir::Evaluate>()->value)> aType; // .as<ir::Evaluate>())> aType;
    return ir::CanonicalSimplify(op->value);
}

TEST(LANG_ARITH, test_simplify)
{
    Expr x = Var("x");
    auto z = x * 4 - x * 2;
    Expr zz = csimplify(z);
    LOG(INFO) << zz;
    CHECK_EQ(zz.as<ir::Mul>()->b.as<ir::IntImm>()->value, 2);

    z = (x / 4) * 2 - (x / 4);
    zz = csimplify(z);
    LOG(INFO) << zz;
    CHECK(zz.as<ir::Div>()->a.same_as(x));
    CHECK_EQ(zz.as<ir::Div>()->b.as<ir::IntImm>()->value, 4);

    z = (x % 4) * 3 + (x % 4);
    zz = csimplify(z);
    LOG(INFO) << zz;
    CHECK_EQ(zz.as<ir::Mul>()->b.as<ir::IntImm>()->value, 4);
    zz = zz.as<ir::Mul>()->a;
    CHECK(zz.as<ir::Mod>()->a.same_as(x));
    CHECK_EQ(zz.as<ir::Mod>()->b.as<ir::IntImm>()->value, 4);
    // assert zz.a == x and zz.b.value == 4

    Expr n = Var("n");
    Expr e = ir::CanonicalSimplify(n % 1);
    CHECK(ir::Equal(e, make_const(Int(32), 0)));

    e = ir::CanonicalSimplify(n / 1);
    CHECK(ir::Equal(e, n));
}

// FIXME need ir_builder
TEST(LANG_ARITH, test_simplify_mod)
{
    Expr A = Var("A", Float(32));
}

TEST(LANG_ARITH, test_simplify_minmax)
{
    Expr x = Var("x");
    Expr e1 = max(x, 1) - max(x, 1);
    Expr e1s = ir::CanonicalSimplify(e1);
    CHECK(e1s.as<IntImm>()->value == 0);

    Expr e2 = min(x, 1) - min(x, 1);
    Expr e2s = ir::CanonicalSimplify(e2);
    CHECK(e2s.as<IntImm>()->value == 0);
}

TEST(LANG_ARITH, test_mul)
{
    Expr x = Var("x");
    Expr e = x * x - x * x;
    Expr es = ir::CanonicalSimplify(e);
    CHECK(es.as<IntImm>()->value == 0);
}

TEST(LANG_ARITH, test_modular)
{
    Var rx = Var("rx");
    Var ry = Var("ry");
    Var y = Var("y");
    Var x = Var("x");

    auto i32_const = [](int x) { return make_const(Int(32), x); };

    Map<Var, Range> vmap { { rx, Range(i32_const(0), i32_const(3)) }, { ry, Range(i32_const(0), i32_const(3)) },
        { y, Range(i32_const(0), i32_const(2)) }, { x, Range(i32_const(0), i32_const(14)) } };

    Expr idx = ry * 16 + rx + y * 16 + x;
    Expr z1 = ir::CanonicalSimplify(idx / 16, vmap);
    Expr z2 = ir::CanonicalSimplify(idx % 16, vmap);

    LOG(INFO) << "vmap=" << vmap;
    LOG(INFO) << "z1=" << z1;
    LOG(INFO) << "z2=" << z2;

    Expr zz1 = ir::CanonicalSimplify(z1 - (ry + y));
    Expr zz2 = ir::CanonicalSimplify(z2 - (rx + x));

    LOG(INFO) << "z1 - (ry + y) =" << (z1 - (ry + y));
    LOG(INFO) << "z2 - (rx + x) =" << (z2 - (rx + x));

    LOG(INFO) << "zz1 =" << zz1;
    LOG(INFO) << "zz2 =" << zz2;

    CHECK(zz1.as<IntImm>()->value == 0);
    CHECK(zz2.as<IntImm>()->value == 0);
}

TEST(LANG_ARITH, test_const_propagation)
{
    Expr x1 = make_const(Int(32), 4);

    Expr x2 = x1 + 5;
    CHECK(x2->is_type<IntImm>());
    CHECK(x2.as<IntImm>()->value == 9);

    Expr x3 = x2 / 3;
    CHECK(x3->is_type<IntImm>());
    CHECK(x3.as<IntImm>()->value == 3);

    Expr x4 = x3 + make_const(Float(32), 0.5);
    CHECK(x4->is_type<FloatImm>());
    CHECK(x4.as<FloatImm>()->value == 3.5);

    Expr x5 = ceil(x4);
    CHECK(x5->is_type<FloatImm>());
    CHECK(x5.as<FloatImm>()->value == 4);

    Expr x6 = cast(Int(32), x5);
    CHECK(x6->is_type<IntImm>());
    CHECK(x6.as<IntImm>()->value == 4);

    Expr x7 = round((make_const(Float(32), 6.5) - 1) / make_const(Float(32), 1.5)) + 2;

    LOG(INFO) << x7;

    CHECK(cast(Int(32), x7).as<IntImm>()->value == 6);
}

/*
    y = (tvm.round((tvm.const(6.5, 'float32') - 1) / 1.5) + 2).astype('int')
    assert isinstance(y, tvm.expr.IntImm) and y.value == 6
*/
