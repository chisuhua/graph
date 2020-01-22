// #include <pybind11/pybind11.h>
// #include <iostream>
#include "gtest/gtest.h"
// #include "ir/IROperator.h"
#include <tvm/attrs.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
// #include <tvm/ir_base.h>
#include <iostream>

using namespace tvm;

template <typename T>
class TD;

TEST(LANG_BASIC, test_const)
{
    // auto x = HalideIR::Internal::make_const(HalideIR::Int(32), 1);
    auto x = tvm::make_const(Int(32), 1);
    std::cout << x.get()->type_key() << std::endl;
    assert(x.type().is_int());
    assert(x.type().bits() == 32);
    assert(x.type().lanes() == 1);

    // NOTE:
    //     x.type(）得到一个Type, 这个Type是node的类型
    // assert(x.get()->is_type<HalideIR::Internal::IntImm>());
    // assert(x.type() == type_of<int32_t>());
    assert(x.type() == Int(32));
    // TD<decltype(x.type())> xType;
    // TD<decltype(HalideIR::type_of<uint32_t>())> yType;
}

TEST(LANG_BASIC, test_make)
{
    auto x = tvm::make_const(Int(32), 1);
    auto y = tvm::Var("x");
    auto z = x + y;
    // assert(tvm::max(x, y).get()->is_type<HalideIR::Max::make
    auto m = tvm::max(x, y);
    assert(m.get()->is_type<tvm::ir::Max>());
    auto n = tvm::min(x, y);
    assert(n.get()->is_type<tvm::ir::Min>());
}

TEST(LANG_BASIC, test_ir)
{
    auto x = tvm::make_const(Int(32), 1);
    auto y = tvm::IntImm::make(Int(32), 1);
    auto z = x + y;
    auto stmt = tvm::ir::Evaluate::make(z);
    // std::cout << stmt.get()->type_index() << std::endl;
    // std::cout << stmt.get()->type_key() << std::endl;
    // TD<decltype(stmt.get()->type_index())> sType;
    assert(stmt.get()->is_type<tvm::ir::Evaluate>());
    // assert isinstance(stmt, tvm.stmt.Evaluate)
}

TEST(LANG_BASIC, test_ir2)
{
    auto x = tvm::Var("n");
    // or auto a = HalideIR::Internal::Variable::make(HalideIR::Handle(), "array");
    VarExpr a = tvm::Var("array", Handle());
    auto st = tvm::ir::Store::make(a, x + 1, tvm::IntImm::make(Int(32), 1), tvm::const_true());
    const tvm::ir::Store* node = st.as<tvm::ir::Store>();
    //
    assert(st.get()->is_type<tvm::ir::Store>());
    // assert isinstance(st, tvm.stmt.Store)
    // TD<decltype(node->buffer_var)> sType;
    // TD<decltype(a)> aType;
    // assert( HalideIR::Internal::ExprEqual(node->buffer_var, a));
    assert((node->buffer_var).same_as(a));
}

TEST(LANG_BASIC, test_let)
{
    auto x = tvm::Var("x");
    auto y = tvm::Var("y");
    auto stmt = tvm::ir::LetStmt::make(
        x, tvm::IntImm::make(Int(32), 10),
        tvm::ir::Evaluate::make(x + 1));
}

TEST(LANG_BASIC, test_cast)
{
    auto x = tvm::Var("x", Float(32));
    // or auto y = HalideIR::Internal::Cast::make(HalideIR::Int(32), x);
    auto y = tvm::cast(Int(32), x);
    auto z = tvm::cast(Float(32, 4), x);
    assert(y.get()->is_type<tvm::ir::Cast>());
    assert(z.get()->is_type<tvm::ir::Broadcast>());
    assert(z.type().lanes() == 4);
}

TEST(LANG_BASIC, test_attr)
{
    auto x = tvm::Var("x");
    auto y = tvm::Var("y");
    auto stmt = tvm::ir::AttrStmt::make(
        y, "stride", 10, tvm::ir::Evaluate::make(x + 1));
    const tvm::ir::AttrStmt* attr_stmt = stmt.as<tvm::ir::AttrStmt>();
    assert(attr_stmt->node == y);

    // tvm.convert(1) is implemented in python/tvm/_ffi/node_generic.py
    auto a = tvm::make_const(Int(32), 1);
    assert(a.as<tvm::ir::IntImm>()->value == 1);
    // TODO
    /*
    try {
        // a.as<HalideIR::Internal::IntImm>()->no_field;
        LOG(FATAL) << "bad";
    } catch (const tvm::AttrError& e) {
    }
    */
}

/*
def test_attr():
    x = tvm.var('x')
    y = tvm.var('y')
    stmt = tvm.make.AttrStmt(
        y, "stride", 10, tvm.make.Evaluate(x + 1));
    assert stmt.node == y

    a = tvm.convert(1)
    assert a.value == 1
    try:
        a.no_field
        assert False
    except AttributeError:
        pass
*/

TEST(LANG_BASIC, test_basic)
{
    auto a = tvm::Var("a");
    auto b = tvm::Var("b");
    auto c = a + b;
    std::ostringstream os;
    os << c;
    assert(os.str() == "(a + b)");
}

TEST(LANG_BASIC, test_stmt)
{
    auto x = tvm::ir::Evaluate::make(0);
    tvm::ir::For::make((VarExpr)tvm::Var("i"),
            tvm::make_const(Int(32), 0),
            tvm::make_const(Int(32), 1),
        ir::ForType::Serial, ir::DeviceAPI::None, x);
}

TEST(LANG_BASIC, test_dir)
{
    auto x = tvm::Var("x");
    // TODO print internal field
    std::cout << x << std::endl;
}

/*
def test_dir():
    x = tvm.var('x')
    dir(x)
*/

TEST(LANG_BASIC, test_dtype)
{
    auto x = tvm::Var("x");
    assert(x.type() == Int(32));
    auto y = tvm::Var("y");
    // assert((x > y).type() == HalideIR::type_of<bool>());
    assert((x > y).type() == Bool(1));
}

/*
def test_dtype():
    x = tvm.var('x')
    assert x.dtype == 'int32'
    y = tvm.var('y')
    assert (x > y).dtype == 'bool'
*/

TEST(LANG_BASIC, test_any)
{
    auto x = tvm::Var("x");
    auto y = tvm::Var("y");
    auto z = tvm::Var("z");
    try {
        auto t = x || x;
        assert(false);
    } catch (dmlc::Error) {
        assert(true);
    }

    auto t = x < y;
    std::ostringstream os;
    os << t;
    assert(os.str() == "(x < y)");

    auto t2 = x < y || y > z + 1 || x < z * 2;
    std::ostringstream os2;
    os2 << t2;
    std::cout << os2.str() << std::endl;
    assert(os2.str() == "(((x < y) || (y > (z + 1))) || (x < (z*2)))");

    auto t3 = tvm::ir::Or::make(x<y, y> z + 1);
    auto t4 = tvm::ir::Or::make(t3, x < z * 2);
    std::ostringstream os3;
    os3 << t4;
    assert(os2.str() == "(((x < y) || (y > (z + 1))) || (x < (z*2)))");
}

TEST(LANG_BASIC, test_all)
{
    auto x = tvm::Var("x");
    auto y = tvm::Var("y");
    auto z = tvm::Var("z");
    try {
        auto t = x and x;
        assert(false);
    } catch (dmlc::Error) {
        assert(true);
    }
    /*  python tvm::all accept on args
    auto t = HalideIR::Internal::And::make(x < y);
    std::ostringstream os;
    os << t;
    assert(os.str() == "(x < y)");
    */

    auto t2 = x < y && x > z;
    std::ostringstream os2;
    os2 << t2;
    std::cout << os2.str() << std::endl;
    assert(os2.str() == "((x < y) && (x > z))");

    auto t3 = tvm::ir::And::make(x<y, y> z + 1);
    auto t4 = tvm::ir::And::make(t3, x < z * 2);
    std::ostringstream os3;
    os3 << t4;
    std::cout << t4 << std::endl;
    assert(os3.str() == "(((x < y) && (y > (z + 1))) && (x < (z*2)))");
}

TEST(LANG_BASIC, test_bitwise)
{
    auto x = tvm::Var("x");
    auto y = tvm::Var("y");

    std::ostringstream os1, os2, os3, os4, os5, os6, os7, os8, os9, os10;
    os1 << (x << y);
    std::cout << os1.str() << std::endl;
    assert(os1.str() == "shift_left(x, y)");
    // assert str(x << y) == 'shift_left(x, y)'
    os2 << (x >> y);
    assert(os2.str() == "shift_right(x, y)");

    os3 << (x & y);
    assert(os3.str() == "bitwise_and(x, y)");

    os4 << (x | y);
    assert(os4.str() == "bitwise_or(x, y)");

    os5 << (x ^ y);
    assert(os5.str() == "bitwise_xor(x, y)");

    os6 << (~x);
    assert(os6.str() == "bitwise_not(x)");

    auto a = tvm::make_const(Int(8, 2), 1);
    os7 << (a >> 1).type();
    assert(os7.str() == "int8x2");

    auto b = tvm::make_const(Int(32, 2), 1);
    os8 << (x >> b).type();
    assert(os8.str() == "int32x2");

    auto z = tvm::Var("z", Int(8, 2));
    os9 << (z << tvm::make_const(Int(8, 2), 1)).type();
    assert(os9.str() == "int8x2");
}
TEST(LANG_BASIC, test_equality)
{
    auto a = tvm::Var("a");
    auto b = tvm::Var("b");
    auto c = (a == b);
    // TODO
    // cut = (!c).as<HalideIR::Internal::Bool>()->value;
    // assert(!c)
}
/*

def test_equality():
    a = tvm.var('a')
    b = tvm.var('b')
    c = (a == b)
    assert not c
    d = (c != c)
    assert not d

*/
