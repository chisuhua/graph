#include "gtest/gtest.h"
// #include "ir/IROperator.h"
#include <tvm/arithmetic.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>

template <typename T>
class TD;

using namespace tvm;

TEST(LANG_PASS_OPT, test_simplify)
{
    auto dtype = Int(64);
    Var n("n");
    Var i("i");
    Var j("j");

    Buffer Ab = tvm::decl_buffer({ n }, dtype);
    Stmt stmt = ir::For::make(i, 2, n, ir::ForType::Serial, ir::DeviceAPI::Host,
        ir::For::make(j, 0, n, ir::ForType::Serial, ir::DeviceAPI::Host,
            ir::IfThenElse::make(ir::LT::make(i + 2, n),
                ir::Store::make(Ab->data, ir::Load::make(dtype, Ab->data, i + 4, const_true(1)) + 1, (j + 1) * 4 - 4 * j + i, const_true(1)))));
    LOG(INFO) << stmt;
    stmt = ir::CanonicalSimplify(stmt);
    LOG(INFO) << stmt;
}
/*
def test_simplify():
    """Not yet working, mock design"""
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    i = tvm.var('i')
    j = tvm.var('j')
    # for i in 0 to n-1:
    stmt = tvm.make.For(
        i, 2, n, 0, 0,
        tvm.make.For(j, 0, n, 0, 0,
                     tvm.make.IfThenElse(
                         tvm.make.LT(i + 2, n),
                         tvm.make.Store(Ab.data,
                                        tvm.make.Load(dtype, Ab.data, i + 4) + 1,
                                        (j + 1) * 4 - 4 * j + i),
                         None)))
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
*/
TEST(LANG_PASS_OPT, test_basic)
{
    Var m("m");
    auto stmt = ir::CanonicalSimplify(ir::Evaluate::make(m - 1));
    std::ostringstream os;
    os << stmt.as<ir::Evaluate>()->value;
    CHECK_EQ(os.str(), "(m - 1)");
}

TEST(LANG_PASS_OPT, test_bound)
{
    Var m("m");
    auto stmt = ir::CanonicalSimplify(ir::Evaluate::make(m - 1));
    std::ostringstream os;
    os << stmt.as<ir::Evaluate>()->value;
    CHECK_EQ(os.str(), "(m - 1)");
}

/* TODO
def test_bound():
    m = tvm.var('m')
    vrange = tvm.convert({m: tvm.Range(tvm.const(0, "int32"), tvm.const(10, "int32"))})
    ret = tvm.ir_pass.Simplify(m % 10, vrange)
    assert ret == m


*/
