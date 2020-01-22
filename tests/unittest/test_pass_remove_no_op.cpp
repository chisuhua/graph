#include "gtest/gtest.h"
#include <tvm/build_module.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/packed_func_ext.h>
#include <tvm/schedule_pass.h>

template <typename T>
class TD;

using namespace tvm;

TEST(LANG_PASS_OPT, test_remove_no_op)
{
    Var i("i");
    Var j("j");
    Var k("k");
    Var m("m");
    Var n("n");
    auto dtype = Int(64);
    Buffer Ab = tvm::decl_buffer({ n }, dtype);

    Stmt stmt = ir::For::make(i, 0, 4, ir::ForType::Serial, ir::DeviceAPI::Host,
        ir::For::make(j, 0, n, ir::ForType::Serial, ir::DeviceAPI::Host,
            ir::For::make(k, 0, m, ir::ForType::Serial, ir::DeviceAPI::Host,
                ir::IfThenElse::make(likely(i * m + j + k < n), ir::Evaluate::make(m), ir::Evaluate::make(n)))));

    LOG(INFO) << "orig stmt:\n"
              << stmt;
    Stmt ret = ir::RemoveNoOp(stmt);
    LOG(INFO) << "remove op stmt:\n"
              << ret;
    CHECK(ret->is_type<ir::Evaluate>());

    Stmt store = ir::Store::make(Ab->data, ir::Load::make(dtype, Ab->data, i, const_true(1)) + 1, i + 1, const_true(1));
    LOG(INFO) << "store stmt:\n"
              << store;

    Stmt stmt2 = ir::Block::make(stmt, store);
    ret = ir::RemoveNoOp(stmt2);
    CHECK(ret->is_type<ir::Store>());
}
