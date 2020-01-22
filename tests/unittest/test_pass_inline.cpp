#include "gtest/gtest.h"
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>

template <typename T>
class TD;

using namespace tvm;

TEST(LANG_PASS_INLINE, test_inline)
{
    Var m("m");
    Tensor A = placeholder({ m });
    Tensor T = compute(
        { m },
        [&](Var i) {
            return A[i] + 10;
        },
        "T");
    Stmt stmt = ir::Evaluate::make(T[10] + 11 * T[100]);
    Array<Var> x;
    for (auto i : T->op.as<ComputeOpNode>()->axis) {
        x.push_back(i->var);
    }
    LOG(INFO) << stmt;
    stmt = ir::Inline(stmt, T->op, x, T->op.as<ComputeOpNode>()->body[0]);
    LOG(INFO) << stmt;
    CHECK(ir::VerifySSA(stmt));
}

TEST(LANG_PASS_INLINE, test_inline2)
{
    Var m("m");
    Tensor A = placeholder({ m });
    Tensor T = compute(
        { m },
        [&](Var i) {
            return A[i] + 10;
        },
        "T");
    Stmt stmt = ir::Evaluate::make(T[10] + 11 * T[100]);
    Array<Var> x;
    for (auto i : T->op.as<ComputeOpNode>()->axis) {
        x.push_back(i->var);
    }
    LOG(INFO) << stmt;
    stmt = ir::Inline(stmt, T->op, x, T->op.as<ComputeOpNode>()->body[0]);

    auto check = [&](const NodeRef& op) {
        if (op->is_type<ir::Call>()) {
            CHECK_NE(op.as<ir::Call>()->func, T->op);
        }
    };
    ir::PostOrderVisit(stmt, [check](const NodeRef& n) { check(n); });
}
