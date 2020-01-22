#include "gtest/gtest.h"
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

TEST(LANG_PASS_INJECT, test_copy2d)
{
    Var m("m");
    Var l("l");
    Tensor A = placeholder({ m, l });
    Tensor B = compute(
        { m, l }, [&](Var i, Var j) { return A[i][j]; }, "B");
    Schedule s = create_schedule({ B->op });
    s[B].pragma(B->op.as<ComputeOpNode>()->axis[0], "memcpy");
    Map<IterVar, Range> bounds = schedule::InferBound(s);
    auto stmt = schedule::ScheduleOps(s, bounds, false);

    Buffer Ab = tvm::decl_buffer(A->shape, HalideIR::Float(32));
    Buffer Bb = tvm::decl_buffer(B->shape, HalideIR::Float(32));

    LOG(INFO) << stmt;
    stmt = ir::StorageFlatten(stmt, { { A, Ab }, { B, Bb } }, 64); // , config->instrument_bound_checkers);
    LOG(INFO) << stmt;

    using CheckFunc = runtime::TypedPackedFunc<Stmt(Buffer, Buffer, Array<Expr>, Array<Expr>, Expr)>;
    CheckFunc checkfunc = [&](Buffer src, Buffer dst, Array<Expr> pad_before, Array<Expr> pad_after, Expr pad_value) {
        CHECK(dst->strides[0].same_as(l));
        CHECK_EQ(dst->strides[1].as<IntImm>()->value, 1);
        CHECK(src->strides[0].same_as(l));
        CHECK(src->shape[0].same_as(m));
        CHECK(src->shape[1].same_as(l));
        return ir::Evaluate::make(0);
    };

    stmt = ir::InjectCopyIntrin(stmt, "memcpy", checkfunc);
    LOG(INFO) << stmt;
}

TEST(LANG_PASS_INJECT, test_copy_pad)
{
    Var m("m");
    Var l("l");
    Tensor A = placeholder({ m, l });
    Tensor B = compute(
        { m + 2, l }, [&](Var i, Var j) { return if_then_else(ir::And::make(i >= 1, i < m + 1),
                                              A[i - 1][j], make_const(Float(32), 1.0)); }, "B");
    Schedule s = create_schedule({ B->op });
    s[B].pragma(B->op.as<ComputeOpNode>()->axis[0], "memcpy");

    Map<IterVar, Range> bounds = schedule::InferBound(s);

    auto stmt = schedule::ScheduleOps(s, bounds, false);

    Buffer Ab = tvm::decl_buffer(A->shape, HalideIR::Float(32));
    Buffer Bb = tvm::decl_buffer(B->shape, HalideIR::Float(32));

    LOG(INFO) << stmt;
    stmt = ir::StorageFlatten(stmt, { { A, Ab }, { B, Bb } }, 64); // , config->instrument_bound_checkers);
    LOG(INFO) << stmt;

    using CheckFunc = runtime::TypedPackedFunc<Stmt(Buffer, Buffer, Array<Expr>, Array<Expr>, Expr)>;

    CheckFunc checkfunc = [&](Buffer src, Buffer dst, Array<Expr> pad_before, Array<Expr> pad_after, Expr pad_value) {
        auto elem_offset = ir::Simplify(src->elem_offset).as<IntImm>()->value;
        CHECK_EQ(elem_offset, 0);
        CHECK_EQ(pad_before[0].as<IntImm>()->value, 1);
        CHECK_EQ(pad_before[1].as<IntImm>()->value, 0);
        CHECK_EQ(pad_after[0].as<IntImm>()->value, 1);
        CHECK_EQ(pad_after[1].as<IntImm>()->value, 0);
        CHECK_EQ(pad_value.as<ir::FloatImm>()->value, 1.0);
        // LOG(INFO) << pad_value->type_key();
        return ir::Evaluate::make(0);
    };

    stmt = ir::InjectCopyIntrin(stmt, "memcpy", checkfunc);
    LOG(INFO) << stmt;
}

TEST(LANG_PASS_INJECT, test_single_point_test)
{
    Tensor A = placeholder({ 1 });
    Tensor B = compute(
        { 1 }, [&](Var i) { return A[i]; }, "B");
    Schedule s = create_schedule({ B->op });
    s[B].pragma(B->op.as<ComputeOpNode>()->axis[0], "memcpy");

    Map<IterVar, Range> bounds = schedule::InferBound(s);

    auto stmt = schedule::ScheduleOps(s, bounds, false);
    LOG(INFO) << stmt;

    Buffer Ab = tvm::decl_buffer(A->shape, A->dtype);
    Buffer Bb = tvm::decl_buffer(B->shape, B->dtype);

    stmt = ir::StorageFlatten(stmt, { { A, Ab }, { B, Bb } }, 64); // , config->instrument_bound_checkers);
    LOG(INFO) << stmt;

    using CheckFunc = runtime::TypedPackedFunc<Stmt(Buffer, Buffer, Array<Expr>, Array<Expr>, Expr)>;

    CheckFunc checkfunc = [&](Buffer src, Buffer dst, Array<Expr> pad_before, Array<Expr> pad_after, Expr pad_value) {
        auto elem_offset = ir::Simplify(src->elem_offset).as<IntImm>()->value;
        CHECK_EQ(elem_offset, 0);

        elem_offset = ir::Simplify(dst->elem_offset).as<IntImm>()->value;
        CHECK_EQ(elem_offset, 0);

        auto stride = ir::Simplify(src->strides[0]);
        CHECK_EQ(stride.as<IntImm>()->value, 1);

        stride = ir::Simplify(dst->strides[0]);
        CHECK_EQ(stride.as<IntImm>()->value, 1);

        return ir::Evaluate::make(0);
    };

    stmt = ir::InjectCopyIntrin(stmt, "memcpy", checkfunc);
}

auto assert_expr_equal = [](Expr a, Expr b) {
    // print(a, b)
    CHECK(ir::Simplify(a - b).as<IntImm>()->value == 0);
};

TEST(LANG_PASS_INJECT, test_copy_pad_split)
{
    auto m = 4 * 3;
    Tensor A = placeholder({ m });
    Tensor Apad = compute(
        { m + 2 }, [&](Var i) { return if_then_else(ir::And::make(i >= 1, i <= m),
                                    A[i - 1], make_const(Float(32), 0.0)); }, "Apad");
    Tensor B = compute({ m }, [&](Var i) { return Apad[i] + Apad[i + 1] + Apad[i + 2]; });
    LOG(INFO) << B;

    Schedule s = create_schedule({ B->op });
    LOG(INFO) << s;

    IterVar xo, xi;
    s[B].split(B->op.as<ComputeOpNode>()->axis[0], 4, &xo, &xi);

    s[Apad].compute_at(s[B], xo);
    s[Apad].pragma(s[Apad]->op.as<ComputeOpNode>()->axis[0], "memcpy");

    Map<IterVar, Range> bounds = schedule::InferBound(s);

    auto stmt = schedule::ScheduleOps(s, bounds, false);
    LOG(INFO) << stmt;

    Buffer Ab = tvm::decl_buffer(A->shape, A->dtype);
    Buffer Bb = tvm::decl_buffer(B->shape, B->dtype);

    stmt = ir::StorageFlatten(stmt, { { A, Ab }, { B, Bb } }, 64); // , config->instrument_bound_checkers);
    LOG(INFO) << stmt;

    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt;

    stmt = ir::CanonicalSimplify(stmt);
    LOG(INFO) << stmt;

    using CheckFunc = runtime::TypedPackedFunc<Stmt(Buffer, Buffer, Array<Expr>, Array<Expr>, Expr)>;

    CheckFunc checkfunc = [&](Buffer src, Buffer dst, Array<Expr> pad_before, Array<Expr> pad_after, Expr pad_value) {
        auto elem_offset = dst->elem_offset.as<IntImm>()->value;
        CHECK_EQ(elem_offset, 0);
        assert_expr_equal(src->elem_offset, max(xo * 4, 1) - 1);

        auto rpad_before = max(1 - xo * 4, 0);
        auto rpad_after = max(xo * 4 - 7, 0);
        assert_expr_equal(pad_before[0], rpad_before);
        assert_expr_equal(pad_after[0], rpad_after);
        assert_expr_equal(src->shape[0], 6 - rpad_before - rpad_after);
        return ir::Evaluate::make(0);
    };

    stmt = ir::InjectCopyIntrin(stmt, "memcpy", checkfunc);
}
