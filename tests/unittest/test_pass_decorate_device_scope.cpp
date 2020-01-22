#include "gtest/gtest.h"
// #include <dmlc/logging.h>
#include <tvm/api_registry.h>
#include <tvm/attrs.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include <tvm/schedule_pass.h>

using namespace tvm;

TEST(LANG_PASS_DECORATE, test_decorate_device)
{
    Var m("m");
    Var l("l");
    Tensor A = placeholder({ m, l });

    Tensor A1 = compute(
        { m, l }, [&](Var i, Var j) { return A[i][j]; }, "A1");
    Tensor A2 = compute(
        { m, l }, [&](Var i, Var j) { return A1[i][j] + 3; }, "A2");

    Schedule s = create_schedule({ A2->op });

    IterVar xo, xi;
    s[A2].split(A2->op.as<ComputeOpNode>()->axis[0], 8, &xo, &xi);

    s[A1].compute_at(s[A2], xo);
    s[A1].set_scope("shared");

    Map<IterVar, Range> bounds = schedule::InferBound(s);
    auto stmt = schedule::ScheduleOps(s, bounds, false);
    LOG(INFO) << stmt;
    auto stmt1 = ir::Simplify(stmt);
    LOG(INFO) << stmt1;
    auto stmt2 = ir::DecorateDeviceScope(stmt1);
    LOG(INFO) << stmt2;

    CHECK(stmt2->is_type<ir::AttrStmt>());
    CHECK_EQ(stmt2.as<ir::AttrStmt>()->attr_key, "device_scope");
    CHECK(stmt2.as<ir::AttrStmt>()->body.same_as(stmt1));
}
