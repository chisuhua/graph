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

TEST(LANG_PASS_LOWER, test_makeapi)
{
    Var n("n");
    Tensor A = placeholder({ n });
    Tensor B = placeholder({ n });
    Tensor C = compute({ n }, [&](Var i) { return A[i] + B[i]; });

    Schedule s = create_schedule({ C->op });

    Map<IterVar, Range> bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);

    Buffer Ab = tvm::decl_buffer(A->shape, A->dtype);
    Buffer Bb = tvm::decl_buffer(B->shape, B->dtype);
    Buffer Cb = tvm::decl_buffer(C->shape, C->dtype);

    stmt = ir::StorageFlatten(stmt, { { A, Ab }, { B, Bb }, { C, Cb } }, 64);
    auto num_unpacked_args = 2;

    LoweredFunc f = ir::MakeAPI(stmt, "myadd", { n, Ab, Bb, Cb }, num_unpacked_args, true);
    // note: f->handle_data_type[Ab->data] is Call
    CHECK_EQ(f->handle_data_type[Ab->data].as<ir::Call>()->type, Ab->dtype);
    /*
    for(auto& i: f->handle_data_type) {
        LOG(INFO) << i.first ;
        LOG(INFO) << i.second;
    }
    */
    LOG(INFO) << f->args.size();
    // output_ssa = False
}
