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

TEST(LANG_PASS_LOWER, test_lower_warp_mem)
{
    auto m = 128;
    Tensor A = placeholder({ m });
    Tensor B = compute({ m }, [&](Var i) { return A[i] + 3; });

    Schedule s = create_schedule({ B->op });
    Tensor AA = s.cache_read(A, "warp", { B->op });

    IterVar xo, xi;
    s[B].split(B->op.as<ComputeOpNode>()->axis[0], 32, &xo, &xi);

    IterVar xi0, xi1;
    s[B].split(xi, 16, &xi0, &xi1);

    auto tx = thread_axis(Range(nullptr), "threadIdx.x");
    s[B].bind(xi1, tx);

    s[B].bind(xo, thread_axis(Range(nullptr), "blockIdx.x"));

    s[AA].compute_at(s[B], xo);
    s[AA].split(s[AA]->op.as<ComputeOpNode>()->axis[0], 16, &xo, &xi);
    s[AA].bind(xi, tx);

    std::unordered_map<Tensor, Buffer> binds;
    LoweredFunc f = lower_func(s, { A, B }, "default", binds, build_config());
    Array<LoweredFunc> fh_and_d = ir::SplitHostDevice(f);
    LoweredFunc fhost = fh_and_d[0];
    LOG(INFO) << "fhost: \n"
              << fhost->body << "\n\n";

    LoweredFunc fdevice = fh_and_d[1];
    LOG(INFO) << "fdevice: \n"
              << fdevice->body << "\n\n";

    fdevice = ir::LowerWarpMemory(fdevice, 16);
    LOG(INFO) << fdevice->body;

    CHECK_EQ(fdevice->body.as<ir::AttrStmt>()->body.as<ir::AttrStmt>()->body.as<ir::Allocate>()->extents[0].as<IntImm>()->value, 2);
    CHECK_EQ(fdevice->body.as<ir::AttrStmt>()->body.as<ir::AttrStmt>()->value.as<ir::StringImm>()->value, "local");
}
