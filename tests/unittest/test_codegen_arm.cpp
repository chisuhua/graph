#include "gtest/gtest.h"
#include <tvm/build_module.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>

template <typename T>
class TD;

using namespace tvm;

TEST(LANG_CODEGEN, test_popcount)
{
    Target tgt = target::llvm({ "-target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon" });

    auto check_correct_assembly = [&](Type type, int elements, auto counts) {
        Expr n = make_const(HalideIR::type_of<int>(), elements);
        Tensor A = placeholder({ n }, type, "A");
        Tensor B = compute(
            A->shape, [&](Var i) { return popcount(A[i]); }, "B");
        Schedule s = create_schedule({ B->op });
        s[B].vectorize(s[B]->op.as<ComputeOpNode>()->axis[0]);

        std::unordered_map<Tensor, Buffer> binds;
        Array<LoweredFunc> f = lower(s, { A, B }, "default", binds, build_config());
        runtime::Module mod = build(f, tgt, Target(), build_config());
        auto assembly = mod->GetSource("asm");
        LOG(INFO) << assembly;
    };

    check_correct_assembly(UInt(16), 8, 1);
}
