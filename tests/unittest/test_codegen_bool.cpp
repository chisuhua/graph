#include "gtest/gtest.h"
#include <tvm/build_module.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>

template <typename T>
class TD;

using namespace tvm;

bool LLVMEnabled()
{
    const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.build_llvm");
    return pf != nullptr;
}

TEST(LANG_CODEGEN, test_cmp_load_store)
{
    int n = 32;
    Tensor A = placeholder({ n }, Int(32), "A");
    Tensor B = placeholder({ n }, Int(32), "B");
    Tensor C = compute(
        A->shape, [&](Var i) { return A[i] > B[i]; }, "C");
    Tensor D = compute(
        C->shape, [&](Var i) { return C[i] & A[i] > 1; }, "D");
    Target tgt = target::llvm();


    auto check_llvm = [&]() {
        if (!LLVMEnabled()) {
            return;
        }
        Schedule s = create_schedule({ D->op });
        IterVar xo, xi, xo1, xo2;
        s[C].split(C->op.as<ComputeOpNode>()->axis[0], 4, &xo, &xi);
        s[C].split(xo, 13, &xo1, &xo2);
        s[C].parallel(xo2);

        std::unordered_map<Tensor, Buffer> binds;
        Array<LoweredFunc> f = lower(s, { A, B, D }, "default", binds, BuildConfig::Create()); // build_config());
        runtime::Module mod = build(f, tgt, target::llvm(), BuildConfig::Create()); // build_config());

        auto assembly = mod->GetSource("asm");
        LOG(INFO) << assembly;
        // just print
    };

    LOG(INFO) << "check_llvm start:\n";
    check_llvm();
    LOG(INFO) << "check_llvm done:\n";
}
