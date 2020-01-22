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
    // Target tgt = target::llvm({" -mcpu=skylate-avx512 -mattr=-avx512f"});
    Target tgt = target::llvm();

    auto check_correct_assembly = [&](Type type, int elements, auto counts) {
        // Expr n = make_const(HalideIR::type_of<int>(), elements);
        Expr n = make_const(Int(32), elements);
        Tensor A = placeholder({ n }, type, "A");
        Tensor B = compute(
            A->shape, [&](Var i) { return popcount(A[i]); }, "B");
        Schedule s = create_schedule({ B->op });
        s[B].vectorize(s[B]->op.as<ComputeOpNode>()->axis[0]);

        std::unordered_map<Tensor, Buffer> binds;
        Array<LoweredFunc> f = lower(s, { A, B }, "default", binds, BuildConfig::Create()); // build_config());
        runtime::Module mod = build(f, tgt, Target(), BuildConfig::Create()); // build_config());
        auto assembly = mod->GetSource("asm");
        LOG(INFO) << assembly;
        // just print
    };

    check_correct_assembly(UInt(16), 8, 1);
}

TEST(LANG_CODEGEN, test_fp16_to_fp32)
{
}
/*
def test_fp16_to_fp32():
    if tvm.codegen.llvm_version_major() < 6:
        print("Skipping due to LLVM version being {} < 6".format(
            tvm.codegen.llvm_version_major()))
        return

    def fp16_to_fp32(target, width, match=None, not_match=None):
        elements = 64
        n = tvm.convert(elements)
        A = tvm.placeholder((n, width), dtype="float16", name='A')
        B = tvm.compute(A.shape, lambda *i: A(*i).astype("float32"), name='B')
        s = tvm.create_schedule(B.op)
        s[B].vectorize(s[B].op.axis[1])
        f = tvm.build(s, [A, B], target)

        assembly = f.get_source('asm').splitlines()
        if match:
            matches = [l for l in assembly if re.search(match, l)]
            assert matches
        if not_match:
            not_matches = [l for l in assembly if re.search(not_match, l)]
            assert not not_matches


    fp16_to_fp32(
        'llvm -mcpu=skylake-avx512', 15,
        match="vcvtph2ps.*ymm", not_match="vcvtph2ps.*zmm")
    fp16_to_fp32(
        'llvm -mcpu=skylake-avx512', 16,
        match="vcvtph2ps.*zmm")
    fp16_to_fp32(
        'llvm -mcpu=skylake-avx512', 17,
        match="vcvtph2ps.*zmm")
    fp16_to_fp32(
        'llvm -mcpu=skylake-avx512', 49,
        match="vcvtph2ps.*zmm")
    fp16_to_fp32(
        'llvm -mcpu=skylake-avx512 -mattr=-avx512f', 49,
        match="vcvtph2ps.*ymm",
        not_match="vcvtph2ps.*zmm")
    fp16_to_fp32(
        'llvm -mcpu=skylake-avx512 -mattr=-f16c,-avx512f', 49,
        not_match="vcvtph2ps")
    fp16_to_fp32(
        'llvm -mcpu=core-avx2', 8,
        match="vcvtph2ps.*ymm")
    fp16_to_fp32(
        'llvm -mcpu=core-avx2', 9,
        match="vcvtph2ps.*ymm")
    fp16_to_fp32(
        'llvm', 9,
        not_match="vcvtph2ps")

*/
