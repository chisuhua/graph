#include "gtest/gtest.h"
// #include "ir/IROperator.h"
#include <tvm/arithmetic.h>
#include <tvm/build_module.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include <tvm/schedule_pass.h>

template <typename T>
class TD;

using namespace tvm;

Buffer BufferWithOffsetAlignment(Array<Expr> shape,
    Type dtype,
    std::string name,
    int data_alignment,
    int offset_factor)
{
    auto data = Var(name, Handle());

    Expr elem_offset;
    if (offset_factor != 0) {
        elem_offset = Var(name + "_elem_offset", shape[0].type());
    } else {
        elem_offset = Expr();
    }

    return BufferNode::make(data, dtype, shape, Array<Expr>(), elem_offset, name, "",
        data_alignment, offset_factor);
}

Stmt lower(Schedule sch,
    const Array<Tensor>& args)
{
    Map<Tensor, Buffer> out_binds;
    Array<NodeRef> out_arg_list;
    BuildConfig config = build_config();
    for (const auto& x : args) {
        if (out_binds.find(x) == out_binds.end()) {
            auto buf = BufferWithOffsetAlignment(x->shape, x->dtype, x->op->name,
                config->data_alignment, config->offset_factor);
            out_binds.Set(x, buf);
            out_arg_list.push_back(buf);
        } else {
            out_arg_list.push_back(out_binds[x]);
        }
    }

    sch = sch.normalize();
    auto bounds = schedule::InferBound(sch);
    auto stmt = schedule::ScheduleOps(sch, bounds, false);
    stmt = ir::StorageFlatten(stmt, out_binds, 64, config->instrument_bound_checkers);
    stmt = ir::CanonicalSimplify(stmt);
    stmt = ir::Simplify(stmt);

    return stmt;
}

TEST(LANG_PASS_OPT, test_basic_pipline)
{
    auto n = make_const(Int(32), 128);
    auto dtype = Int(32);
    Tensor A = placeholder({ n }, dtype, "A");
    std::vector<Tensor> stages;
    auto num_stage = 3;
    Tensor B = A;
    for (int k = 0; k < num_stage; ++k) {
        stages.push_back(B);
        auto name = "Ak_" + k;
        B = compute(
            { n }, [&](Var i) { return B[i] + k; }, name);
    }

    Schedule s = create_schedule({ B->op });
    Stmt st = lower(s, { A, B });
    LOG(INFO) << st;

    IterVar xo, xi;
    s[B].split_by_nparts(B->op.as<ComputeOpNode>()->axis[0], 1, &xo, &xi);
    s[B].bind(xo, thread_axis(Range(nullptr), "pipeline"));
    s[B].split(xi, 4, &xo, &xi);

    for (auto& S : stages) {
        s[S].compute_at(s[B], xo);
    }

    LOG(INFO) << s;
    Stmt stmt = lower(s, { A, B });
    LOG(INFO) << stmt;
}

/*
def lower(s, args):
    binds = {}
    arg_list = []

    for x in args:
        assert isinstance(x, tvm.tensor.Tensor)
        buf = tvm.decl_buffer(x.shape, dtype=x.dtype, name=x.op.name)
        binds[x] = buf
        arg_list.append(buf)
    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt = tvm.ir_pass.StorageFlatten(stmt, binds, 64)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    return stmt

def test_basic_pipeline():
    n = tvm.convert(128)
    A = tvm.placeholder((n,), name='A')
    stages = []
    num_stage = 3

    B = A
    for k in range(num_stage):
        stages.append(B)
        B = tvm.compute((n,), lambda i: B[i] + k, name="A%s" % k)

    s = tvm.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], nparts=1)
    s[B].bind(xo, tvm.thread_axis("pipeline"))
    xo, xi = s[B].split(xi, factor=4)
    for S in stages:
        s[S].compute_at(s[B], xo)

    stmt = lower(s, [A, B])
    stmt = tvm.ir_pass.SplitPipeline(stmt, False)
    print(stmt)
    stmt = tvm.ir_pass.NarrowChannelAccess(stmt)
    print(stmt)
    assert(tvm.ir_pass.VerifySSA(stmt))
    */
TEST(LANG_PASS_OPT, test_conv1d)
{
    Var n("n");
    Tensor A = compute(
        { n + 2 }, [](Var i) { return 1; }, "A");

    auto computeB = [&](Var ii) {
        auto i = ii + 1;
        return A[i - 1] + A[i] + A[i + 1];
    };

    Tensor B = compute({ n }, computeB, "B");
    Schedule s = create_schedule({ B->op });
    IterVar px, xi;
    s[B].split_by_nparts(B->op.as<ComputeOpNode>()->axis[0], 1, &px, &xi);
    s[B].bind(px, thread_axis(Range(nullptr), "pipeline"));
    s[A].compute_at(s[B], px);

    Stmt stmt = lower(s, { A, B });
    LOG(INFO) << stmt;

    stmt = ir::SplitPipeline(stmt, false);
    LOG(INFO) << stmt;

    stmt = ir::NarrowChannelAccess(stmt);
    LOG(INFO) << stmt;
}
/*
def test_conv1d():
    n = tvm.var('n')
    A = tvm.compute((n+2), lambda i: 1,  name='A')
    def computeB(ii):
        i = ii + 1
        return A[i-1] + A[i] + A[i+1]
    B = tvm.compute(n, computeB, name='B')
    s = tvm.create_schedule(B.op)
    px, xi = s[B].split(B.op.axis[0], nparts=1)
    s[B].bind(px, tvm.thread_axis("pipeline"))
    s[A].compute_at(s[B], px)
    stmt = lower(s, [B])
    stmt = tvm.ir_pass.SplitPipeline(stmt, False)
    print(stmt)
    stmt = tvm.ir_pass.NarrowChannelAccess(stmt)
    print(stmt)


*/
