#include "gtest/gtest.h"
// #include "ir/IROperator.h"
#include <tvm/arithmetic.h>
#include <tvm/build_module.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/schedule_pass.h>
#include <string>

template <typename T>
class TD;

using namespace tvm;

TEST(LANG_PASS_STORAGE, test_storage_sync)
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

    s[A2].bind(xo, thread_axis(Range(nullptr), "blockIdx.x"));
    s[A1].compute_at(s[A2], xo);
    s[A1].set_scope("shared");

    Map<IterVar, Range> bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);
    Buffer Ab = decl_buffer(A->shape, A->dtype, "A");
    Buffer A2b = decl_buffer(A2->shape, A2->dtype, "A2");
    stmt = ir::StorageFlatten(stmt, { { A, Ab }, { A2, A2b } }, 64);
    LOG(INFO) << stmt;

    LoweredFunc f = ir::MakeAPI(stmt, "test", { Ab, A2b }, 0, true);

    auto flist = ir::SplitHostDevice(f);
    f = flist[1];
    LOG(INFO) << f->body;
    f = ir::ThreadSync(f, "shared");
    LOG(INFO) << f->body;

    const ir::Block* body = f->body.as<ir::AttrStmt>()->body.as<ir::AttrStmt>()->body.as<ir::Allocate>()->body.as<ir::Block>();
    LOG(INFO) << body->rest.as<ir::ProducerConsumer>()->body.as<ir::Block>();
    //     body_list = tvm.make.stmt_list(f.body.body.body.body)
    //    assert(body_list[1].value.name == "tvm_storage_sync")
}
/*
def test_coproc_sync():
    @tvm.register_func("tvm.info.mem.global.cache")
    def meminfo_cache():
        return tvm.make.node(
            "MemoryInfo",
            unit_bits=8,
            max_simd_bits=32,
            max_num_bits=128,
            head_address=tvm.call_extern("handle", "global_cache"))
    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    cp = tvm.thread_axis((0, 1), "cop")
    A = ib.allocate("float32", 128, name="A", scope="global.cache")
    with ib.for_range(0, n, name="i") as i:
        A[i] = A[i] + 1
        with ib.for_range(0, 8, name="k") as k:
            with ib.for_range(0, 10, name="j") as j:
                ib.scope_attr(cp, "coproc_scope", 1)
                A[j] = A[j + k * 10] + 2
    stmt = ib.get()
    stmt = tvm.ir_pass.CoProcSync(stmt)
    body = stmt.body.body.body
    blist = tvm.make.stmt_list(body)
    assert(blist[1].value.name == "cop.coproc_read_barrier")
    assert(blist[1].value.args[3].value == 80)
    assert(blist[-2].value.name == "cop.coproc_sync")
    assert(blist[-1].value.name == "cop.coproc_write_barrier")
    assert(blist[-1].value.args[3].value == 10)


def test_coproc_sync2():
    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    cp = tvm.thread_axis((0, 1), "cop")
    ty = tvm.thread_axis("cthread")
    A = ib.allocate("float32", 128, name="A")
    ib.scope_attr(ty, "virtual_thread", 2)
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 2)
        A[ty] = 0.0
    with ib.for_range(0, n, name="i") as i:
        with ib.new_scope():
            ib.scope_attr(cp, "coproc_scope", 1)
            A[ty] = 1.0
        with ib.new_scope():
            ib.scope_attr(cp, "coproc_scope", 2)
            A[ty] = 1.0
    stmt = ib.get()
    stmt = tvm.ir_pass.CoProcSync(stmt)

def test_coproc_sync3():
    def __check_list(tvm_array, py_list):
        for ti, li in zip(tvm_array, py_list):
            if ti.value != li:
                return False
        return True

    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    cp = tvm.thread_axis((0, 1), "cop")
    A = ib.allocate("float32", 128, name="A", scope="global.cache")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, n, name="i") as j:
            with ib.new_scope():
                ib.scope_attr(cp, "coproc_scope", 1)
                A[i] = 1.0
            with ib.new_scope():
                ib.scope_attr(cp, "coproc_scope", 2)
                A[i] = 1.0
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 3)
        A[0] = 0.0

    stmt = ib.get()
    stmt = tvm.ir_pass.CoProcSync(stmt)
    slist = tvm.make.stmt_list(stmt.first.body.body)
    push_st = slist[2]
    slist = tvm.make.stmt_list(slist[-1])
    pop_st = slist[0].body.first

    assert(push_st.value.name == "cop.coproc_dep_push")
    assert(__check_list(push_st.value.args, [2,3]))
    assert(pop_st.value.name == "cop.coproc_dep_pop")
    assert(__check_list(pop_st.value.args, [2,3]))

*/
