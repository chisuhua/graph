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

TEST(LANG_PASS_STORAGE, test_flatten2)
{
    Var m("m");
    Var l("l");
    Tensor A = placeholder({ m, l }, Int(32), "A");
    Tensor A1 = compute(
        { m, l }, [&](Var i, Var j) { return A[i][j]; }, "A1");
    Tensor A2 = compute(
        { m, l }, [&](Var i, Var j) { return A1[i][j] + 3; }, "A2");

    Schedule s = create_schedule({ A2->op });
    IterVar xo, xi;

    s[A2].split(A2->op.as<ComputeOpNode>()->axis[0], 8, &xo, &xi);

    s[A1].compute_at(s[A2], xo);

    auto bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);
    LOG(INFO) << stmt;

    Buffer Ab = tvm::decl_buffer(A->shape, A->dtype, "A");
    Buffer A2b = tvm::decl_buffer(A2->shape, A2->dtype, "A2");
    stmt = ir::StorageFlatten(stmt, { { A, Ab }, { A2, A2b } }, 64);
    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt;
}

TEST(LANG_PASS_STORAGE, test_flatten_prefetch)
{
    Var i("i");
    Var j("j");
    Tensor A = placeholder({ 25, 100, 4 }, Int(32), "A");
    Buffer _A = tvm::decl_buffer(A->shape, A->dtype, "A");

    Array<HalideIR::Internal::Range> region;
    region.push_back(Range::make_by_min_extent(i, 2));
    region.push_back(Range::make_by_min_extent(j, 8));
    region.push_back(Range::make_by_min_extent(2, 4));
    Stmt stmt = ir::Prefetch::make(A->op, 0, A->dtype, region);

    LOG(INFO) << stmt;
    Map<Tensor, Buffer> bind;
    bind.Set(A, _A);

    stmt = ir::StorageFlatten(stmt, bind, 64);
    LOG(INFO) << stmt;

    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt;
    CHECK_EQ(stmt.as<ir::For>()->extent.as<IntImm>()->value, 2);
    CHECK(stmt.as<ir::For>()->body->is_type<ir::For>());
    CHECK_EQ(stmt.as<ir::For>()->body.as<ir::For>()->extent.as<IntImm>()->value, 2);
}

/*
def test_flatten_prefetch():
    A = tvm.placeholder((25, 100, 4), name = 'A')
    _A= tvm.decl_buffer(A.shape, A.dtype, name = 'A');
    i = tvm.var('i')
    j = tvm.var('j')
    region = [tvm.make.range_by_min_extent(i[0], i[1]) for i in [(i, 2), (j, 8), (0, 4)]]
    stmt = tvm.make.Prefetch(A.op, 0, A.dtype, region)
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: _A}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert stmt.extent.value == 2
    assert isinstance(stmt.body, tvm.stmt.For)
    assert stmt.body.extent.value == 2
*/
TEST(LANG_PASS_STORAGE, test_flatten_align)
{
    auto m = 8;
    auto l = 16;
    Tensor A = placeholder({ m, l });
    Tensor A1 = compute(
        { m, l }, [&](Var i, Var j) { return A[i][j]; }, "A1");
    Tensor A2 = compute(
        { m, l }, [&](Var i, Var j) { return A1[i][j] + 3; }, "A2");

    Schedule s = create_schedule({ A2->op });
    s[A1].storage_align(A1->op.as<ComputeOpNode>()->axis[0], 2, 1);

    auto bounds = schedule::InferBound(s);
    auto stmt = schedule::ScheduleOps(s, bounds, false);
    LOG(INFO) << "\n"
              << stmt;

    Buffer Ab = tvm::decl_buffer(A->shape, A->dtype, "A");
    Buffer A2b = tvm::decl_buffer(A2->shape, A2->dtype, "A2");

    stmt = ir::StorageFlatten(stmt, { { A, Ab }, { A2, A2b } }, 64);
    LOG(INFO) << "\n"
              << stmt;

    stmt = ir::Simplify(stmt);
    LOG(INFO) << "\n"
              << stmt;

    CHECK_EQ(stmt.as<ir::AttrStmt>()->body.as<ir::Allocate>()->extents[0].as<IntImm>()->value, 17 * 8);
    // CHECK_EQ(stmt.as<ir::For>()->body.as<ir::For>()->extent.as<IntImm>()->value, 2);
}

/*
def test_flatten_storage_align():
    m = 8
    l = 16
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.create_schedule(A2.op)
    s[A1].storage_align(A1.op.axis[0], 2, 1)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    A2b = tvm.decl_buffer(A2.shape, A2.dtype, name='A2')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, A2: A2b}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(stmt.body.extents[0].value == 17 * 8)
    */

/*
TEST(LANG_PASS_STORAGE, test_flatten_double_buffer)
{
    auto dtype = Int(64);
    auto n = 100;
    auto m = 4;
    IterVar tx = thread_axis("threadIdx.x");

    Tensor A = placeholder({m, l});
    Tensor A1 = compute({m, l}, [&](Var i, Var j){ return A[i][j];}, "A1");
    Tensor A2 = compute({m, l}, [&](Var i, Var j){ return A1[i][j] + 3;}, "A2");

    Schedule s = create_schedule({A2->op});
    s[A1].storage_align(A1->op.as<ComputeOpNode>()->axis[0], 2, 0);

    auto bounds = schedule::InferBound(s);
    auto stmt = schedule::ScheduleOps(s, bounds, false);
    LOG(INFO) << "\n" << stmt;

    Buffer Ab = tvm::decl_buffer(A->shape, A->dtype, "A");
    Buffer A2b = tvm::decl_buffer(A2->shape, A2->dtype, "A2");

    stmt = ir::StorageFlatten(stmt, {{A, Ab}, {A2, A2b}}, 64);
    LOG(INFO) << "\n" <<  stmt;

    stmt = ir::Simplify(stmt);
    LOG(INFO) << "\n" << stmt;

    CHECK_EQ(stmt.as<ir::AttrStmt>()->body.as<ir::Allocate>()->extents[0].as<IntImm>()->value, 17*8);
    // CHECK_EQ(stmt.as<ir::For>()->body.as<ir::For>()->extent.as<IntImm>()->value, 2);
}

def test_flatten_double_buffer():
    dtype = 'int64'
    n = 100
    m = 4
    tx = tvm.thread_axis("threadIdx.x")
    ib = tvm.ir_builder.create()
    A = ib.pointer("float32", name="A")
    C = ib.pointer("float32", name="C")
    ib.scope_attr(tx, "thread_extent", 1)
    with ib.for_range(0, n) as i:
        B = ib.allocate("float32", m, name="B", scope="shared")
        with ib.new_scope():
            ib.scope_attr(B.asnode(), "double_buffer_scope", 1)
            with ib.for_range(0, m) as j:
                B[j] = A[i * 4 + j]
        with ib.for_range(0, m) as j:
            C[j] = B[j] + 1

    stmt = ib.get()
    stmt = tvm.ir_pass.StorageFlatten(stmt, {}, 64)
    stmt = tvm.ir_pass.InjectDoubleBuffer(stmt, 2)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert isinstance(stmt.body.body, tvm.stmt.Allocate)
    assert stmt.body.body.extents[0].value == 2
    f = tvm.ir_pass.MakeAPI(stmt, "db", [A.asnode(), C.asnode()], 2, True)
    f = tvm.ir_pass.ThreadSync(f, "shared")
    count = [0]
    def count_sync(op):
        if isinstance(op, tvm.expr.Call) and op.name == "tvm_storage_sync":
            count[0] += 1
    tvm.ir_pass.PostOrderVisit(f.body, count_sync)
    assert count[0] == 4

*/
