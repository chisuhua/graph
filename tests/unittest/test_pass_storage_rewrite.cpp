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

TEST(LANG_PASS_STORAGE, test_storage_share)
{
    Var m("m");
    Var l("l");
    Tensor A = placeholder({ m, l });
    auto num_stage = 5;
    auto B = A;
    for (auto t = 0; t < num_stage; t++) {
        // stringstream name = "A";
        // name << t;
        // B = compute({m, l}, [&](Var i, Var j) { return B[i][j]+(t+1);}, name.str());
        B = compute(
            { m, l }, [&](Var i, Var j) { return B[i][j] + (t + 1); }, "A_" + std::to_string(t));
    }

    Schedule s = create_schedule({ B->op });
    Map<IterVar, Range> bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);
    LOG(INFO) << "\n"
              << stmt;

    Buffer Ab = tvm::decl_buffer(A->shape, A->dtype, "A");
    Buffer Bb = tvm::decl_buffer(B->shape, B->dtype, "B");

    stmt = ir::StorageFlatten(stmt, { { A, Ab }, { B, Bb } }, 64);

    stmt = ir::CanonicalSimplify(stmt);
    stmt = ir::Simplify(stmt);
    LOG(INFO) << "\n"
              << stmt;

    stmt = ir::StorageRewrite(stmt);
    LOG(INFO) << "\n"
              << stmt;

    auto num_alloc = 0;
    auto verify = [&](const NodeRef& n) {
        if (n->is_type<ir::Allocate>()) {
            num_alloc += 1;
        }
    };
    ir::PostOrderVisit(stmt, verify);

    CHECK_EQ(num_alloc, 1);
}
/*
def register_mem(scope_tb, max_bits):
    #Register mem
    @tvm.register_func("tvm.info.mem.%s" % scope_tb)
    def mem_info_inp_buffer():
        return tvm.make.node("MemoryInfo",
                        unit_bits= 16,
                        max_simd_bits=32,
                        max_num_bits=max_bits,
                        head_address=None)

def test_alloc_seq():
    scope_tb = "local.L0A"
    max_bits = 1024 * 1024 * 1024

    register_mem(scope_tb, max_bits)

    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", 200, name="A", scope=scope_tb)
            A[j] = 1.2
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", 200, name="B", scope=scope_tb)
            A[j] = 1.3

    body = ib.get()
    body = tvm.ir_pass.StorageRewrite(body)
    num_alloc = [0]
    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            num_alloc[0] += 1
            assert n.extents[0].value == 200
    tvm.ir_pass.PostOrderVisit(body, verify)
    assert num_alloc[0] == 1

def test_alloc_different_dtypes():
    def stmt_generater(dtype_list, length):
        ib = tvm.ir_builder.create()
        base_dtype = dtype_list[0]
        global_a = tvm.placeholder((length,), name = "global_a", dtype = base_dtype)
        assert len(dtype_list) == 4
        with ib.for_range(0, length, name="j") as j:
            dtype = dtype_list[0]
            A = ib.allocate(dtype, length, name="A", scope="local.L0A")
            A[j] = tvm.const(1, dtype = dtype)
        with ib.for_range(0, length, name="j") as j:
            dtype = dtype_list[1]
            B = ib.allocate(dtype, length, name="B", scope="local.L0A")
            B[j] = tvm.const(1, dtype = dtype)
        with ib.for_range(0, length, name="j") as j:
            dtype = dtype_list[2]
            C = ib.allocate(dtype, length, name="C", scope="local.L0A")
            C[j] = tvm.const(1, dtype = dtype)
        with ib.for_range(0, length, name="j") as j:
            dtype = dtype_list[3]
            D = ib.allocate(dtype, length, name="D", scope="local.L0A")
            D[j] = tvm.const(1, dtype = dtype)
        with ib.for_range(0, length, name="j") as j:
            dtype = "int8"
            E = ib.allocate(dtype, length, name="E", scope="local.L0A")
            E[j] = A[j].astype(dtype) + B[j].astype(dtype) + C[j].astype(dtype) + D[j].astype(dtype)
        return ib.get()

    def dtype_bit_len(dtype):
        index = 0
        for i in dtype:
            if i.isdigit():
                break
            index += 1
        return int(dtype[index:])

    def offset_generater(dtype_list, length):
        dtype_len_list = [dtype_bit_len(i) for i in dtype_list]
        base_len = dtype_len_list[0]
        return sum([i * length / base_len for i in dtype_len_list])

    def dtype_test(dtype_list, length):
        def verify(n):
            if isinstance(n, tvm.stmt.Allocate):
                assert n.extents[0].value == offset

        body = stmt_generater(dtype_list, length)
        offset = offset_generater(dtype_list, length)
        body = tvm.ir_pass.StorageRewrite(body)
        tvm.ir_pass.PostOrderVisit(body, verify)

    length = 1024
    dtype_list = ["float16", "int32", "uint16", "int8"]
    dtype_test(dtype_list, length)

    dtype_list = ["float32", "int32", "uint16", "int8"]
    dtype_test(dtype_list, length)

    dtype_list = ["float64", "int32", "uint16", "int8"]
    dtype_test(dtype_list, length)

    dtype_list = ["int8", "int32", "uint16", "uint8"]
    dtype_test(dtype_list, length)
*/

TEST(LANG_PASS_STORAGE, test_inplace_rule)
{
    auto m = 10;
    Tensor A = placeholder({ m });
    Tensor A0 = compute(
        { m }, [&](Var i) { return A[i]; }, "A0");
    Tensor A1 = compute(
        { m }, [&](Var i) { return A[i] + 1; }, "A1");
    Tensor AA = compute(
        { m }, [&](Var i) { return A0[i] + A1[i] + A1[0]; }, "AA");

    Tensor B = compute(
        { m }, [&](Var i) { return AA[i] + 1; }, "B");

    Schedule s = create_schedule({ B->op });
    Map<IterVar, Range> bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);
    LOG(INFO) << "\n"
              << stmt;

    Buffer Ab = tvm::decl_buffer(A->shape, A->dtype, "A");
    Buffer Bb = tvm::decl_buffer(B->shape, B->dtype, "B");

    stmt = ir::StorageFlatten(stmt, { { A, Ab }, { B, Bb } }, 64);

    stmt = ir::CanonicalSimplify(stmt);
    stmt = ir::Simplify(stmt);
    LOG(INFO) << "\n"
              << stmt;

    stmt = ir::StorageRewrite(stmt);
    LOG(INFO) << "\n"
              << stmt;

    auto num_alloc = 0;
    auto verify = [&](const NodeRef& n) {
        if (n->is_type<ir::Allocate>()) {
            num_alloc += 1;
        }
    };
    ir::PostOrderVisit(stmt, verify);

    CHECK_EQ(num_alloc, 2);
}

TEST(LANG_PASS_STORAGE, test_storage_combine)
{
    auto n = 8;
    Tensor A = placeholder({ 4 });
    auto num_stage = 5;
    auto B = A;
    std::vector<Tensor> stages;
    for (auto t = 0; t < num_stage; t++) {
        B = compute(
            { n }, [&](Var i, Var j) { return B[i] + B[0] + (t + 1); }, "A_" + std::to_string(t));
        stages.push_back(B);
    }

    Schedule s = create_schedule({ B->op });

    for (auto S : stages) {
        s[S].set_scope("global:tag");
    }
    Map<IterVar, Range> bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);
    LOG(INFO) << "\n"
              << stmt;

    Buffer Ab = tvm::decl_buffer(A->shape, A->dtype, "A");
    Buffer Bb = tvm::decl_buffer(B->shape, B->dtype, "B");

    stmt = ir::StorageFlatten(stmt, { { A, Ab }, { B, Bb } }, 64);
    stmt = ir::CanonicalSimplify(stmt);
    stmt = ir::Simplify(stmt);
    LOG(INFO) << "\n"
              << stmt;

    stmt = ir::StorageRewrite(stmt);
    LOG(INFO) << "\n"
              << stmt;

    auto num_alloc = 0;
    auto verify = [&](const NodeRef& n) {
        if (n->is_type<ir::Allocate>()) {
            num_alloc += 1;
            CHECK_EQ(n.as<ir::Allocate>()->extents[0].as<ir::IntImm>()->value, 16);
        }
    };
    ir::PostOrderVisit(stmt, verify);

    CHECK_EQ(num_alloc, 1);
}

TEST(LANG_PASS_STORAGE, test_storage_share_gpu)
{
    Var m("m");
    std::vector<Tensor> A { placeholder({ m }) };
    auto num_stage = 5;
    for (auto t = 0; t < num_stage; t++) {
        A.push_back(compute(
            { m }, [&](Var i) { return A[A.size() - 1][i] + (t + 1); }, "A" + std::to_string(t) + "_s"));
        A.push_back(compute(
            { m }, [&](Var i) { return A[A.size() - 1][i]; }, "A" + std::to_string(t)));
    }

    Schedule s = create_schedule({ A[A.size() - 1]->op });

    for (auto t = 0; t < num_stage; t++) {
        auto x = A[2 * t + 2]->op.as<ComputeOpNode>()->axis[0];
        IterVar bx, tx;
        s[A[2 * t + 2]].split(x, 32, &bx, &tx);
        s[A[2 * t + 2]].bind(bx, thread_axis(Range(nullptr), "blockIdx.x"));
        s[A[2 * t + 2]].bind(tx, thread_axis(Range(nullptr), "threadIdx.x"));
        s[A[2 * t + 1]].compute_at(s[A[2 * t + 2]], tx);
        s[A[2 * t + 1]].set_scope("shared");
    }

    Map<IterVar, Range> bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);
    LOG(INFO) << "\n"
              << stmt;

    Buffer Ab = tvm::decl_buffer(A[0]->shape, A[0]->dtype, "A");
    Buffer Bb = tvm::decl_buffer(A[0]->shape, A[0]->dtype, "B");

    stmt = ir::StorageFlatten(stmt, { { A[0], Ab }, { A[A.size() - 1], Bb } }, 64);
    stmt = ir::CanonicalSimplify(stmt);
    stmt = ir::Simplify(stmt);
    LOG(INFO) << "\n"
              << stmt;

    stmt = ir::StorageRewrite(stmt);
    LOG(INFO) << "\n"
              << stmt;

    std::map<std::string, int> alloc_stats { { "global", 0 }, { "shared", 0 } };
    auto verify = [&](const NodeRef& n) {
        if (n->is_type<ir::AttrStmt>()) {
            if (n.as<ir::AttrStmt>()->value->is_type<ir::StringImm>()) {
                alloc_stats[n.as<ir::AttrStmt>()->value.as<ir::StringImm>()->value] += 1;
                // LOG(INFO) << n.as<ir::AttrStmt>()->value.as<ir::StringImm>()->value;
            };
        }
    };
    ir::PostOrderVisit(stmt, verify);

    CHECK_EQ(alloc_stats["global"], 2);
    CHECK_EQ(alloc_stats["shared"], num_stage);
}
/*
def test_parallel_alloc():
    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    with ib.for_range(0, n, name="i", for_type="parallel") as i:
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", n, name="A", scope="global")
            A[j] = A[j] + 2

    body = ib.get()
    body = tvm.ir_pass.StorageRewrite(body)
    assert (isinstance(body.body.body, tvm.stmt.Allocate))

    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    with ib.for_range(0, n, name="t") as i:
        ib.scope_attr(
            tvm.const(1, "int32") , "pragma_scope",
            tvm.make.StringImm("parallel_launch_point"))
        with ib.for_range(0, n, name="i", for_type="parallel") as i:
            with ib.for_range(0, 10, name="j") as j:
                A = ib.allocate("float32", n, name="A", scope="global")
                A[j] = A[j] + 2
    body = ib.get()
    body = tvm.ir_pass.StorageRewrite(body)

    assert(isinstance(body.body.body.body.body, tvm.stmt.Allocate))

def test_inplace_rule2(scope_tb = "local_TB2", max_bits = 1024 * 1024 * 1024):
    #Test Buffer
    register_mem(scope_tb, max_bits)
    m = 10
    A = tvm.placeholder((m,), name='A')
    C = tvm.placeholder((m,), name='C')
    D = tvm.placeholder((m,), name='D')
    A0 = tvm.compute((m,), lambda i: A[i] + C[i], name='A0')
    A1 = tvm.compute((m,), lambda i: D[i] * D[i], name='A1')
    A2 = tvm.compute((m,), lambda i: A0[i] + A1[i], name='A2')
    B = tvm.compute((m,), lambda i: A2[i], name='B')
    s = tvm.create_schedule(B.op)
    A0L = s.cache_read(A0, scope_tb, [A2])
    A1L = s.cache_read(A1, scope_tb, [A2])
    A2L = s.cache_read(A2, scope_tb, [B])
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    Cc = tvm.decl_buffer(C.shape, B.dtype, name='C')
    Dd = tvm.decl_buffer(D.shape, B.dtype, name='D')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B: Bb, C: Cc, D:Dd}, 64)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.StorageRewrite(stmt)
    # verify only have one allocations.
    # verify inplace folding works
    num_alloc = [0]
    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            num_alloc[0] += 1
    tvm.ir_pass.PostOrderVisit(stmt, verify)
    assert num_alloc[0] == 2

def test_exceed_mem():
    max_bits = 639
    # The critical max_num_bits is between 639 and 640
    loc = -1
    try:
        test_inplace_rule2("local_TEM", max_bits)
    except Exception as e:
        estr = str(e)
        loc = estr.find('Allocation exceed bound of memory')
        assert loc != -1

def test_inplace_rule3():
    #Test Buffer
    scope_tb = "local_TB3"
    max_bits=1024 * 1024 * 1024

    register_mem(scope_tb, max_bits)
    m = 10
    B0 = tvm.placeholder((m,), name='B0')
    B1 = tvm.placeholder((m,), name='B1')
    B2 = tvm.placeholder((m,), name='B2')
    B3 = tvm.placeholder((m,), name='B3')
    B4 = tvm.placeholder((m,), name='B4')
    B5 = tvm.placeholder((m,), name='B5')

    B6 = tvm.compute((m,), lambda i: B1[i] * B5[i], name='B6')
    B7 = tvm.compute((m,), lambda i: B2[i] * B4[i], name='B7')
    B8 = tvm.compute((m,), lambda i: B6[i] - B7[i], name='B8')

    B9 = tvm.compute((m,), lambda i: B2[i] * B3[i], name='B9')
    B10 = tvm.compute((m,), lambda i: B0[i] * B5[i], name='B10')
    B11 = tvm.compute((m,), lambda i: B9[i] - B10[i], name='B11')

    B12 = tvm.compute((m,), lambda i: B0[i] * B4[i], name='B12')
    B13 = tvm.compute((m,), lambda i: B1[i] * B3[i], name='B13')
    B14 = tvm.compute((m,), lambda i: B12[i] - B13[i], name='B14')

    B = tvm.compute((m,), lambda i: B8[i] * B11[i] + B14[i], name='B')
    s = tvm.create_schedule(B.op)

    B1L = s.cache_read(B1, scope_tb, [B6, B13])
    B5L = s.cache_read(B5, scope_tb, [B6, B10])
    B2L = s.cache_read(B2, scope_tb, [B7, B9])
    B4L = s.cache_read(B4, scope_tb, [B7, B12])
    B3L = s.cache_read(B3, scope_tb, [B9, B13])
    B0L = s.cache_read(B0, scope_tb, [B10, B12])

    B8L = s.cache_write(B8, scope_tb)
    B11L = s.cache_write(B11, scope_tb)
    B14L = s.cache_write(B14, scope_tb)
    B6L = s.cache_write(B6, scope_tb)
    B7L = s.cache_write(B7, scope_tb)
    B9L = s.cache_write(B9, scope_tb)
    B10L = s.cache_write(B10, scope_tb)
    B12L = s.cache_write(B12, scope_tb)
    B13L = s.cache_write(B13, scope_tb)

    s[B12].compute_inline()
    s[B13].compute_inline()
    s[B8].compute_inline()
    s[B11].compute_inline()
    s[B14].compute_inline()
    s[B6].compute_inline()
    s[B7].compute_inline()
    s[B9].compute_inline()
    s[B10].compute_inline()

    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

    B0a = tvm.decl_buffer(B0.shape, B0.dtype, name='B0')
    B1a = tvm.decl_buffer(B1.shape, B1.dtype, name='B1')
    B2a = tvm.decl_buffer(B2.shape, B2.dtype, name='B2')
    B3a = tvm.decl_buffer(B3.shape, B3.dtype, name='B3')
    B4a = tvm.decl_buffer(B4.shape, B4.dtype, name='B4')
    B5a = tvm.decl_buffer(B5.shape, B5.dtype, name='B5')

    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {B0: B0a, B1: B1a, B2: B2a, B3: B2a, B4: B4a, B5: B5a, B: Bb}, 64)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.StorageRewrite(stmt)
    # verify only have one allocations.
    # verify inplace folding works
    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            assert n.extents[0].value == 70
    tvm.ir_pass.PostOrderVisit(stmt, verify)

def test_alloc_seq_type():
    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", 200, name="A", scope="local.L0A")
            A1 = ib.allocate("float32", 200, name="A1", scope="local.L0A")
            A[j] = 1.2
            A1[j] = 1.3
            B = ib.allocate("int16", 200, name="B", scope="local.L0A")
            B[j] = tvm.const(1, "int16")
            C = ib.allocate("int16", 200, name="C", scope="local.L0A")
            C[j] = tvm.const(1, "int16")
            D = ib.allocate("int16", 200, name="D", scope="local.L0A")
            D[j] = B[j] + C[j]
            A2 = ib.allocate("float32", 200, name="A2", scope="local.L0A")
            A2[j] = A[j]

    body = ib.get()
    body = tvm.ir_pass.StorageRewrite(body)
    num_alloc = [0]
    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            num_alloc[0] += 1
            assert n.extents[0].value == 500
    tvm.ir_pass.PostOrderVisit(body, verify)
    assert num_alloc[0] == 1

def test_alloc_seq_type2():
    scope_tb = "local.L0A2"
    max_bits=1024 * 1024 * 1024

    register_mem(scope_tb, max_bits)

    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", 200, name="A", scope=scope_tb)
            A[j] = 1.2
        with ib.for_range(0, 20, name="j") as j:
            B = ib.allocate("int16", 400, name="B", scope=scope_tb)
            B[j] = tvm.const(1, "int16")
        with ib.for_range(0, 10, name="j") as j:
            C = ib.allocate("float32", 200, name="C", scope=scope_tb)
            C[j] = 1.2

    body = ib.get()
    body = tvm.ir_pass.StorageRewrite(body)
    num_alloc = [0]
    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            num_alloc[0] += 1
            assert n.extents[0].value == 200
    tvm.ir_pass.PostOrderVisit(body, verify)
    assert num_alloc[0] == 1


def test_reuse_small_buffer():
    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("int16", 200, name="A", scope="local.L0A")
            A[j] = tvm.const(1, "int16")
            B = ib.allocate("int16", 200, name="B", scope="local.L0A")
            B[j] = tvm.const(1, "int16")
            B1 = ib.allocate("int16", 200, name="B1", scope="local.L0A")
            B1[j] = A[j] + B[j]
            C = ib.allocate("int16", 400, name="C", scope="local.L0A")
            C[j] = tvm.const(1, "int16")
            D = ib.allocate("int16", 400, name="D", scope="local.L0A")
            D[j] = tvm.const(1, "int16")
            E = ib.allocate("int16", 400, name="E", scope="local.L0A")
            E[j] = C[j]

    body = ib.get()
    body = tvm.ir_pass.StorageRewrite(body)

    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            num_alloc[0] += 1
            assert n.extents[0].value == 800
    tvm.ir_pass.PostOrderVisit(body, verify)
    assert num_alloc[0] == 1
*/

TEST(LANG_PASS_STORAGE, test_replace_dataflow)
{
    Array<Expr> shape = { 255 };
    Tensor A = placeholder(shape);
    Tensor B = compute(
        shape, [&](Var i) { return A[i] + A[i]; }, "B");
    Tensor C = compute(
        shape, [&](Var i) { return A[i] + B[i]; }, "C");
    Tensor D = compute(
        shape, [&](Var i) { return A[i] + C[i]; }, "D");
    Tensor E = compute(
        shape, [&](Var i) { return A[i] + D[i]; }, "E");

    Schedule s = create_schedule({ E->op });
    s.cache_read(A, "local", { B->op, C->op, D->op, E->op });
    Map<IterVar, Range> bounds = schedule::InferBound(s);

    Stmt stmt = schedule::ScheduleOps(s, bounds, false);
    LOG(INFO) << "\n"
              << stmt;
}
/*
def test_large_input():
    @tvm.hybrid.script
    def compute(a, b):
        n = 16384
        c = output_tensor((n, n), 'int32')
        for i in range(n):
            for j in range(n):
                c[i, j] = a[i, j] - b[i, j]
        return c

    n = 16384
    shape = (n, n)
    a = tvm.placeholder(shape, name='a', dtype='int32')
    b = tvm.placeholder(shape, name='b', dtype='int32')
    c = tvm.compute(shape, lambda i, j: compute(a, b)[i, j])
    c = tvm.compute(shape, lambda i, j: 1 + c[i, j])
    s = tvm.create_schedule(c.op)
    stmt = tvm.lower(s, [a, b, c], simple_mode=True)
    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            assert n.extents[0].value == 268435456
    tvm.ir_pass.PostOrderVisit(stmt, verify)

TEST(LANG_PASS_STORAGE, test_large_input)
{
    auto compute = [](Tensor A, Tensor B) {
        auto n = 16384;
        Tensor 
    }
    Array<Expr> shape = {255};
    Tensor A = placeholder(shape);
    Tensor B = compute(shape, [&](Var i) { return A[i] + A[i]; }, "B");
    Tensor C = compute(shape, [&](Var i) { return A[i] + B[i]; }, "C");
    Tensor D = compute(shape, [&](Var i) { return A[i] + C[i]; }, "D");
    Tensor E = compute(shape, [&](Var i) { return A[i] + D[i]; }, "E");

    Schedule s = create_schedule({E->op});
    s.cache_read(A, "local", {B, C, D, E});
    Map<IterVar, Range> bounds = schedule::InferBound(s);
}
*/
