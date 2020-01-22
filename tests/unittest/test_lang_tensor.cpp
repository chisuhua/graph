#include "gtest/gtest.h"
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <dmlc/logging.h>
/*
 * #include "ir/IREquality.h"

using namespace HalideIR::Internal;
*/
using namespace tvm;

TEST(LANG_TENSOR, test_tensor)
{
    Var m { "m" };
    Var n { "n" };
    Var l { "l" };
    Tensor A = placeholder({ m, l }, Int(32), "A");
    Tensor B = placeholder({ n, l }, Int(32), "A");
    Tensor T = compute(
        { m, n, l },
        [&](Var i, Var j, Var k) {
            return A[i][k] * B[j][k];
        },
        "T");
    LOG(INFO) << T;
    LOG(INFO) << T->op.as<ComputeOpNode>()->body;
    Array<Expr> shape { m, n, l };

    // IRCompareCache cache(5);
    // IRComparer::CmpResult r = IRComparer(&cache).compare_expr(T->shape, shape);
    for (auto i = 0u; i < T->shape.size(); i++) {
        CHECK(T->shape[0].same_as(shape[0]));
    }

    // CHECK_EQ(A->op->type_key(), std::string("PlaceholderOp"));
    CHECK(A->op->is_type<PlaceholderOpNode>());
    CHECK_EQ(A, A);

    LOG(INFO) << "T's type_key is " << T->op->type_key();

    // CHECK_EQ(T->op->type_key(), std::string("ComputeOp"));
    CHECK(T->op->is_type<ComputeOpNode>());
    CHECK_EQ(T->op.output(0), T);
    // TODO
    // assert(T[0][0][0].astype('float16').dtype == 'float16')
}

/*
def test_tensor():
    m = tvm.var('m')
    n = tvm.var('n')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    T = tvm.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])
    print(T)
    print(T.op.body)
    assert(tuple(T.shape) == (m, n, l))
    assert(isinstance(A.op, tvm.tensor.PlaceholderOp))
    assert(A == A)
    assert(T.op.output(0) == T)
    assert(T.op.output(0).__hash__() == T.__hash__())
    d = {T.op.output(0) : 1}
    assert(d[T] == 1)
    assert(T[0][0][0].astype('float16').dtype == 'float16')
    */
TEST(LANG_TENSOR, test_rank_zero)
{
    auto m = Var("m");
    Tensor A = placeholder({ m }, Int(32), "A");
    Tensor scale = placeholder(Array<Expr>(), Int(32), "s");
    IterVar k = reduce_axis(Range { 0, m }, "k");
    auto T = compute(
        { 1 }, [&](Var i) -> Expr {
            return sum(A[k] * scale(), { k });
        },
        "T");
    LOG(INFO) << T;
    LOG(INFO) << T->op.as<ComputeOpNode>()->body;
}

/*
def test_rank_zero():
    m = tvm.var('m')
    A = tvm.placeholder((m,), name='A')
    scale = tvm.placeholder((), name='s')
    k = tvm.reduce_axis((0, m), name="k")
    T = tvm.compute((), lambda : tvm.sum(A[k] * scale(), axis=k))
    print(T)
    print(T.op.body)
    assert(tuple(T.shape) == ())
    */

TEST(LANG_TENSOR, test_conv1d)
{
    auto n = Var("n");
    Tensor A = placeholder({ n + 2 }, Int(32), "A");

    auto computeB = [&](Var ii) {
        auto i = ii + 1;
        return A[i - 1] + A[i] + A[i + 1];
    };
    auto B = compute({ n }, computeB);
}

/*
def test_conv1d():
    n = tvm.var('n')
    A = tvm.placeholder((n+2), name='A')
    def computeB(ii):
        i = ii + 1
        return A[i-1] + A[i] + A[i+1]
    B = tvm.compute(n, computeB)
*/

TEST(LANG_TENSOR, test_tensor_slice)
{
    auto n = Var("n");
    auto A = compute({ n, n }, [](Var i, Var j) { return 1; });
    auto B = compute({ n }, [&](Var i) { return A[0][i] + A[0][i]; });
}

/*
def test_tensor_slice():
    n = tvm.var('n')
    A = tvm.compute((n, n), lambda i, j: 1)
    B = tvm.compute((n,), lambda i: A[0][i] + A[0][i])
*/

TEST(LANG_TENSOR, test_tensor_reduce_multi_axis)
{
    auto m = Var("m");
    auto n = Var("n");
    Tensor A = placeholder({ m, n }, Int(32), "A");
    IterVar k1 = reduce_axis(Range { 0, n }, "k");
    IterVar k2 = reduce_axis(Range { 0, m }, "k");

    Tensor C1 = compute({ 1 }, [&](Var i) { return sum(A[k1][k2], { k1, k2 }); });
    Tensor C2 = compute({ 1 }, [&](Var i) { return sum(A[k1][k2], { k1, k2 }); });
}

/*
def test_tensor_reduce_multi_axis():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    k1 = tvm.reduce_axis((0, n), "k")
    k2 = tvm.reduce_axis((0, m), "k")
    C = tvm.compute((1,), lambda _: tvm.sum(A[k1, k2], axis=(k1, k2)))
    C = tvm.compute((1,), lambda _: tvm.sum(A[k1, k2], axis=[k1, k2]))
*/

TEST(LANG_TENSOR, test_tensor_comm_reducer)
{
    auto m = Var("m");
    auto n = Var("n");
    Tensor A = placeholder({ m, n }, Int(32), "A");
    IterVar k = reduce_axis(Range { 0, n }, "k");

    auto mysum = [&](Expr source, Array<IterVar> rdom) {
        Var x("x", source.type()), y("y", source.type());
        Expr result = ir::Add::make(x, y);
        Expr identity_element = make_zero(source.type());
        ir::CommReducer combiner = ir::CommReducerNode::make({ x }, { y }, { result }, { identity_element });
        return ir::Reduce::make(combiner, { source }, rdom, make_const(Bool(1), true), 0);
    };

    Tensor C = compute({ m }, [&](Var i) { return mysum(A[i][k], { k }); });
}

/*
def test_tensor_comm_reducer():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    k = tvm.reduce_axis((0, n), "k")
    mysum = tvm.comm_reducer(lambda x, y: x+y, lambda t: tvm.const(0, dtype=t))
    C = tvm.compute((m,), lambda i: mysum(A[i, k], axis=k))
    */
/* FIXME schi
TEST(LANG_TENSOR, test_tensor_comm_reducer_overload) {
    auto m = Var("m");
    auto n = Var("n");
    Tensor A = placeholder({m, n}, Int(32), "A");
    IterVar k = reduce_axis(Range { 0, n }, "k");

    auto mysum = [&](Expr source, Array<IterVar> rdom) {
        Var x("x", source.type()), y("y", source.type());
        Expr result = ir::Add::make(x, y);
        Expr identity_element = make_zero(source.type());
        ir::CommReducer combiner = ir::CommReducerNode::make({ x }, { y }, { result }, { identity_element });
        return ir::Reduce::make(combiner, { source }, rdom, make_const(Bool(1), true), 0);
    };

    auto sum_res = mysum(m, n);
}

def test_tensor_comm_reducer_overload():
    m = tvm.var('m')
    n = tvm.var('n')
    mysum = tvm.comm_reducer(lambda x, y: x+y, lambda t: tvm.const(0, dtype=t))
    sum_res = mysum(m, n)
    */
TEST(LANG_TENSOR, test_tensor_reduce)
{
    auto m = Var("m");
    auto n = Var("n");
    auto l = Var("l");
    Tensor A = placeholder({ m, l }, Int(32), "A");
    Tensor B = placeholder({ n, l }, Int(32), "B");
    Tensor T = compute({ m, n, l }, [&](Var i, Var j, Var k) { return A[i][k] * B[j][k]; });

    IterVar rv = reduce_axis(Range { 0, A->shape[1] }, "k");

    Tensor C = compute({ m, n }, [&](Var i, Var j) { return sum(T[i][j][rv + 1], { rv }); });
}

/*
def test_tensor_reduce():
    m = tvm.var('m')
    n = tvm.var('n')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    T = tvm.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])
    rv = tvm.reduce_axis((0, A.shape[1]), "k")
    C = tvm.compute((m, n), lambda i, j: tvm.sum(T(i, j, rv+1), axis=rv))
    # json load save
    C_json = tvm.save_json(C)
    C_loaded = tvm.load_json(C_json)
    assert(isinstance(C_loaded, tvm.tensor.Tensor))
    assert(str(C_loaded) == str(C))
    */

/* FIXME schi need add ir_builder
TEST(LANG_TENSOR, test_tensor_compute1) {
    auto m = 1024;
    auto factor = 16;
    auto dtype = Float(32);

    auto intrin_vadd = [](Var n) {
        Tensor x = placeholder({n}, Float(32));
        Tensor y = placeholder({n}, Float(32));
        Tensor z = compute(x->shape, [&](Var i){ return x[i] + y[i];});

        auto intrin_func = [](auto ins, auto outs) {
            auto ib = ir_builder.create();
        }
    };

}

def test_tensor_compute1():
    m = 1024
    factor = 16
    dtype = 'float32'

    def intrin_vadd(n):
        x = tvm.placeholder((n,))
        y = tvm.placeholder((n,))
        z = tvm.compute(x.shape, lambda i: x[i] + y[i])

        def intrin_func(ins, outs):
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_extern(outs[0].dtype, 'vadd', ins[0].access_ptr("r"), ins[1].access_ptr('r'), outs[0].access_ptr('wr')))
            return ib.get()

        with tvm.build_config(offset_factor=n):
            return tvm.decl_tensor_intrin(z.op, intrin_func)

    vadd = intrin_vadd(factor)

    A = tvm.placeholder((m//factor, factor), name="A", dtype=dtype)
    B = tvm.placeholder((m//factor, factor), name="B", dtype=dtype)
    C = tvm.compute((m//factor, factor),
          lambda i: vadd(A[i, 0:factor], B[i, 0:factor]))

    s = tvm.create_schedule(C.op)
    stmt = tvm.lower(s, [A, B, C], simple_mode=True)
    assert isinstance(stmt.body.body, tvm.stmt.Evaluate)
*/

extern int buffer_read;
extern int buffer_write;
extern int buffer_rw;
// FIXME
TEST(LANG_TENSOR, test_tensor_compute2)
{
    auto M = 1024;
    auto N = 1024;
    auto L = 1024;
    auto factor = 16;
    auto factor1 = 32;
    auto factor2 = 32;
    auto dtype = Float(32);

    auto intrin_gemm = [](Var m, Var n, Var l) {
        IterVar k = reduce_axis(Range { 0, l });
        Tensor x = placeholder({ m, l }, Float(32));
        Tensor y = placeholder({ n, l }, Float(32));
        Tensor z = compute({ m, n }, [&](Var i, Var j) { return sum(x[i][k] * y[i][k], { k }); });

        auto intrin_func = [&](Array<Buffer> ins, Array<Buffer> outs) {
            Expr x_ptr = ins[0].access_ptr(buffer_read);
            Expr y_ptr = ins[1].access_ptr(buffer_read);
            Expr z_ptr = outs[0].access_ptr(buffer_write);
            /*
            auto body = ir::Call::make(Int(32), ir::intrinsic::tvm_call_packed, {ir::StringImm::make(runtime::symbol::gemv), x_ptr, y_ptr, z_ptr, m, n, l}, ir::Call::Intrinsic);
            auto reset = ir::Call::make(Int(32), ir::intrinsic::tvm_call_packed, {ir::StringImm::make(runtime::symbol::fill_zero), z_ptr, m, n}, ir::Call::Intrinsic);
            auto update = ir::Call::make(Int(32), ir::intrinsic::tvm_call_packed, {ir::StringImm::make(runtime::symbol::gemv_add), x_ptr, y_ptr, z_ptr, m, n, l}, ir::Call::Intrinsic);
            return {body, reset, update};
            */
        };
    };
}

/*
def test_tensor_compute2():
    M = 2048
    N = 1024
    L = 1024
    factor = 16
    factor1 = 32
    factor2 = 32
    dtype = 'float32'

    def intrin_gemm(m, n, l):
        k = tvm.reduce_axis((0, l))
        x = tvm.placeholder((m, l))
        y = tvm.placeholder((n, l))
        # in theory, no relation
        z = tvm.compute((m, n), lambda i, j: tvm.sum(x[i][k] * y[j][k], axis=k))

        def intrin_func(ins, outs):
            x_ptr = ins[0].access_ptr("r")
            y_ptr = ins[1].access_ptr("r")
            z_ptr = outs[0].access_ptr("w")
            body = tvm.call_packed(
                "gemv", x_ptr, y_ptr, z_ptr, m, n, l)
            reset = tvm.call_packed(
                "fill_zero", z_ptr, m, n)
            update = tvm.call_packed(
                "gemv_add", x_ptr, y_ptr, z_ptr, m, n, l)
            return body, reset, update

        with tvm.build_config(offset_factor=n):
            return tvm.decl_tensor_intrin(z.op, intrin_func)

    vgemm = intrin_gemm(factor1, factor2, factor)

    A = tvm.placeholder((M//factor1, L//factor, factor1, factor), name="A", dtype=dtype)
    B = tvm.placeholder((N//factor2, L//factor, factor2, factor), name="B", dtype=dtype)
    k = tvm.reduce_axis((0, L//factor), name='k')
    C = tvm.compute((M//factor1, N//factor2, factor1, factor2),
          lambda i, j: vgemm(A[i, k, 0:factor1, 0:factor], B[j, k, 0:factor2, 0:factor], reduce_axis=k))

    s = tvm.create_schedule(C.op)
    stmt = tvm.lower(s, [A, B, C], simple_mode=True)
    assert isinstance(stmt.body.body.body.first, tvm.stmt.Evaluate)
    assert isinstance(stmt.body.body.body.rest.body, tvm.stmt.Evaluate)

def test_tensor_scan():
    m = tvm.var("m")
    n = tvm.var("n")
    x = tvm.placeholder((m, n))
    s = tvm.placeholder((m, n))
    res = tvm.scan(tvm.compute((1, n), lambda _, i: x[0, i]),
                   tvm.compute((m, n), lambda t, i: s[t-1, i] + x[t, i]),
                   s)
    assert tuple(res.shape) == (m, n)

def test_scan_multi_out():
    m = tvm.var("m")
    n = tvm.var("n")
    x1 = tvm.placeholder((m, n))
    s1 = tvm.placeholder((m, n))
    x2 = tvm.placeholder((m, n))
    s2 = tvm.placeholder((m, n))
    s1_init = tvm.compute((1, n), lambda _, i: x1[0, i])
    s2_init = tvm.compute((1, n), lambda _, i: x2[0, i])
    s1_update = tvm.compute((m, n), lambda t, i: s1[t-1, i] + s2[t-1, i] + x1[t, i])
    s2_update = tvm.compute((m, n), lambda t, i: x2[t, i] + s2[t-1,i])

    r0, r1 = tvm.scan([s1_init, s2_init],
                      [s1_update, s2_update],
                      [s1, s2])
    assert(r0.value_index == 0)
    assert(r1.value_index == 1)
    json_str = tvm.save_json(r0.op)
    zz = tvm.load_json(json_str)
    assert isinstance(zz, tvm.tensor.ScanOp)

def test_extern():
    m = tvm.var('m')
    A = tvm.placeholder((m,), name='A')

    def extern_func(ins, outs):
        assert(isinstance(ins[0], tvm.schedule.Buffer))
        return tvm.call_packed("myadd", ins[0].data, outs[0].data, m)
    B = tvm.extern((m,), [A], extern_func)
    assert(tuple(B.shape) == (m,))


def test_extern_multi_out():
    m = tvm.var('m')
    A = tvm.placeholder((m,), name='A')
    B = tvm.compute((m,), lambda i: A[i] * 10)

    def extern_func(ins, outs):
        assert(isinstance(ins[0], tvm.schedule.Buffer))
        return tvm.call_packed(
            "myadd", ins[0].data, outs[0].data, outs[1].data, m)
    res = tvm.extern([A.shape, A.shape], [A, B], extern_func)
    assert(len(res) == 2)
    assert(res[1].value_index == 1)

def test_tuple_inputs():
    m = tvm.var('m')
    n = tvm.var('n')
    A0 = tvm.placeholder((m, n), name='A0')
    A1 = tvm.placeholder((m, n), name='A1')
    T0, T1 = tvm.compute((m, n), lambda i, j: (A0[i, j] * 2, A1[i, j] * 3), name='T')
    s = tvm.create_schedule(T0.op)

    for i in range(len(T0.shape)):
      assert(T0.shape[i] == T1.shape[i])
    assert(T0.op == T1.op)
    assert(T0.value_index == 0)
    assert(T1.value_index == 1)

def test_tuple_with_different_deps():
    m = tvm.var('m')
    n = tvm.var('n')
    A0 = tvm.placeholder((m, n), name='A1')
    A1 = tvm.placeholder((m, n), name='A2')
    B0, B1 = tvm.compute((m, n), lambda i, j: (A0[i, j] * 2, A1[i, j] * 3), name='B')
    C = tvm.compute((m, n), lambda i, j: B0[i, j] + 4, name='C')

    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=10)
    s[B0.op].compute_at(s[C], xo)
    sch = s.normalize()
    bounds = tvm.schedule.InferBound(sch)
    stmt = tvm.schedule.ScheduleOps(sch, bounds)

    def get_B1_realize(x):
        if isinstance(x, tvm.stmt.Realize) and \
           x.func == B1.op and x.value_index == 1:
            ret.append(x)
    ret = []
    tvm.ir_pass.PostOrderVisit(stmt, get_B1_realize)

    assert stmt.node == C.op and len(ret) == 1


def test_tensor_inputs():
    x = tvm.placeholder((1,), name='x')
    y = tvm.compute(x.shape, lambda i: x[i] + x[i])
    assert tuple(y.op.input_tensors) == (x,)


def test_tensor_pool():
    def intrin_pool():
        A = tvm.placeholder((64, 16, 16), name='A')
        kh = tvm.reduce_axis((0, 3), name='kh')
        kw = tvm.reduce_axis((0, 3), name='kw')
        P = tvm.compute((64, 14, 14),
                        lambda c, oh, ow: tvm.max(A[c, oh + kh, ow + kw],
                                                  axis=[kh, kw]),
                        name='p')

        def intrin_func(ins, outs):
            dinp = ins[0]
            dout = outs[0]
            return tvm.call_packed("op", dinp, dout)

        with tvm.build_config(offset_factor=1):
            return tvm.decl_tensor_intrin(P.op, intrin_func)

    A = tvm.placeholder((1, 64, 16, 16), name='A')
    P = pool(data=A, kernel=(3, 3), stride=(1, 1), padding=(0, 0, 0, 0),
             pool_type='max')
    s = tvm.create_schedule(P.op)
    _, oh, _, _ = P.op.axis
    intrin = intrin_pool()
    s[P].tensorize(oh, intrin)
    tvm.lower(s, [A, P])
*/
