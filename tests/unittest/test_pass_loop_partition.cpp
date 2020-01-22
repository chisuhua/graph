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

std::vector<bool> collect_visit(Stmt stmt, auto f)
{
    std::vector<bool> ret;
    ir::PostOrderVisit(stmt, [&](const NodeRef& n) { return ret.push_back(f(n)); });
    return std::move(ret);
}

TEST(LANG_PASS_LOOP, test_basic)
{
    Var n("n");
    Tensor A = placeholder({ n });
    Tensor B = placeholder({ n });

    Tensor T = compute({ n }, [&](Var i) { return A[i] + B[i]; });
    Schedule s = create_schedule({ T->op });
    IterVar xo, xi;
    s[T].split(T->op.as<ComputeOpNode>()->axis[0], 4, &xo, &xi);

    Map<IterVar, Range> bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);

    LOG(INFO) << stmt;

    stmt = ir::LoopPartition(stmt, false);
    LOG(INFO) << stmt;

    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt;

    std::ostringstream os;
    // os << stmt.as<body[0];
    os << stmt.as<ir::AttrStmt>()->body.as<ir::Realize>()->body.as<ir::ProducerConsumer>()->body.as<ir::Block>()->first;

    LOG(INFO) << os.str();

    CHECK(os.str().rfind("if") == std::string::npos);
}

TEST(LANG_PASS_LOOP, test_const_loop)
{
    int n = 21;
    Tensor A = placeholder({ n });
    Tensor B = placeholder({ n });

    Tensor T = compute({ n }, [&](Var i) { return A[i] + B[i]; });
    Schedule s = create_schedule({ T->op });

    IterVar xo, xi;
    s[T].split(T->op.as<ComputeOpNode>()->axis[0], 4, &xo, &xi);

    Map<IterVar, Range> bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);

    LOG(INFO) << stmt;

    stmt = ir::LoopPartition(stmt, true);
    LOG(INFO) << stmt;

    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt;

    std::ostringstream os;
    // os << stmt.as<body[0];
    os << stmt.as<ir::AttrStmt>()->body.as<ir::Realize>()->body.as<ir::ProducerConsumer>()->body.as<ir::Block>()->first;

    LOG(INFO) << os.str();

    CHECK(os.str().find("if") == std::string::npos);
}

TEST(LANG_PASS_LOOP, test_multi_loop)
{
    Var m("m");
    Var n("n");

    Var i = Var("i");
    Var j = Var("j");
    Var k = Var("k");

    Stmt stmt = ir::For::make(i, 0, 4, ir::ForType::Serial, ir::DeviceAPI::Host,
        ir::For::make(j, 0, n, ir::ForType::Serial, ir::DeviceAPI::Host,
            ir::For::make(k, 0, m, ir::ForType::Serial, ir::DeviceAPI::Host,
                //                    ir::Evaluate::make(if_then_else(likely(i*m+j+k < n),  m, n))
                ir::IfThenElse::make(likely(i * m + j + k < n), ir::Evaluate::make(m), ir::Evaluate::make(n)))));
    LOG(INFO) << stmt;
    stmt = ir::LoopPartition(stmt, true);
    LOG(INFO) << stmt;

    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt;

    auto stmt_wo_if = stmt.as<ir::For>()->body.as<ir::Block>()->first;

    std::vector<bool> ret = collect_visit(stmt_wo_if, [](const NodeRef& n) { return n->is_type<ir::IfThenElse>(); });

    bool result = any_of(ret.begin(), ret.end(), [](bool n) { return n; });
    CHECK(!result);

    /*
    ib = tvm.ir_builder.create()
    m = tvm.var('m')
    n = tvm.var('n')
    with ib.for_range(0, 4, "i") as i:
        with ib.for_range(0, n, "j") as j:
            with ib.for_range(0, m, "k") as k:
                with ib.if_scope(ib.likely(i*m+j+k < n)):
                    ib.emit(tvm.make.Evaluate(m))
                with ib.else_scope():
                    ib.emit(tvm.make.Evaluate(n))
    stmt = ib.get()
    stmt = tvm.ir_pass.LoopPartition(stmt, False)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(not any(collect_visit(stmt.body.first, lambda x: isinstance(x, tvm.stmt.IfThenElse))))
    */
}

TEST(LANG_PASS_LOOP, test_multi_if)
{
    Var m("m");
    Var n("n");

    Var i = Var("i");
    Var j = Var("j");
    Var k = Var("k");

    Stmt stmt = ir::For::make(i, 0, 4, ir::ForType::Serial, ir::DeviceAPI::Host,
        ir::For::make(j, 0, n, ir::ForType::Serial, ir::DeviceAPI::Host,
            ir::For::make(k, 0, m, ir::ForType::Serial, ir::DeviceAPI::Host,
                ir::Block::make(
                    ir::IfThenElse::make(likely(i * m + j + k < n),
                        ir::Evaluate::make(m),
                        ir::Evaluate::make(n)),
                    ir::IfThenElse::make(likely(i * m + j - k < n),
                        ir::Evaluate::make(m),
                        ir::Evaluate::make(n))))));
    // LOG(INFO)  << stmt;
    stmt = ir::LoopPartition(stmt, true);
    // LOG(INFO) << stmt;

    stmt = ir::Simplify(stmt);
    // LOG(INFO) << stmt;
    auto stmt_wo_if = stmt.as<ir::For>()->body.as<ir::Block>()->first;

    std::ostringstream os;

    os << stmt_wo_if;

    LOG(INFO) << os.str();
    CHECK(os.str().find("if") == std::string::npos);
}

TEST(LANG_PASS_LOOP, test_thread_axis)
{
    Var m("m");
    Var l("l");
    Tensor A = placeholder({ m, l });
    Tensor B = compute({ m, l }, [&](Var i, Var j) { return A[i][j] + 3; });

    Schedule s = create_schedule({ B->op });
    s[B].set_scope("shared");
    auto num_thread = 16;

    IterVar xo, xi, xi0, xi1;
    s[B].split(B->op.as<ComputeOpNode>()->axis[0], 32, &xo, &xi);
    s[B].split_by_nparts(xi, num_thread, &xi0, &xi1);
    s[B].bind(xi0, thread_axis(Range(nullptr), "threadIdx.x"));

    Map<IterVar, Range> bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);
    stmt = ir::LoopPartition(stmt, false);

    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt.as<ir::AttrStmt>()->body.as<ir::Realize>()->body.as<ir::ProducerConsumer>()->body.as<ir::Block>()->first;
    auto stmt_wo_if = stmt.as<ir::AttrStmt>()->body.as<ir::Realize>()->body.as<ir::ProducerConsumer>()->body.as<ir::Block>()->first;
    std::ostringstream os;
    os << stmt_wo_if;
    LOG(INFO) << os.str();
    CHECK(os.str().find("if") == std::string::npos);
}

TEST(LANG_PASS_LOOP, test_vectorize)
{
    Var n("n");
    Tensor A = placeholder({ n });
    Tensor B = placeholder({ n });

    Var bias("bias", Float(32));
    Var scale("scale", Float(32));

    Tensor C = compute(A->shape, [&](Var i) { return A[i] + B[i] * scale + bias; });

    Schedule s = create_schedule({ C->op });
    auto num_thread = 32;
    IterVar bx, x, tx, tmp;

    s[C].split(C->op.as<ComputeOpNode>()->axis[0], num_thread * 4, &bx, &x);
    s[C].split_by_nparts(x, num_thread, &tx, &x);
    s[C].split(x, 4, &tmp, &x);

    s[C].bind(bx, thread_axis(Range(nullptr), "blockIdx.x"));
    s[C].bind(tx, thread_axis(Range(nullptr), "threadIdx.x"));
    s[C].vectorize(x);

    std::unordered_map<Tensor, Buffer> binds;

    Stmt stmt = lower_stmt(s, { A, B }, "vectorize", binds, build_config(), true);
    LOG(INFO) << stmt;

    auto body = stmt.as<ir::AttrStmt>()->body.as<ir::Allocate>()->body.as<ir::ProducerConsumer>()->body.as<ir::AttrStmt>()->body.as<ir::AttrStmt>()->body.as<ir::IfThenElse>();
    // auto body  = stmt.as<ir::AttrStmt>()->body.as<ir::Allocate>()->body.as<ir::ProducerConsumer>()->body.as<ir::AttrStmt>()->body.as<ir::AttrStmt>()->body.as<ir::For>()->body.as<ir::IfThenElse>();

    std::ostringstream os;
    os << body->condition;
    LOG(INFO) << "for body's condition is " << os.str();
    LOG(INFO) << "IterVar x name is " << x->var->name_hint; // x->var is Variable, the name is name_hint
    CHECK(os.str().rfind(x->var->name_hint) == std::string::npos);

    std::vector<bool> ret = collect_visit(body->then_case, [](const NodeRef& n) { return n->is_type<ir::Ramp>(); });

    bool result = any_of(ret.begin(), ret.end(), [](bool n) { return n; });
    CHECK(result);
}

TEST(LANG_PASS_LOOP, test_condition)
{
    Var m("m");
    Var n("n");
    Var i("i");
    Var j("j");

    Stmt stmt = ir::For::make(i, 0, ((n + 3) / 4), ir::ForType::Serial, ir::DeviceAPI::Host,
        ir::For::make(j, 0, 4, ir::ForType::Serial, ir::DeviceAPI::Host,
            ir::Evaluate::make(ir::Select::make(likely(i * 4 + j < n), m, n))));

    // LOG(INFO)  << stmt;
    stmt = ir::LoopPartition(stmt, true);
    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt;

    auto body = stmt.as<ir::Block>();

    std::vector<bool> ret = collect_visit(body->first, [](const NodeRef& n) { return n->is_type<ir::Select>(); });

    bool result = any_of(ret.begin(), ret.end(), [](bool n) { return n; });
    CHECK(!result);
}

TEST(LANG_PASS_LOOP, test_thread_axis2)
{
    auto n = make_const(Int(32), 4096);
    Var m("m");
    Tensor A = placeholder({ n });
    Tensor B = placeholder({ n });
    Tensor C = compute(A->shape, [&](Var i) { return A[i] + B[i]; });
    Schedule s = create_schedule({ C->op });

    auto num_thread = 32;
    IterVar bx, x, tx, tmp;
    s[C].split(C->op.as<ComputeOpNode>()->axis[0], 32, &bx, &x);
    s[C].split_by_nparts(x, num_thread, &tx, &x);
    s[C].split(x, m, &tmp, &x);

    s[C].bind(bx, thread_axis(Range(nullptr), "blockIdx.x"));
    s[C].bind(tx, thread_axis(Range(nullptr), "threadIdx.x"));

    std::unordered_map<Tensor, Buffer> binds;
    Stmt stmt = lower_stmt(s, { A, B, C }, "vectorize", binds, build_config(), true); // true will run loop_partition

    LOG(INFO) << stmt;
    auto for_body = stmt.as<ir::ProducerConsumer>()->body.as<ir::AttrStmt>()->body.as<ir::AttrStmt>()->body.as<ir::Block>()->first;
    LOG(INFO) << for_body->type_key();

    std::ostringstream os;
    os << for_body.as<ir::For>()->extent;
    LOG(INFO) << os.str();
    CHECK(os.str().rfind("threadIdx") == std::string::npos);
}

TEST(LANG_PASS_LOOP, test_everything_during_deduction)
{
    Var m("m");
    Var n("n");
    Var i("i");
    Var j("j");

    Stmt stmt = ir::For::make(i, 0, n, ir::ForType::Serial, ir::DeviceAPI::Host,
        ir::For::make(j, 0, 32, ir::ForType::Serial, ir::DeviceAPI::Host,
            ir::IfThenElse::make(likely(i / j < m), ir::Evaluate::make(m))));

    // LOG(INFO)  << stmt;
    stmt = ir::LoopPartition(stmt, false);
    stmt = ir::Simplify(stmt);

    LOG(INFO) << stmt;
    auto body = stmt.as<ir::For>()->body.as<ir::For>()->body;
    CHECK(body->is_type<ir::IfThenElse>());
}

TEST(LANG_PASS_LOOP, test_single_likely)
{
    auto n = 60;
    Var i("i");
    Var j("j");

    Tensor A = placeholder({ n });
    Tensor B = placeholder({ n });

    Tensor T = compute(A->shape, [&](Var i) { return A[i] + B[i]; });
    Schedule s = create_schedule({ T->op });

    IterVar x = T->op.as<ComputeOpNode>()->axis[0];

    IterVar xo, xi;
    s[T].split(x, 16, &xo, &xi);

    Map<IterVar, Range> bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);

    stmt = ir::LoopPartition(stmt, true);
    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt;
    //
    std::vector<bool> ret = collect_visit(stmt, [](const NodeRef& n) { return n->is_type<ir::IfThenElse>(); });

    bool result = any_of(ret.begin(), ret.end(), [](bool n) { return n; });
    CHECK(!result);
}

TEST(LANG_PASS_LOOP, test_multi_likely)
{
    auto n = 94;
    auto m = 62;
    Var i("i");
    Var j("j");

    Tensor A = placeholder({ n, m });
    Tensor B = placeholder({ n, m });

    Tensor T = compute(A->shape, [&](Var i) { return A[i][j] + B[i][j]; });
    Schedule s = create_schedule({ T->op });

    Map<IterVar, Range> bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);

    IterVar x = T->op.as<ComputeOpNode>()->axis[0];
    IterVar y = T->op.as<ComputeOpNode>()->axis[1];

    IterVar xo, xi, yo, yi;
    s[T].split(x, 16, &xo, &xi);
    s[T].split(y, 16, &yo, &yi);

    bounds = schedule::InferBound(s);
    stmt = schedule::ScheduleOps(s, bounds, false);

    stmt = ir::LoopPartition(stmt, true);
    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt;
    //
    std::vector<bool> ret = collect_visit(stmt, [](const NodeRef& n) { return n->is_type<ir::IfThenElse>(); });

    bool result = any_of(ret.begin(), ret.end(), [](bool n) { return n; });
    CHECK(!result);
}

TEST(LANG_PASS_LOOP, test_oneD_pool)
{
    Var m("m");

    Buffer data = tvm::decl_buffer({ m }, HalideIR::Float(32));
    Buffer out = tvm::decl_buffer({ m }, HalideIR::Float(32));

    Var ow("ow");
    Var kw("kw");

    std::vector<Stmt> stmts;

    stmts.push_back(
        ir::For::make(ow, 0, 16, ir::ForType::Serial, ir::DeviceAPI::Host,
            ir::For::make(kw, 0, 3, ir::ForType::Serial, ir::DeviceAPI::Host,
                ir::IfThenElse::make(likely(ow > 15),
                    ir::IfThenElse::make(likely(ow < 15),
                        out.vstore({ ow }, max(out.vload({ ow }, Float(32)), data.vload({ ow + kw - 1 }, Float(32)))))))));
    stmts.push_back(
        ir::For::make(ow, 0, 16, ir::ForType::Serial, ir::DeviceAPI::Host,
            ir::For::make(kw, 0, 3, ir::ForType::Serial, ir::DeviceAPI::Host,
                ir::IfThenElse::make(likely(ow < 1),
                    ir::IfThenElse::make(likely(kw > 0),
                        out.vstore({ ow }, max(out.vload({ ow }, Float(32)), data.vload({ ow + kw - 1 }, Float(32)))))))));
    stmts.push_back(
        ir::For::make(ow, 0, 16, ir::ForType::Serial, ir::DeviceAPI::Host,
            ir::For::make(kw, 0, 3, ir::ForType::Serial, ir::DeviceAPI::Host,
                ir::IfThenElse::make(likely(ow > 14),
                    ir::IfThenElse::make(likely(kw < 2),
                        out.vstore({ ow }, max(out.vload({ ow }, Float(32)), data.vload({ ow + kw - 1 }, Float(32)))))))));
    Stmt stmt = ir::Block::make(stmts);

    stmt = ir::LoopPartition(stmt, true);
    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt;
    //
    std::vector<bool> ret = collect_visit(stmt, [](const NodeRef& n) { return n->is_type<ir::IfThenElse>(); });

    bool result = any_of(ret.begin(), ret.end(), [](bool n) { return n; });
    CHECK(!result);
}

TEST(LANG_PASS_LOOP, test_cce_loop_1)
{
    auto dtype = Float(16);
    auto m = 514;
    auto n = 514;

    Tensor _A = placeholder({ n * m }, dtype, "A");
    Buffer Ab = tvm::decl_buffer({ n * m }, dtype, "A");
    // auto A = Ab.access_ptr();

    Tensor _B = placeholder({ n * m }, dtype, "A");
    Buffer Bb = tvm::decl_buffer({ n * m }, dtype, "A");
    // auto B = Bb.access_ptr();

    Var i("i");
    Var j("j");
    // for i in 0 to n-1
    Stmt stmt = ir::For::make(i, 0, 11, ir::ForType::Serial, ir::DeviceAPI::Host,
        ir::For::make(j, 0, 160, ir::ForType::Serial, ir::DeviceAPI::Host,
            ir::IfThenElse::make(likely((i * 160 + j) < 1600),
                Ab.vstore({ (i + 1) * m + j + 1 }, Bb.vload({ i * m + j + 1 }, dtype) + Bb.vload({ (i + 1) * m + j + 1 }, dtype) + Bb.vload({ (i + 2) * m + j + 1 }, dtype)))));

    stmt = ir::LoopPartition(stmt, true);
    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt;
    //
    std::vector<bool> ret = collect_visit(stmt, [](const NodeRef& n) { return n->is_type<ir::IfThenElse>(); });

    bool result = any_of(ret.begin(), ret.end(), [](bool n) { return n; });
    CHECK(!result);
}

/*
def test_cce_loop_1():
  ib = tvm.ir_builder.create()
  dtype = 'float16'
  n = 514
  m = 514
  _A = tvm.placeholder((n*m,), name = 'A')
  Ab = tvm.decl_buffer((n*m,), dtype, name="A")
  A = ib.buffer_ptr(Ab)
  _B = tvm.placeholder((n*m,), name = 'B')
  Bb = tvm.decl_buffer((n*m,), dtype, name="B")
  B = ib.buffer_ptr(Bb)
  #for i in 0 to n-1:
  with ib.for_range(0, 11, name="i") as i:
      with ib.for_range(0, 160, name="j") as j:
          with ib.if_scope(ib.likely(((i*160) + j) < 1600)):
               A[(i+1)*m+j+1] = B[(i)*m+j+1] + B[(i+1)*m+j+1] + B[(i+2)*m+j+1]
  stmt = ib.get()
  stmt = tvm.ir_pass.LoopPartition(stmt, True)
  stmt = tvm.ir_pass.Simplify(stmt)
  assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.stmt.IfThenElse))))
  */
/* FIXME how to set let stmt correct
TEST(LANG_PASS_LOOP, test_cce_loop_2)
{
    auto len = 112;
    auto tile = 32;
    int loop = (len+tile-1) / tile;


    Var i("i");
    Var head("head");
    Var tail("tail");
    Stmt stmt =
        ir::For::make(i, 0, loop, ir::ForType::Serial, ir::DeviceAPI::Host,
            ir::Block::make(
                ir::Evaluate::make(ir::Let::make(head, i*tile, i*tile)),
                ir::IfThenElse::make(likely(head+tile > len),
                    ir::Block::make(
                        ir::LetStmt::make(tail, tail, ir::Evaluate::make(len)),
                        ir::Evaluate::make(ir::Call::make(Float(32), "cce_intrisic", {head, tail}, ir::Call::Extern))
                    ),
                    ir::Block::make(
                        ir::LetStmt::make(tail, tail, ir::Evaluate::make(head+tile)),
                        ir::Evaluate::make(ir::Call::make(Float(32), "cce_intrisic", {head, tail}, ir::Call::Extern))
                    )
                )
            )
        );

    stmt = ir::LoopPartition(stmt, true);
    stmt = ir::Simplify(stmt);
    LOG(INFO)  << stmt;
    //
    std::vector<bool> ret = collect_visit(stmt, [](const NodeRef& n) { return n->is_type<ir::IfThenElse>();});

    bool result = any_of(ret.begin(), ret.end(), [](bool n) {return n;});
    CHECK(!result);
}
 * */

/*
def test_cce_loop_2():
  ib = tvm.ir_builder.create()
  len = 112
  tile = 32
  loop = (len + tile - 1) // tile
  with ib.for_range(0, loop, 'i') as i:
    head = i * tile
    with ib.if_scope(ib.likely(head + tile > len)):
      tail = len
      ib.emit(tvm.call_extern('float32', "cce_intrisic", head, tail))
    with ib.else_scope():
      tail = head + tile
      ib.emit(tvm.call_extern('float32', "cce_intrisic", head, tail))

  stmt = ib.get()
  stmt = tvm.ir_pass.LoopPartition(stmt, True)
  stmt = tvm.ir_pass.Simplify(stmt)
  assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.stmt.IfThenElse))))


def test_cce_loop_3():
    ib = tvm.ir_builder.create()
    loop1 = 4
    loop2 = 9998
    tile = 39991
    with ib.for_range(0,loop2,'i') as i:
        with ib.for_range(0,loop1,'j') as j:
            head1 = i
            head2 = j
            with ib.if_scope(ib.likely(head1*loop1 + head2 < tile)):
                ib.emit(tvm.call_extern('float16',"cce_intrisic",head1))

    stmt = ib.get()
    stmt = tvm.ir_pass.LoopPartition(stmt,True)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.stmt.IfThenElse))))
*/
TEST(LANG_PASS_LOOP, test_conv_tiling)
{
    auto HSTR = 1;
    auto WSTR = 1;
    auto in_channel = 128;
    auto kernel_height = 3;
    auto kernel_width = 3;
    auto out_channel = 64;
    auto batch_size = 1;
    auto in_height = 64;
    auto in_width = 64;
    auto out_height = in_height - kernel_height + 1;
    auto out_width = out_height;

    Tensor data = placeholder({ batch_size, in_channel, in_height, in_width });
    Tensor kernel = placeholder({ kernel_height, kernel_width, in_channel, out_channel });

    IterVar ic = reduce_axis(Range(0, in_channel), "ic");
    IterVar kh = reduce_axis(Range(0, kernel_height), "kh");
    IterVar kw = reduce_axis(Range(0, kernel_width), "kw");

    Tensor conv = compute({ batch_size, out_channel, out_height, out_width },
        [&](Var n, Var oc, Var oh, Var ow) {
            return sum(data[n][ic][oh * HSTR + kh][ow * WSTR + kw] * kernel[kh][kw][ic][oc], { ic, kh, kw });
        });
    Schedule s = create_schedule({ conv->op });

    IterVar n = conv->op.as<ComputeOpNode>()->axis[0];
    IterVar oc = conv->op.as<ComputeOpNode>()->axis[1];
    IterVar oh = conv->op.as<ComputeOpNode>()->axis[2];
    IterVar ow = conv->op.as<ComputeOpNode>()->axis[3];

    IterVar oho, owo, ohi, owi;
    s[conv].tile(oh, ow, 16, 16, &oho, &owo, &ohi, &owi);

    Map<IterVar, Range> bounds = schedule::InferBound(s);
    Stmt stmt = schedule::ScheduleOps(s, bounds, false);
    LOG(INFO) << stmt;

    stmt = ir::LoopPartition(stmt, true);
    stmt = ir::Simplify(stmt);
    LOG(INFO) << stmt;

    std::vector<bool> ret = collect_visit(stmt, [](const NodeRef& n) { return n->is_type<ir::IfThenElse>(); });

    bool result = any_of(ret.begin(), ret.end(), [](bool n) { return n; });
    CHECK(!result);
}
/*
def test_conv_tiling():
    HSTR = WSTR = 1
    in_channel = 128
    kernel_height = kernel_width = 3
    out_channel = 64
    batch_size = 1
    in_height = in_width = 64
    out_height = out_width = in_height - kernel_height + 1
    data = tvm.placeholder((batch_size, in_channel, in_height, in_width), name='data')
    kernel = tvm.placeholder((kernel_height, kernel_width, in_channel,
        out_channel), name='kernel')
    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_height), name='kh')
    kw = tvm.reduce_axis((0, kernel_width), name='kw')
    conv = tvm.compute((batch_size, out_channel, out_height, out_width),
                       lambda n, oc, oh, ow: tvm.sum(data[n, ic, oh*HSTR + kh, ow*WSTR + kw] *
                                                     kernel[kh, kw, ic, oc],
                                                     axis=[ic, kh, kw]),
                       name="conv2d")
    s = tvm.create_schedule(conv.op)

    n, oc, oh, ow = conv.op.axis
    oho, owo, ohi, owi = s[conv].tile(oh, ow, 16, 16)
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt = tvm.ir_pass.LoopPartition(stmt, True)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.stmt.IfThenElse))))

*/
