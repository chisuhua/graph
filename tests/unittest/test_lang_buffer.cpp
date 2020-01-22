#include "tvm/node/container.h"
#include "gtest/gtest.h"
#include <tvm/buffer.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_pass.h>
#include <tvm/packed_func_ext.h>

template <typename T>
class TD;

using namespace tvm;

int buffer_read = 1;
int buffer_write = 2;
int buffer_rw = buffer_read | buffer_write;

TEST(LANG_BUFFER, test_buffer)
{
    auto m = tvm::Var("m");
    auto n = tvm::Var("n");
    auto l = tvm::Var("l");
    // auto Ab = HalideIR::Internal::BufferNode::make();
    Array<Expr> shape1;
    shape1.push_back(m);
    shape1.push_back(n);

    Array<Expr> shape2;
    shape2.push_back(n);
    shape2.push_back(l);

    Buffer Ab = tvm::decl_buffer(shape1, HalideIR::Float(32));
    Buffer Bb = tvm::decl_buffer(shape2, HalideIR::Float(32));

    std::cout << Ab->type_key() << std::endl;
    // assert(Ab->type_key() == std::string("Buffer"));
    CHECK(Ab->is_type<BufferNode>());
    assert(Ab->dtype == HalideIR::Float(32));
    assert(Ab->shape == shape1);
}

TEST(LANG_BUFFER, test_buffer_access_ptr)
{
    auto m = tvm::Var("m");
    auto n = tvm::Var("n");
    Array<Expr> shape;
    shape.push_back(m);
    shape.push_back(n);

    Array<Expr> strides;
    strides.push_back(n + 1);
    strides.push_back(1);
    // Expr stride = make_const(shape[2].type(), 1);
    auto Ab = tvm::BufferNode::make(Var("Ab", Handle()),
        HalideIR::Float(32), shape, strides, Expr(), "Ab", "", 0, 0);
    auto aptr = Ab.access_ptr(buffer_rw);
    assert(aptr.as<ir::Call>()->args[0].type() == Ab->dtype);
    assert(aptr.as<ir::Call>()->args[4].as<HalideIR::Internal::IntImm>()->value == buffer_rw);
    aptr = Ab.access_ptr(buffer_write);
    assert(aptr.as<ir::Call>()->args[4].as<HalideIR::Internal::IntImm>()->value == buffer_write);
}

TEST(LANG_BUFFER, test_buffer_access_ptr_offset)
{
    auto m = tvm::Var("m");
    auto n = tvm::Var("n");
    auto Ab = tvm::decl_buffer({ m, n }, HalideIR::Float(32));
    auto aptr = Ab.access_ptr(buffer_rw, Handle(), 1, make_const(Int(32), 100));
    auto offset = tvm::ir::Simplify(aptr.as<ir::Call>()->args[2]);
    CHECK(tvm::ir::Equal(offset, 100));
    CHECK(aptr.as<ir::Call>()->args[4].as<HalideIR::Internal::IntImm>()->value == buffer_rw);

    auto v = tvm::Var("v");
    aptr = Ab.access_ptr(buffer_rw, Handle(), 1, make_const(Int(32), 100) + 100 + v);
    offset = tvm::ir::Simplify(aptr.as<ir::Call>()->args[2]);
    std::cout << offset << std::endl;
    CHECK(tvm::ir::Equal(offset, v + 200));
    CHECK(aptr.as<ir::Call>()->args[4].as<HalideIR::Internal::IntImm>()->value == buffer_rw);

    aptr = Ab.access_ptr(buffer_rw, Handle(), 1, ir::Call::make(Int(32), "test_call", { 100 + 100 + v }, ir::Call::Extern));
    offset = tvm::ir::Simplify(aptr.as<ir::Call>()->args[2]);
    assert(tvm::ir::Equal(offset, ir::Call::make(Int(32), "test_call", { v + 200 }, ir::Call::Extern)));
    assert(aptr.as<ir::Call>()->args[4].as<HalideIR::Internal::IntImm>()->value == buffer_rw);
}

TEST(LANG_BUFFER, test_buffer_access_ptr_extent)
{
    auto m = tvm::Var("m");
    auto n = tvm::Var("n");
    auto Ab = tvm::decl_buffer({ m, n }, HalideIR::Float(32));
    auto aptr = Ab.access_ptr(buffer_rw);
    assert(tvm::ir::Equal(aptr.as<ir::Call>()->args[3], m * n));

    aptr = Ab.access_ptr(buffer_rw, Handle(), 1, make_const(Int(32), 100));
    assert(tvm::ir::Equal(aptr.as<ir::Call>()->args[3], m * n - 100));

    Ab = tvm::BufferNode::make(Var("Ab", Handle()),
        HalideIR::Float(32), { m, n }, { n + 1, 1 }, Expr(), "Ab", "", 0, 0);
    aptr = Ab.access_ptr(buffer_rw, Handle(), 1, make_const(Int(32), 100));
    assert(tvm::ir::Equal(aptr.as<ir::Call>()->args[3], Ab->strides[0] * m - 100));
}

TEST(LANG_BUFFER, test_buffer_vload)
{
    auto m = tvm::Var("m");
    auto n = tvm::Var("n");

    auto Ab = tvm::BufferNode::make(Var("Ab", Handle()),
        HalideIR::Float(32), { m, n }, Array<Expr>(), make_const(Int(32), 100), "Ab", "", 0, 0);
    auto load = Ab.vload({ 2, 3 }, Ab->dtype);
    auto offset = tvm::ir::Simplify(load.as<ir::Load>()->index);
    assert(tvm::ir::Equal(offset, n * 2 + 103));
}

TEST(LANG_BUFFER, test_buffer_index_merge_mult_mod)
{
    auto m = tvm::Var("m");
    auto n = tvm::Var("n");
    auto s = tvm::Var("s");
    auto k0 = tvm::Var("k0");
    auto k1 = tvm::Var("k1");

    auto A = tvm::decl_buffer({ m, n }, HalideIR::Float(32));
    auto A_stride = tvm::BufferNode::make(Var("A_stride", Handle()),
        HalideIR::Float(32), { m, n }, { s, 1 }, make_const(Int(32), 0), "Ab", "", 0, 0);

    auto assert_simplified_equal = [](auto index_simplified, auto index_direct) {
        std::cout << "index_simplified=" << index_simplified << ", index_direct=" << index_direct << std::endl;
        assert(tvm::ir::Equal(index_simplified, index_direct));
    };

    // test case1
    auto index_simplified = A_stride.vload({ (k0 % k1) / s, (k0 % k1) % s + (k0 / k1) * k1 }, A_stride->dtype);
    auto index_direct = A_stride.vload({ 0, k0 }, A_stride->dtype);
    assert_simplified_equal(index_simplified, index_direct);

    // test case2
    index_simplified = A.vload({ (k0 % (k1 / s)) / n,
                                   (k0 % (k1 / s)) % n + (k0 % k1) },
        A->dtype);
    index_direct = A.vload({ 0, k0 % k1 + k0 % (k1 / s) },
        A->dtype);
    assert_simplified_equal(index_simplified, index_direct);

    // test case3
    index_simplified = A.vload({ ((k0 / (k1 / s)) * (k1 / s)) / n + (k0 % (k1 / s)) / n,
                                   ((k0 / (k1 / s)) * (k1 / s)) % n + (k0 % (k1 / s)) % n },
        A->dtype);
    index_direct = A.vload({ 0, k0 },
        A->dtype);
    assert_simplified_equal(index_simplified, index_direct);

    // test case4
    index_simplified = A.vload({ (k0 % (k1 / s)) / n,
                                   (k0 % (k1 / n)) % n + (k0 % k1) },
        A->dtype);
    index_direct = A.vload({ 0, ((k0 % (k1 / s)) / n) * n + ((k0 % (k1 / n)) % n + (k0 % k1)) },
        A->dtype);
    assert_simplified_equal(index_simplified, index_direct);
}
