#include "gtest/gtest.h"
// #include <dmlc/logging.h>
#include <tvm/api_registry.h>
#include <tvm/attrs.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/operation.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tensor.h>

using namespace tvm;
using namespace tvm::runtime;
// Attrs used to python API
namespace tvm {
namespace test {
    // test example usage docs
    struct TestAttrs : public AttrsNode<TestAttrs> {
        int axis;
        std::string name;
        Expr expr;
        double learning_rate;
        Array<Expr> padding;
        TypedEnvFunc<int(int)> func;

        TVM_DECLARE_ATTRS(TestAttrs, "attrs.TestAttrs")
        {
            TVM_ATTR_FIELD(axis)
                .set_default(10)
                .set_lower_bound(1)
                .set_upper_bound(10)
                .describe("axis field");
            TVM_ATTR_FIELD(name)
                .describe("name of the field");
            TVM_ATTR_FIELD(expr)
                .describe("expression field")
                .set_default(make_const(Int(32), 1));
            TVM_ATTR_FIELD(learning_rate)
                .describe("learning_rate")
                .set_default(0.1);
            TVM_ATTR_FIELD(padding)
                .describe("padding of input")
                .set_default(Array<Expr>({ 0, 0 }));
            TVM_ATTR_FIELD(func)
                .describe("some random env function")
                .set_default(TypedEnvFunc<int(int)>(nullptr));
        }
    };
}
}

TEST(LANG_REFLECTION, test_const_saveload_json)
{
    auto x = make_const(UInt(32), 1);
    auto y = make_const(UInt(32), 10);
    auto z = x + y;
    z = z + z;
    auto json_str = SaveJSON(z);
    auto zz = LoadJSON<Array<Expr>>(json_str);
    assert(SaveJSON(zz) == json_str);
}

TEST(LANG_REFLECTION, test_make_smap)
{
    auto x = make_const(Int(32), 1);
    auto y = make_const(Int(32), 10);
    auto z = ir::Add::make(x, y);
    Map<std::string, Expr> smap { { "z", z }, { "x", x } };
    Array<decltype(smap)> v_smap; // = {smap};
    v_smap.push_back(smap);
    auto json_str = SaveJSON(v_smap);
    auto arr = LoadJSON<Array<Map<std::string, Expr>>>(json_str);
    std::cout << arr << std::endl;
    std::cout << arr.size() << std::endl;
    assert(arr.size() == 1);
    assert(arr[0]["z"].as<ir::Add>()->a.same_as(arr[0]["x"]));
}

TEST(LANG_REFLECTION, test_make_node)
{
    auto x = IntImm::make(Int(32), 10);
    assert(x->type_key() == std::string("IntImm"));
    assert(x.as<IntImm>()->value == 10);

    Tensor A = placeholder({
                               10,
                           },
        Int(32), "A");
    Array<Expr> shape = A->op.as<PlaceholderOpNode>()->output_shape(0);
    Type dtype = A->op.as<PlaceholderOpNode>()->output_dtype(0);
    // std::cout << shape << std::endl;
    Tensor AA = TensorNode::make(shape, dtype, A->op, 0);

    assert(AA->op.same_as(A->op));
    assert(AA->value_index == A->value_index);
}

TEST(LANG_REFLECTION, test_make_attrs)
{
    try {
        auto x = make_node<test::TestAttrs>();
        x->InitBySeq("unknown_key", 1, "name", "xx");
        LOG(FATAL) << "bad";
    } catch (tvm::AttrError& e) {
        std::string what = e.what();
        std::cout << what << std::endl;
        CHECK(what.find("unknown_key") != std::string::npos);
    }

    try {
        auto x = make_node<test::TestAttrs>();
        x->InitBySeq("axis", 100, "name", "xx");
        LOG(FATAL) << "bad";
    } catch (tvm::AttrError& e) {
        std::string what = e.what();
        CHECK(what.find("upper bound") != std::string::npos);
    }
    auto x = make_node<test::TestAttrs>();
    Array<Expr> padding { make_const(Int(32), 3), make_const(Int(32), 4) };
    x->InitBySeq("name", "xx", "padding", padding);
    CHECK_EQ(x->name, "xx");
    CHECK_EQ(x->padding[0].as<IntImm>()->value, 3);
    CHECK_EQ(x->padding[1].as<IntImm>()->value, 4);
    CHECK_EQ(x->axis, 10);

    Map<std::string, NodeRef> d;
    d.Set("x", make_const(Int(32), 1));
    d.Set("y", make_const(Int(32), 10));
    d.Set("name", ir::StringImm::make("xyz"));
    d.Set("padding", Array<Expr>({ 0, 0 }));
    auto dattr = DictAttrsNode::make(d);

    CHECK_EQ(dattr.as<DictAttrsNode>()->dict["x"].as<IntImm>()->value, 1);
    dattr = LoadJSON<decltype(dattr)>(SaveJSON(dattr));
    CHECK_EQ(dattr.as<DictAttrsNode>()->dict["name"].as<ir::StringImm>()->value, "xyz");
}

template <typename T>
class TD;

TEST(LANG_REFLECTION, test_make_sum)
{
    Tensor A = placeholder({ 2, 10 }, Int(32), "A");
    IterVar k = reduce_axis(Range { 0, 10 }, "k");

    Tensor B = compute(
        {
            2,
        },
        [&](Var i) {
            return sum(A[i][k], { k });
        },
        "B");

    // TD<decltype(B)> btype;
    auto json_str = SaveJSON(B);
    Tensor BB = LoadJSON<decltype(B)>(json_str);
    // TD<decltype(B->op.as<ComputeOpNode>()->body[0])> tType;
    LOG(INFO) << B->op.as<ComputeOpNode>()->body[0].as<ir::Reduce>()->combiner->result;
    CHECK(B->op.as<ComputeOpNode>()->body[0].as<ir::Reduce>()->combiner->result.size() >= 1);
    CHECK(BB->op.as<ComputeOpNode>()->body[0].as<ir::Reduce>()->combiner->result.size() >= 1);
}

TEST(LANG_REFLECTION, test_env_func)
{
    auto addone = [](int x) -> int {
        return x + 1;
    };
    TypedPackedFunc<int(int)> ftyped(addone);
    PackedFunc packed = ftyped;

    TVMFunctionHandle* f = reinterpret_cast<TVMFunctionHandle*>(&packed);
    TVMFuncRegisterGlobal("test.env_func", f, 0);
    // TVMFuncGetGlobal("test.env_func", &x);
    EnvFunc x = EnvFunc::Get("test.env_func");
    CHECK_EQ(x->name, "test.env_func");
    Array<EnvFunc> xx { x };
    auto json_str = SaveJSON(xx);
    auto yy = LoadJSON<Array<EnvFunc>>(json_str);
    EnvFunc y = yy[0];

    CHECK_EQ(x->name, y->name);
    CHECK_EQ((int)(y(1)), 2);
    CHECK((int)(y->func(1)) == 2);

    auto a = make_node<test::TestAttrs>();
    Array<Expr> padding { make_const(Int(32), 3), make_const(Int(32), 4) };
    a->InitBySeq("name", "xx", "padding", padding, "func", y);
    CHECK_EQ(a->name, "xx");
    CHECK_EQ(a->padding[0].as<IntImm>()->value, 3);
    CHECK_EQ(a->padding[1].as<IntImm>()->value, 4);
    CHECK_EQ(a->axis, 10);
    Array<EnvFunc> aa { a };
    auto a_json = LoadJSON<decltype(aa)>(SaveJSON(aa));
    // assert(a_json[0]->type_key() == std::string("EvnFunc"));
    // CHECK_EQ(int(a_json[0]->func(10)), 11);
    /*
    x = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3,4), func=y)
    x = tvm.load_json(tvm.save_json(x))
    assert isinstance(x.func, tvm.container.EnvFunc)
    assert x.func(10) == 11
*/
}
