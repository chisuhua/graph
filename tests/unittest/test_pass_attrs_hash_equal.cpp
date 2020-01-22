#include "gtest/gtest.h"
#include <tvm/api_registry.h>
#include <tvm/attrs.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/runtime/packed_func.h>

template <typename T>
class TD;

using namespace tvm;

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

TEST(LANG_PASS_ATTR, test_attrs_equal)
{
    auto x = make_node<test::TestAttrs>();
    x->InitBySeq("name", "xx", "padding", Array<Expr>({ 3, 4 }));

    auto y = make_node<test::TestAttrs>();
    y->InitBySeq("name", "xx", "padding", Array<Expr>({ 3, 4 }));

    auto z = make_node<test::TestAttrs>();
    Array<Expr> padding { make_const(Int(32), 3), make_const(Int(32), 4), make_const(Int(32), 1) };
    z->InitBySeq("name", "xx", "padding", padding);

    CHECK(AttrsEqual()(static_cast<NodeRef>(x), static_cast<NodeRef>(y)));
    CHECK(!AttrsEqual()(static_cast<NodeRef>(x), static_cast<NodeRef>(z)));

    Map<std::string, NodeRef> d;
    d.Set("x", make_const(Int(32), 1));
    d.Set("y", make_const(Int(32), 10));
    d.Set("name", ir::StringImm::make("xyz"));
    d.Set("padding", Array<Expr>({ 0, 0 }));
    auto dattr = DictAttrsNode::make(d);
    auto dattr2 = DictAttrsNode::make(d);
    CHECK(AttrsEqual()(static_cast<NodeRef>(dattr), static_cast<NodeRef>(dattr2)));

    Map<std::string, NodeRef> dx, dy;
    dx.Set("x", static_cast<NodeRef>(x));
    dy.Set("x", static_cast<NodeRef>(y));
    CHECK(AttrsEqual()(static_cast<NodeRef>(dx), static_cast<NodeRef>(dy)));
}
/*
def test_attrs_equal():
    x = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3, 4))
    y = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3, 4))
    z = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3,4,1))
    assert tvm.ir_pass.AttrsEqual(x, y)
    assert not tvm.ir_pass.AttrsEqual(x, z)

    dattr = tvm.make.node("DictAttrs", x=1, y=10, name="xyz", padding=(0,0))
    assert not tvm.ir_pass.AttrsEqual(dattr, x)
    dattr2 = tvm.make.node("DictAttrs", x=1, y=10, name="xyz", padding=(0,0))
    assert tvm.ir_pass.AttrsEqual(dattr, dattr2)

    assert tvm.ir_pass.AttrsEqual({"x": x}, {"x": y})
    # array related checks
    assert tvm.ir_pass.AttrsEqual({"x": [x, x]}, {"x": [y, x]})
    assert not tvm.ir_pass.AttrsEqual({"x": [x, 1]}, {"x": [y, 2]})

    n = tvm.var("n")
    assert tvm.ir_pass.AttrsEqual({"x": n+1}, {"x": n+1})
*/

TEST(LANG_PASS_ATTR, test_attrs_hash)
{
    AttrsHash hasher;
    auto x = make_node<test::TestAttrs>();
    x->InitBySeq("name", "xx", "padding", Array<Expr>({ 3, 4 }));

    auto y = make_node<test::TestAttrs>();
    y->InitBySeq("name", "xx", "padding", Array<Expr>({ 3, 4 }));

    Map<std::string, NodeRef> dx, dy;
    dx.Set("x", static_cast<NodeRef>(x));
    dy.Set("x", static_cast<NodeRef>(y));
    auto dxattr = DictAttrsNode::make(dx);
    auto dyattr = DictAttrsNode::make(dy);

    CHECK_EQ(hasher(dxattr), hasher(dxattr));
}

/*
     fhash = tvm.ir_pass.AttrsHash
    x = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3, 4))
    y = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3, 4))
    assert fhash({"x": x}) == fhash({"x": y})
    assert fhash({"x": x}) != fhash({"x": [y, 1]})
    assert fhash({"x": [x, 1]}) == fhash({"x": [y, 1]})
    assert fhash({"x": [x, 2]}) == fhash({"x": [y, 2]})
*/
