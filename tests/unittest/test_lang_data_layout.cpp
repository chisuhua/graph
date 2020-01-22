#include "gtest/gtest.h"
#include <tvm/data_layout.h>
#include <tvm/expr.h>

using namespace tvm;

TEST(LANG_DATA_LAYOUT, test_layout)
{
    Layout layout = LayoutNode::make("NCHW16c");
    LOG(INFO) << layout->type_key();
    CHECK(layout->type_key() == std::string("Layout"));
    CHECK(layout->is_type<LayoutNode>());
    CHECK(layout.FactorOf(LayoutAxis::Get('c')) == 16);
    CHECK(layout.FactorOf(LayoutAxis::Get('C')) == 16);
    CHECK(layout.FactorOf(LayoutAxis::Get('N')) == -1);

    CHECK(layout.IndexOf(LayoutAxis::Get('N')) == 0);
    CHECK(layout.IndexOf(LayoutAxis::Get('C')) == 1);
    CHECK(layout.IndexOf(LayoutAxis::Get('H')) == 2);
    CHECK(layout.IndexOf(LayoutAxis::Get('W')) == 3);
    CHECK(layout.IndexOf(LayoutAxis::Get('c')) == 4);
    CHECK(layout.IndexOf(LayoutAxis::Get('O')) == -1);

    CHECK(layout.Contains(LayoutAxis::Get('N')));
    CHECK(layout.Contains(LayoutAxis::Get('C')));
    CHECK(layout.Contains(LayoutAxis::Get('H')));
    CHECK(layout.Contains(LayoutAxis::Get('W')));
    CHECK(layout.Contains(LayoutAxis::Get('c')));
    CHECK(!layout.Contains(LayoutAxis::Get('O')));
}

TEST(LANG_DATA_LAYOUT, test_bilayout_convertible)
{
    auto layout1 = BijectiveLayoutNode::make(LayoutNode::make("NCHW"), LayoutNode::make("ABCD"));
    auto layout2 = BijectiveLayoutNode::make(LayoutNode::make("NCHW"), LayoutNode::make("NCHW16c"));
    // FIXME
    // CHECK(layout1 == nullptr);
    // CHECK(layout2 != nullptr);
}

TEST(LANG_DATA_LAYOUT, test_bilayout_shape)
{
    auto bilayout = BijectiveLayoutNode::make(LayoutNode::make("NCHW"), LayoutNode::make("NCHW16c"));
    CHECK_EQ(bilayout->type_key(), "BijectiveLayout");
    CHECK(bilayout->is_type<BijectiveLayoutNode>());
    auto dst_shape = bilayout.ForwardShape({ 1, 32, 7, 7 });
    auto src_shape = bilayout.BackwardShape(dst_shape);
    CHECK(src_shape[0].as<IntImm>()->value == 1);
    CHECK(src_shape[1].as<IntImm>()->value == 32);
    CHECK(src_shape[2].as<IntImm>()->value == 7);
    CHECK(src_shape[3].as<IntImm>()->value == 7);
    // CHECK(src_shape.same_as(Array<Expr>{1, 32, 7, 7}));
}

TEST(LANG_DATA_LAYOUT, test_bilayout_index)
{
    auto bilayout = BijectiveLayoutNode::make(LayoutNode::make("NCHW"), LayoutNode::make("NCHW16c"));

    auto dst_index = bilayout.ForwardIndex({ 0, 18, 6, 6 });
    auto src_index = bilayout.BackwardIndex({ 0, 1, 6, 6, 2 });
    LOG(INFO) << "src_index" << src_index;
    LOG(INFO) << "dst_index" << dst_index;
    CHECK(dst_index[0].as<IntImm>()->value == 0);
    CHECK(dst_index[1].as<IntImm>()->value == 1);
    CHECK(dst_index[2].as<IntImm>()->value == 6);
    CHECK(dst_index[3].as<IntImm>()->value == 6);
    CHECK(dst_index[4].as<IntImm>()->value == 2);

    CHECK(src_index[0].as<IntImm>()->value == 0);
    CHECK(src_index[1].as<IntImm>()->value == 18);
    CHECK(src_index[2].as<IntImm>()->value == 6);
    CHECK(src_index[3].as<IntImm>()->value == 6);
    // CHECK(src_shape.same_as(Array<Expr>{1, 32, 7, 7}));
}
