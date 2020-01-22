#include "gtest/gtest.h"
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/tensor.h>

using namespace tvm;

template <typename T>
class TD;

TEST(LANG_SCHEDULE, test_schedule_create)
{
    Var m("m"), n("n"), l("l");
    Tensor A = placeholder({ m, l });
    Tensor B = placeholder({ n, l });
    Tensor AA = compute({ n, l }, [&](Var i, Var j) { return A[i][j]; });
    Tensor T = compute({ m, n, l }, [&](Var i, Var j, Var k) { return AA[i][k] * B[j][k]; });
    Schedule s = create_schedule({ T->op });
    s[AA].set_scope("shared");
    IterVar xo, xi;
    IterVar xi1, xi2;
    s[T].split(T->op.as<ComputeOpNode>()->axis[0], 10, &xo, &xi);
    s[T].split(xi, 2, &xi1, &xi2);
    s[AA].compute_at(s[T], xi1);
    s[AA].split(AA->op.as<ComputeOpNode>()->axis[0], 10, &xo, &xi);
    s[T].reorder({ xi2, xi1 });

    // save load json
    auto json_str = SaveJSON(s);
    auto s_loaded = LoadJSON<Schedule>(json_str);
    // CHECK_EQ(s_loaded->type_key(), "Schedule");
    CHECK(s_loaded->is_type<ScheduleNode>());
    std::ostringstream lhs, rhs;
    lhs << s_loaded->outputs[0].as<ComputeOpNode>()->body[0];
    rhs << s->outputs[0].as<ComputeOpNode>()->body[0];
    CHECK(lhs.str() == rhs.str());
    LOG(INFO) << lhs.str();
}

TEST(LANG_SCHEDULE, test_reorder)
{
    Var m("m");
    Tensor A = placeholder({ m });
    Tensor T = compute({ m }, [&](Var i) { return A[i + 1]; });

    Schedule s = create_schedule({ T->op });
    IterVar xo, xi;
    s[T].split(T->op.as<ComputeOpNode>()->axis[0], 10, &xo, &xi);
    IterVar xi1, xi2;
    s[T].split(xi, 2, &xi1, &xi2);
    std::ostringstream old_order, new_order;
    old_order << s[T]->leaf_iter_vars;
    LOG(INFO) << "Old order is " << old_order.str();
    CHECK_EQ(old_order.str(), "[iter_var(ax0.outer, ), iter_var(ax0.inner.outer, ), iter_var(ax0.inner.inner, )]");
    auto order = { xi2, xi1, xo };
    s[T].reorder(order);
    new_order << s[T]->leaf_iter_vars;
    LOG(INFO) << "New order is " << new_order.str();
    CHECK_EQ(new_order.str(), "[iter_var(ax0.inner.inner, ), iter_var(ax0.inner.outer, ), iter_var(ax0.outer, )]");
}

TEST(LANG_SCHEDULE, test_split)
{
    Var m("m");
    Tensor A = placeholder({ m });
    Tensor T = compute({ m }, [&](Var i) { return A[i]; });

    Schedule s = create_schedule({ T->op });

    IterVar xo, xi;
    s[T].split(T->op.as<ComputeOpNode>()->axis[0], 10, &xo, &xi);
    std::ostringstream order;
    order << s[T]->leaf_iter_vars;
    LOG(INFO) << "order is " << order.str();
    CHECK_EQ(order.str(), "[iter_var(ax0.outer, ), iter_var(ax0.inner, )]");
}

TEST(LANG_SCHEDULE, test_tile)
{
    Var m("m");
    Var n("n");
    Tensor A = placeholder({ m, n });
    Tensor T = compute({ m, n }, [&](Var i, Var j) { return A[i][j]; });

    Schedule s = create_schedule({ T->op });

    IterVar xo, yo, xi, yi;
    s[T].tile(T->op.as<ComputeOpNode>()->axis[0], T->op.as<ComputeOpNode>()->axis[1], 10, 5, &xo, &yo, &xi, &yi);
    std::ostringstream order;
    order << s[T]->leaf_iter_vars;
    LOG(INFO) << "order is " << order.str();
    CHECK_EQ(order.str(), "[iter_var(ax0.outer, ), iter_var(ax1.outer, ), iter_var(ax0.inner, ), iter_var(ax1.inner, )]");
}

TEST(LANG_SCHEDULE, test_fuse)
{
    Var m("m");
    Var n("n");
    Tensor A = placeholder({ m, n });
    Tensor T = compute({ m, n }, [&](Var i, Var j) { return A[i][j]; });

    Schedule s = create_schedule({ T->op });

    IterVar xo, yo, xi, yi;
    s[T].tile(T->op.as<ComputeOpNode>()->axis[0], T->op.as<ComputeOpNode>()->axis[1], 10, 5, &xo, &yo, &xi, &yi);
    IterVar fused;
    s[T].fuse(xo, yo, &fused);
    bool have_fuse_relation = false;
    for (auto& itr : s[T]->relations) {
        // if (itr->type_key() == std::string("Fuse")) {
        if (itr->is_type<FuseNode>()) {
            have_fuse_relation = true;
        }
    }
    CHECK(have_fuse_relation);
    /*
    std::any_of(s[T]->relations.begin(), s[T]->relations.end(), [&](auto iter){
            std::cout << iter << std::endl;
            });
            */

    std::ostringstream order;
    order << s[T]->leaf_iter_vars;
    LOG(INFO) << "order is " << order.str();
    // CHECK_EQ(order.str(), "[iter_var(ax0.outer, ), iter_var(ax1.outer, ), iter_var(ax0.inner, ), iter_var(ax1.inner, )]");
}

inline Tensor compute(Array<Expr> shape,
    std::function<Expr()> f,
    std::string name = "tensor",
    std::string tag = "",
    Map<std::string, NodeRef> attrs = {})
{
    FCompute fc = [f](const Array<Var>& i) { return f(); };
    return compute(shape, fc, name, tag, attrs);
}

TEST(LANG_SCHEDULE, test_singleton)
{
    Tensor A = placeholder({});
    Tensor T = compute(Array<Expr>(), [&]() { return A() + 1; });
    Schedule s = create_schedule({ T->op });
    // FIXME
    // IterVar fused;
    // s[T].fuse(xo, yo, &fused);
}

TEST(LANG_SCHEDULE, test_vectorize)
{
    Var m("m");
    Var n("n");
    Tensor A = placeholder({ m, n });
    Tensor T = compute({ m, n }, [&](Var i, Var j) { return A[i][j]; });

    Schedule s = create_schedule({ T->op });

    IterVar xo, yo, xi, yi;
    s[T].tile(T->op.as<ComputeOpNode>()->axis[0], T->op.as<ComputeOpNode>()->axis[1], 10, 5, &xo, &yo, &xi, &yi);
    s[T].vectorize(yi);
    s[T].unroll(xi);

    CHECK_EQ(s[T]->iter_var_attrs[xi]->iter_type, IterVarType::kUnrolled);
    CHECK_EQ(s[T]->iter_var_attrs[yi]->iter_type, IterVarType::kVectorized);
}

// TODO TEST(LANG_SCHEDULE, test_vectorize_commreduce)

TEST(LANG_SCHEDULE, test_pragma)
{
    // #auto m = 100u;
    Var m("m");
    Tensor A = placeholder({ m });
    Tensor T = compute({ m }, [&](Var i) { return A[i]; });

    Schedule s = create_schedule({ T->op });

    IterVar xo, xi;
    s[T].split(T->op.as<ComputeOpNode>()->axis[0], 10, &xo, &xi);

    s[T].pragma(xo, "pragma1");
    s[T].pragma(xi, "vectorize");

    CHECK_EQ(s[T]->iter_var_attrs[xo]->pragma_keys[0].as<ir::StringImm>()->value, "pragma1");
    CHECK_EQ(s[T]->iter_var_attrs[xi]->iter_type, IterVarType::kVectorized);
}

TEST(LANG_SCHEDULE, test_rfactor)
{
    Var n("n");
    IterVar k1 = reduce_axis(Range { 0, n }, "k1");
    IterVar k2 = reduce_axis(Range { 0, n }, "k2");
    Tensor A = placeholder({ n, n, n });
    Tensor B = compute({ n }, [&](Var i) { return sum(A[i][k1][k2], { k1, k2 }); });

    Schedule s = create_schedule({ B->op });

    auto BF = s.rfactor(B, k1);
    CHECK(BF[0]->shape[0].same_as(n));
    CHECK(BF[0]->shape[1].same_as(n));

    // auto BF = BF.rfactor(B, k1);
    // TODO
}

// TODO
// TEST(LANG_SCHEDULE, test_intrin) {
