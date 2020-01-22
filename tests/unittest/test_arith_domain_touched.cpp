#include "gtest/gtest.h"
#include <tvm/ir.h>
#include <tvm/operation.h>

template <typename T>
class TD;
using namespace tvm;
using namespace tvm::ir;

TEST(LANG_ARITH, test_domain_touched)
{
    Var i = Var("i");
    Var j = Var("j");
    Expr n = make_const(Int(32), 100);
    Var m = Var("m");
    Tensor a = placeholder({ n, m });
    Tensor b = placeholder({ n, m });
    Stmt ir = ir::For::make(i, 0, n, ForType::Serial, DeviceAPI::None,
        ir::For::make(j, 0, m, ForType::Serial, DeviceAPI::None,
            ir::Provide::make(a->op,
                0,
                ir::Call::make(b->dtype, "b", { i - 1, j + 1 }, ir::Call::Halide, b->op, 0) + ir::Call::make(a->dtype, "a", { i - 1, j - 1 }, ir::Call::Halide, a->op, 0),
                { i, j })));

    Domain a_domain_r = arith::DomainTouched(ir, a, true, false);
    CHECK_EQ(a_domain_r[0]->min.as<IntImm>()->value, -1);
    CHECK_EQ(a_domain_r[0]->extent.as<IntImm>()->value, 100);
    CHECK_EQ(a_domain_r[1]->min.as<IntImm>()->value, -1);
    CHECK_EQ(a_domain_r[1]->extent.as<Variable>()->name_hint, std::string("m"));

    Domain a_domain_w = arith::DomainTouched(ir, a, false, true);
    CHECK_EQ(a_domain_w[0]->min.as<IntImm>()->value, 0);
    CHECK_EQ(a_domain_w[0]->extent.as<IntImm>()->value, 100);
    CHECK_EQ(a_domain_w[1]->min.as<IntImm>()->value, 0);
    CHECK_EQ(a_domain_w[1]->extent.as<Variable>()->name_hint, std::string("m"));

    Domain a_domain_rw = arith::DomainTouched(ir, a, true, true);
    CHECK_EQ(a_domain_rw[0]->min.as<IntImm>()->value, -1);
    CHECK_EQ(a_domain_rw[0]->extent.as<IntImm>()->value, 101);
    CHECK_EQ(a_domain_rw[1]->min.as<IntImm>()->value, -1);
    CHECK(a_domain_rw[1]->extent->is_type<ir::Add>());
    CHECK_EQ(a_domain_rw[1]->extent.as<ir::Add>()->a.as<Variable>()->name_hint, std::string("m"));
    CHECK_EQ(a_domain_rw[1]->extent.as<ir::Add>()->b.as<IntImm>()->value, 1);

    Domain b_domain_r = arith::DomainTouched(ir, b, true, false);
    CHECK_EQ(b_domain_r[0]->min.as<IntImm>()->value, -1);
    CHECK_EQ(b_domain_r[0]->extent.as<IntImm>()->value, 100);
    CHECK_EQ(b_domain_r[1]->min.as<IntImm>()->value, 1);
    CHECK_EQ(b_domain_r[1]->extent.as<Variable>()->name_hint, std::string("m"));

    Domain b_domain_w = arith::DomainTouched(ir, b, false, true);
    // LOG(INFO) << b_domain_w->type_key();
    // CHECK(b_domain_w->is_type<Array<Expr>>());
    CHECK_EQ(b_domain_w.size(), 0);
}
