#include "gtest/gtest.h"
// #include "ir/IROperator.h"
#include <tvm/arithmetic.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/operation.h>
#include <type_traits>

template <typename T>
class TD;

using namespace tvm;

namespace detail {
// implementation details for  CallExpr
template <bool stop, std::size_t I, typename F>
struct tuple_for_each_dispatcher {
    template <typename TTuple>
    static void run(F& f, const TTuple& tuple)
    { // NOLINT(*)
        f(I, std::get<I>(tuple));
        tuple_for_each_dispatcher<
            (I + 1) == std::tuple_size<TTuple>::value, (I + 1), F>::run(f, tuple);
    }
};

template <std::size_t I, typename F>
struct tuple_for_each_dispatcher<true, I, F> {
    template <typename TTuple>
    static void run(F& f, const TTuple& tuple) {} // NOLINT(*)
};

template <typename F, typename TTuple>
inline void tuple_for_each(F& f, const TTuple& tuple)
{ // NOLINT(*)
    tuple_for_each_dispatcher<std::tuple_size<TTuple>::value == 0, 0, F>::run(f, tuple);
}

struct Functor {
    Expr m_a, m_b;

    explicit Functor(const Expr& a, const Expr& b)
        : m_a(a)
        , m_b(b)
    {
    }

    template <typename T>
    void operator()(size_t i, T&& t) const
    {
        // auto f = std::make_shared<T>();
        using U = typename std::decay<T>::type;
        Expr x = t.make(m_a, m_b);
        std::cout << "x.a is " << x.as<U>()->a << ",  m_a is " << m_a << std::endl;

        // assert( (std::is_same<typename std::decay<decltype(x)>::type,U>::value) == true );
        assert(x.get()->is_type<U>());

        Expr a = x.as<U>()->a;
        std::cout << a.as<ir::FloatImm>()->value << std::endl;
        assert(a.as<ir::FloatImm>()->value == m_a.as<ir::FloatImm>()->value);
        assert(x.as<U>()->b.same_as(m_b));
    }
};
}

TEST(LANG_CONSTRUCTOR, test_expr_constructor)
{
    auto x = tvm::Var("xx", HalideIR::Float(32));
    assert(x->name_hint == "xx");
    /*
    Var _x {"_x"};
    Var _y {"_y"};
    Expr result = ir::And::make(_x, _y);
    Expr identity_element = make_zero(HalideIR::Int(32));
    auto combiner = ir::CommReducerNode::make({_x}, {_y}, {result}, { identity_element });
//  Expr cond = const_true();
*/
    {
        IterVar rv = reduce_axis(Range { 0, 1 }, "x");
        auto x = ir::Reduce::make(ir::CommReducer(), { make_const(HalideIR::Int(32), 1) }, { rv }, Expr(), 0);
        // assert(x->type_key() == std::string("Reduce"));
        CHECK(x->is_type<ir::Reduce>());
        assert(x.as<ir::Reduce>()->combiner == ir::CommReducer());
        assert(x.as<ir::Reduce>()->value_index == 0);
    }
    {
        auto x = ir::FloatImm::make(HalideIR::Float(32), 1.0);
        // std::cout << x.type() << std::endl;
        // assert(x->type_key() == std::string("FloatImm"));
        CHECK(x->is_type<ir::FloatImm>());
        assert(x.as<ir::FloatImm>()->value == 1.0);
        assert(x.type() == HalideIR::Float(32));
    }
    {
        auto x = ir::IntImm::make(HalideIR::Int(64), 2);
        // std::cout << x.type() << std::endl;
        // assert(x->type_key() == std::string("IntImm"));
        CHECK(x->is_type<ir::IntImm>());
        assert(x.as<ir::IntImm>()->value == 2);
        assert(x.type() == HalideIR::Int(64));
    }
    {
        auto x = ir::UIntImm::make(HalideIR::UInt(16), 2);
        // std::cout << x.type() << std::endl;
        // assert(x->type_key() == std::string("UIntImm"));
        CHECK(x->is_type<ir::UIntImm>());
        assert(x.as<ir::UIntImm>()->value == 2);
        assert(x.type() == HalideIR::UInt(16));
    }
    {
        auto x = ir::StringImm::make("xyza");
        std::cout << x.type() << std::endl;
        // assert(x->type_key() == std::string("StringImm"));
        CHECK(x->is_type<ir::StringImm>());
        assert(x.as<ir::StringImm>()->value == "xyza");
    }
    {
        auto x = ir::Cast::make(HalideIR::Float(32), make_const(HalideIR::Int(32), 1));
        std::cout << x.type() << std::endl;
        std::cout << x->type_key() << std::endl;
        // assert(x->type_key() == std::string("Cast"));
        CHECK(x->is_type<ir::Cast>());
        assert((x.as<ir::Cast>()->value).as<ir::IntImm>()->value == 1);
        assert(x.type() == HalideIR::Float(32));
    }

    {
        auto a = make_const(HalideIR::Float(32), 1.0);
        auto b = Var("x", HalideIR::Float(32));

        auto binops = std::make_tuple(ir::Add(),
            ir::Sub(),
            ir::Mul(),
            ir::Div(),
            ir::Mod(),
            ir::Min(),
            ir::Max(),
            ir::LT(),
            ir::LE(),
            ir::GT(),
            ir::GE());

        detail::Functor f(a, b);
        detail::tuple_for_each(f, binops);
    }

    auto a = tvm::Var("x") > 1;
    auto b = tvm::Var("x") == 1;
    {
        auto binops = std::make_tuple(ir::And(),
            ir::Or());
        detail::Functor f(a, b);
    }
    {
        auto x = !a;
        assert(x.get()->is_type<ir::Not>());
        // TD<decltype(x)> xType;
        assert(x.as<ir::Not>()->a.same_as(a));
    }
    {
        auto x = ir::Select::make(a, a, b);
        assert(x.get()->is_type<ir::Select>());
        assert(x.as<ir::Select>()->true_value.same_as(a));
        assert(x.as<ir::Select>()->false_value.same_as(b));
        assert(x.as<ir::Select>()->condition.same_as(a));
    }
    {
        auto buffer_var = tvm::Var("x", HalideIR::Handle());
        auto x = ir::Load::make(HalideIR::Float(32), buffer_var, 1, a);
        assert(x.get()->is_type<ir::Load>());
        assert(x.type() == HalideIR::Float(32));
        assert(x.as<ir::Load>()->buffer_var.same_as(buffer_var));
        assert(x.as<ir::Load>()->index.as<ir::IntImm>()->value == 1);
        assert(x.as<ir::Load>()->predicate.same_as(a));
    }
    {
        auto x = ir::Ramp::make(1, 2, 10);
        assert(x.get()->is_type<ir::Ramp>());
        assert(x.as<ir::Ramp>()->base.as<ir::IntImm>()->value == 1);
        assert(x.as<ir::Ramp>()->stride.as<ir::IntImm>()->value == 2);
        assert(x.as<ir::Ramp>()->lanes == 10);
    }
    {
        auto x = ir::Broadcast::make(a, 10);
        assert(x.get()->is_type<ir::Broadcast>());
        assert(x.as<ir::Broadcast>()->value.same_as(a));
        assert(x.as<ir::Broadcast>()->lanes == 10);
    }
    {
        auto x = ir::Shuffle::make({ a }, { 0 });
        assert(x.get()->is_type<ir::Shuffle>());
        assert(x.as<ir::Shuffle>()->vectors[0].same_as(a));
        assert((x.as<ir::Shuffle>()->indices[0]).as<ir::IntImm>()->value == 0);
    }
    {
        Expr x = ir::Call::make(HalideIR::Float(32), "xyz", { a }, ir::Call::Extern);
        assert(x.get()->is_type<ir::Call>());
        assert(x.type() == HalideIR::Float(32));
        assert(x.as<ir::Call>()->name == std::string("xyz"));
        assert((x.as<ir::Call>()->args[0]).same_as(a));
        assert(x.as<ir::Call>()->call_type == ir::Call::Extern);
        assert(x.as<ir::Call>()->func == HalideIR::IR::FunctionRef());
        assert(x.as<ir::Call>()->value_index == 0);
    }
    {
        auto v = tvm::Var("aa");
        auto x = ir::Let::make(v, 1, v);
        assert(x.as<ir::Let>()->var.same_as(v));
        assert(x.as<ir::Let>()->value.as<ir::IntImm>()->value == 1);
        assert(x.as<ir::Let>()->body.same_as(v));
    }
}

TEST(LANG_CONSTRUCTOR, test_stmt_constructor)
{
    auto v = tvm::Var("aa");
    auto buffer_var = tvm::Var("buf", HalideIR::Handle());
    auto nop = ir::Evaluate::make(1);
    {
        auto x = ir::LetStmt::make(v, 1, ir::Evaluate::make(1));
        assert(x.get()->is_type<ir::LetStmt>());
        assert(x.as<ir::LetStmt>()->var.same_as(v));
        assert(x.as<ir::LetStmt>()->value.as<ir::IntImm>()->value == 1);
        assert(x.as<ir::LetStmt>()->body.get()->is_type<ir::Evaluate>());
    }
    {
        auto x = ir::AttrStmt::make(v == 1, "xx", 1, ir::Evaluate::make(1));
        assert(x.get()->is_type<ir::AttrStmt>());
        assert(x.as<ir::AttrStmt>()->value.as<ir::IntImm>()->value == 1);
    }
    {
        auto x = ir::Block::make(ir::Evaluate::make(11), nop);
        assert(x.get()->is_type<ir::Block>());
        assert(x.as<ir::Block>()->first.as<ir::Evaluate>()->value.as<ir::IntImm>()->value == 11);
        assert(x.as<ir::Block>()->rest.same_as(nop));
    }
    {
        auto x = ir::AssertStmt::make(make_const(HalideIR::UInt(1), 1),
            ir::StringImm::make("hellow"),
            nop);
        assert(x.get()->is_type<ir::AssertStmt>());
        assert(x.as<ir::AssertStmt>()->body.same_as(nop));
    }
    {
        auto x = ir::ProducerConsumer::make(FunctionRef(), true, nop);
        assert(x.get()->is_type<ir::ProducerConsumer>());
        assert(x.as<ir::ProducerConsumer>()->body.same_as(nop));
    }
    {
        auto x = ir::For::make(tvm::Var("x"), make_const(HalideIR::Int(32), 0), make_const(HalideIR::Int(32), 10), ir::ForType::Serial, ir::DeviceAPI::None, nop);
        assert(x.get()->is_type<ir::For>());
        assert(x.as<ir::For>()->min.as<ir::IntImm>()->value == 0);
        assert(x.as<ir::For>()->extent.as<ir::IntImm>()->value == 10);
        assert(x.as<ir::For>()->body.same_as(nop));
    }
    {
        auto x = ir::Store::make(buffer_var, 1, 10, make_const(HalideIR::UInt(1), 1));
        assert(x.get()->is_type<ir::Store>());
        assert(x.as<ir::Store>()->buffer_var.same_as(buffer_var));
        assert(x.as<ir::Store>()->index.as<ir::IntImm>()->value == 10);
        assert(x.as<ir::Store>()->value.as<ir::IntImm>()->value == 1);
    }
    {
        auto t = placeholder({}, Float(32));
        auto x = ir::Provide::make(t->op, 0, 10, {});
        assert(x.get()->is_type<ir::Provide>());
        assert(x.as<ir::Provide>()->value_index == 0);
        assert(x.as<ir::Provide>()->value.as<ir::IntImm>()->value == 10);
    }
    {
        auto x = ir::Allocate::make(buffer_var, Float(32), { 10 }, make_const(UInt(1), 1), nop);
        assert(x.get()->is_type<ir::Allocate>());
        assert(x.as<ir::Allocate>()->type == Float(32));
        assert(x.as<ir::Allocate>()->buffer_var.same_as(buffer_var));
        assert(x.as<ir::Allocate>()->body.same_as(nop));
    }
    {
        auto x = ir::AttrStmt::make(buffer_var, "xyz", make_const(UInt(1), 1), nop);
        assert(x.get()->is_type<ir::AttrStmt>());
        assert(x.as<ir::AttrStmt>()->node.same_as(buffer_var));
        assert(x.as<ir::AttrStmt>()->attr_key == "xyz");
        assert(x.as<ir::AttrStmt>()->body.same_as(nop));
    }
    {
        auto x = ir::Free::make(buffer_var);
        assert(x.get()->is_type<ir::Free>());
        assert(x.as<ir::Free>()->buffer_var.same_as(buffer_var));
    }
    {
        auto x = ir::Realize::make(FunctionRef(), 0, Float(32), {}, make_const(UInt(1), 1), nop);
        assert(x.get()->is_type<ir::Realize>());
        assert(x.as<ir::Realize>()->body.same_as(nop));
    }
    {
        auto x = ir::IfThenElse::make(make_const(UInt(1), 1), ir::Evaluate::make(11), nop);
        assert(x.get()->is_type<ir::IfThenElse>());
        assert(x.as<ir::IfThenElse>()->then_case.as<ir::Evaluate>()->value.as<ir::IntImm>()->value == 11);
        assert(x.as<ir::IfThenElse>()->else_case.same_as(nop));
    }
    {
        auto x = ir::Prefetch::make(FunctionRef(), 1, Float(32), {});
        assert(x.get()->is_type<ir::Prefetch>());
        assert(x.as<ir::Prefetch>()->value_index == 1);
    }
}
/*

def test_stmt_constructor():

    x = tvm.stmt.Prefetch(None, 1, "float32", [])
    assert isinstance(x, tvm.stmt.Prefetch)
    assert x.value_index == 1

*/
