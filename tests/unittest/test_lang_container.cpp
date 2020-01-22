#include "gtest/gtest.h"
// #include "ir/IROperator.h"
#include <tvm/arithmetic.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/operation.h>
#include <type_traits>
#include <vector>

template <typename T>
class TD;

using namespace tvm;

TEST(LANG_CONTAINER, test_array)
{
    Array<Expr> a { 1, 2, 3 };
    assert(a.size() == 3);
    assert(a[2].as<IntImm>()->value == 3);
}
TEST(LANG_CONTAINER, test_save_load_json)
{
    Array<Expr> a { 1, 2, 3 };
    auto json_str = SaveJSON(a);
    auto a_loaded = LoadJSON<Array<Expr>>(json_str);
    assert(a_loaded[1].as<IntImm>()->value == 2);
}
TEST(LANG_CONTAINER, test_map)
{
    Var a("a"), b("b");
    Map<Expr, Expr> amap { { a, make_const(Int(32), 2) }, { b, make_const(Int(32), 3) } };
    size_t t = amap.count(a);
    assert(t == 1);
    size_t len = amap.size();
    assert(len == 2);
    assert(amap.count(a + 1) == 0);
}
TEST(LANG_CONTAINER, test_str_map)
{
    Map<std::string, Expr> amap { { "a", make_const(Int(32), 2) }, { "b", make_const(Int(32), 3) } };
    size_t t = amap.count("a");
    assert(t == 1);
    size_t len = amap.size();
    assert(len == 2);
    assert(amap["a"].as<IntImm>()->value == 2);
}

/*
def test_array():
    a = tvm.convert([1,2,3])
    assert len(a) == 3
    assert a[-1].value == 3
    a_slice = a[-3:-1]
    assert (a_slice[0].value, a_slice[1].value) == (1, 2)

def test_str_map():
    amap = tvm.convert({'a': 2, 'b': 3})
    assert 'a' in amap
    assert len(amap) == 2
    dd = dict(amap.items())
    assert amap['a'].value == 2
    assert 'a' in dd
    assert 'b' in dd


def test_map_save_load_json():
    a = tvm.var('a')
    b = tvm.var('b')
    amap = tvm.convert({a: 2,
                        b: 3})
    json_str = tvm.save_json(amap)
    amap = tvm.load_json(json_str)
    assert len(amap) == 2
    dd = {kv[0].name : kv[1].value for kv in amap.items()}
    assert(dd == {"a": 2, "b": 3})

*/
