#from pybind11_tests import factory_constructors as m
#from pybind11_tests.factory_constructors import tag
#from pybind11_tests import ConstructorStats


def test_init_factory_basic():
    """Tests py::init_factory() wrapper around various ways of returning the object"""

    cstats = [ConstructorStats.get(c) for c in [m.TestFactory1, m.TestFactory2, m.TestFactory3]]
    cstats[0].alive()  # force gc
    n_inst = ConstructorStats.detail_reg_inst()

    x1 = m.TestFactory1(tag.unique_ptr, 3)
    assert x1.value == "3"
    y1 = m.TestFactory1(tag.pointer)
    assert y1.value == "(empty)"
    z1 = m.TestFactory1("hi!")
    assert z1.value == "hi!"

    assert ConstructorStats.detail_reg_inst() == n_inst + 3

    x2 = m.TestFactory2(tag.move)
    assert x2.value == "(empty2)"
    y2 = m.TestFactory2(tag.pointer, 7)
    assert y2.value == "7"
    z2 = m.TestFactory2(tag.unique_ptr, "hi again")
    assert z2.value == "hi again"

    assert ConstructorStats.detail_reg_inst() == n_inst + 6

    x3 = m.TestFactory3(tag.shared_ptr)
    assert x3.value == "(empty3)"
    y3 = m.TestFactory3(tag.pointer, 42)
    assert y3.value == "42"
    z3 = m.TestFactory3("bye")
    assert z3.value == "bye"

    #with pytest.raises(TypeError) as excinfo:
    #    m.TestFactory3(tag.null_ptr)
    #assert str(excinfo.value) == "pybind11::init(): factory function returned nullptr"

    #assert [i.alive() for i in cstats] == [3, 3, 3]
    #assert ConstructorStats.detail_reg_inst() == n_inst + 9

    #del x1, y2, y3, z3
    #assert [i.alive() for i in cstats] == [2, 2, 1]
    #assert ConstructorStats.detail_reg_inst() == n_inst + 5
    #del x2, x3, y1, z1, z2
    #assert [i.alive() for i in cstats] == [0, 0, 0]
    #assert ConstructorStats.detail_reg_inst() == n_inst

    #assert [i.values() for i in cstats] == [
    #    ["3", "hi!"],
    #    ["7", "hi again"],
    #    ["42", "bye"]
    #]
    #assert [i.default_constructions for i in cstats] == [1, 1, 1]

test_init_factory_basic()

