/*
    tests/test_sequences_and_iterators.cpp -- supporting Pythons' sequence protocol, iterators,
    etc.

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/
#include "gtest/gtest.h"
#include "dmlc/logging.h"
#include <pybind11/embed.h>
#include <tuple>
#include <list>
#include <vector>
#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <cstdlib>


template<typename T>
class NonZeroIterator {
    const T* ptr_;
public:
    NonZeroIterator(const T* ptr) : ptr_(ptr) {}
    const T& operator*() const { return *ptr_; }
    NonZeroIterator& operator++() { ++ptr_; return *this; }
};

class NonZeroSentinel {};

template<typename A, typename B>
bool operator==(const NonZeroIterator<std::pair<A, B>>& it, const NonZeroSentinel&) {
    return !(*it).first || !(*it).second;
}

template <typename PythonType>
py::list test_random_access_iterator(PythonType x) {
    if (x.size() < 5)
        throw py::value_error("Please provide at least 5 elements for testing.");

    auto checks = py::list();
    auto assert_equal = [&checks](py::handle a, py::handle b) {
        auto result = PyObject_RichCompareBool(a.ptr(), b.ptr(), Py_EQ);
        if (result == -1) { throw py::error_already_set(); }
        checks.append(result != 0);
    };

    auto it = x.begin();
    assert_equal(x[0], *it);
    assert_equal(x[0], it[0]);
    assert_equal(x[1], it[1]);

    assert_equal(x[1], *(++it));
    assert_equal(x[1], *(it++));
    assert_equal(x[2], *it);
    assert_equal(x[3], *(it += 1));
    assert_equal(x[2], *(--it));
    assert_equal(x[2], *(it--));
    assert_equal(x[1], *it);
    assert_equal(x[0], *(it -= 1));

    assert_equal(it->attr("real"), x[0].attr("real"));
    assert_equal((it + 1)->attr("real"), x[1].attr("real"));

    assert_equal(x[1], *(it + 1));
    assert_equal(x[1], *(1 + it));
    it += 3;
    assert_equal(x[1], *(it - 2));

    checks.append(static_cast<std::size_t>(x.end() - x.begin()) == x.size());
    checks.append((x.begin() + static_cast<std::ptrdiff_t>(x.size())) == x.end());
    checks.append(x.begin() < x.end());

    return checks;
}
    class Sliceable{
    public:
      Sliceable(int n): size(n) {}
      int start,stop,step;
      int size;
    };

    // test_sequence
    class Sequence {
    public:
        Sequence(size_t size) : m_size(size) {
            print_created(this, "of size", m_size);
            m_data = new float[size];
            memset(m_data, 0, sizeof(float) * size);
        }
        Sequence(const std::vector<float> &value) : m_size(value.size()) {
            print_created(this, "of size", m_size, "from std::vector");
            m_data = new float[m_size];
            memcpy(m_data, &value[0], sizeof(float) * m_size);
        }
        Sequence(const Sequence &s) : m_size(s.m_size) {
            print_copy_created(this);
            m_data = new float[m_size];
            memcpy(m_data, s.m_data, sizeof(float)*m_size);
        }
        Sequence(Sequence &&s) : m_size(s.m_size), m_data(s.m_data) {
            print_move_created(this);
            s.m_size = 0;
            s.m_data = nullptr;
        }

        ~Sequence() { print_destroyed(this); delete[] m_data; }

        Sequence &operator=(const Sequence &s) {
            if (&s != this) {
                delete[] m_data;
                m_size = s.m_size;
                m_data = new float[m_size];
                memcpy(m_data, s.m_data, sizeof(float)*m_size);
            }
            print_copy_assigned(this);
            return *this;
        }

        Sequence &operator=(Sequence &&s) {
            if (&s != this) {
                delete[] m_data;
                m_size = s.m_size;
                m_data = s.m_data;
                s.m_size = 0;
                s.m_data = nullptr;
            }
            print_move_assigned(this);
            return *this;
        }

        bool operator==(const Sequence &s) const {
            if (m_size != s.size()) return false;
            for (size_t i = 0; i < m_size; ++i)
                if (m_data[i] != s[i])
                    return false;
            return true;
        }
        bool operator!=(const Sequence &s) const { return !operator==(s); }

        float operator[](size_t index) const { return m_data[index]; }
        float &operator[](size_t index) { return m_data[index]; }

        bool contains(float v) const {
            for (size_t i = 0; i < m_size; ++i)
                if (v == m_data[i])
                    return true;
            return false;
        }

        Sequence reversed() const {
            Sequence result(m_size);
            for (size_t i = 0; i < m_size; ++i)
                result[m_size - i - 1] = m_data[i];
            return result;
        }

        size_t size() const { return m_size; }

        const float *begin() const { return m_data; }
        const float *end() const { return m_data+m_size; }

    private:
        size_t m_size;
        float *m_data;
    };

    // test_map_iterator
    // Interface of a map-like object that isn't (directly) an unordered_map, but provides some basic
    // map-like functionality.
    class StringMap {
    public:
        StringMap() = default;
        StringMap(std::unordered_map<std::string, std::string> init)
            : map(std::move(init)) {}

        void set(std::string key, std::string val) { map[key] = val; }
        std::string get(std::string key) const { return map.at(key); }
        size_t size() const { return map.size(); }
    private:
        std::unordered_map<std::string, std::string> map;
    public:
        decltype(map.cbegin()) begin() const { return map.cbegin(); }
        decltype(map.cend()) end() const { return map.cend(); }
    };

PYBIND11_EMBEDDED_MODULE(sequences_and_iterators, m) {
    // test_sliceable
    py::class_<Sliceable>(m,"Sliceable")
        .def(py::init<int>())
        .def("__getitem__",[](const Sliceable &s, py::slice slice) {
          ssize_t start, stop, step, slicelength;
          if (!slice.compute(s.size, &start, &stop, &step, &slicelength))
              throw py::error_already_set();
          int istart = static_cast<int>(start);
          int istop =  static_cast<int>(stop);
          int istep =  static_cast<int>(step);
          return std::make_tuple(istart,istop,istep);
        })
        ;

    py::class_<Sequence>(m, "Sequence")
        .def(py::init<size_t>())
        .def(py::init<const std::vector<float>&>())
        /// Bare bones interface
        .def("__getitem__", [](const Sequence &s, size_t i) {
            if (i >= s.size()) throw py::index_error();
            return s[i];
        })
        .def("__setitem__", [](Sequence &s, size_t i, float v) {
            if (i >= s.size()) throw py::index_error();
            s[i] = v;
        })
        .def("__len__", &Sequence::size)
        /// Optional sequence protocol operations
        .def("__iter__", [](const Sequence &s) { return py::make_iterator(s.begin(), s.end()); },
                         py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */)
        .def("__contains__", [](const Sequence &s, float v) { return s.contains(v); })
        .def("__reversed__", [](const Sequence &s) -> Sequence { return s.reversed(); })
        /// Slicing protocol (optional)
        .def("__getitem__", [](const Sequence &s, py::slice slice) -> Sequence* {
            size_t start, stop, step, slicelength;
            if (!slice.compute(s.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            Sequence *seq = new Sequence(slicelength);
            for (size_t i = 0; i < slicelength; ++i) {
                (*seq)[i] = s[start]; start += step;
            }
            return seq;
        })
        .def("__setitem__", [](Sequence &s, py::slice slice, const Sequence &value) {
            size_t start, stop, step, slicelength;
            if (!slice.compute(s.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            if (slicelength != value.size())
                throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");
            for (size_t i = 0; i < slicelength; ++i) {
                s[start] = value[i]; start += step;
            }
        })
        /// Comparisons
        .def(py::self == py::self)
        .def(py::self != py::self)
        // Could also define py::self + py::self for concatenation, etc.
        ;

    py::class_<StringMap>(m, "StringMap")
        .def(py::init<>())
        .def(py::init<std::unordered_map<std::string, std::string>>())
        .def("__getitem__", [](const StringMap &map, std::string key) {
                try { return map.get(key); }
                catch (const std::out_of_range&) {
                    throw py::key_error("key '" + key + "' does not exist");
                }
        })
        .def("__setitem__", &StringMap::set)
        .def("__len__", &StringMap::size)
        .def("__iter__", [](const StringMap &map) { return py::make_key_iterator(map.begin(), map.end()); },
                py::keep_alive<0, 1>())
        .def("items", [](const StringMap &map) { return py::make_iterator(map.begin(), map.end()); },
                py::keep_alive<0, 1>())
        ;

    // test_generalized_iterators
    class IntPairs {
    public:
        IntPairs(std::vector<std::pair<int, int>> data) : data_(std::move(data)) {}
        const std::pair<int, int>* begin() const { return data_.data(); }
    private:
        std::vector<std::pair<int, int>> data_;
    };
    py::class_<IntPairs>(m, "IntPairs")
        .def(py::init<std::vector<std::pair<int, int>>>())
        .def("nonzero", [](const IntPairs& s) {
                return py::make_iterator(NonZeroIterator<std::pair<int, int>>(s.begin()), NonZeroSentinel());
        }, py::keep_alive<0, 1>())
        .def("nonzero_keys", [](const IntPairs& s) {
            return py::make_key_iterator(NonZeroIterator<std::pair<int, int>>(s.begin()), NonZeroSentinel());
        }, py::keep_alive<0, 1>())
        ;


#if 0
    // Obsolete: special data structure for exposing custom iterator types to python
    // kept here for illustrative purposes because there might be some use cases which
    // are not covered by the much simpler py::make_iterator

    struct PySequenceIterator {
        PySequenceIterator(const Sequence &seq, py::object ref) : seq(seq), ref(ref) { }

        float next() {
            if (index == seq.size())
                throw py::stop_iteration();
            return seq[index++];
        }

        const Sequence &seq;
        py::object ref; // keep a reference
        size_t index = 0;
    };

    py::class_<PySequenceIterator>(seq, "Iterator")
        .def("__iter__", [](PySequenceIterator &it) -> PySequenceIterator& { return it; })
        .def("__next__", &PySequenceIterator::next);

    On the actual Sequence object, the iterator would be constructed as follows:
    .def("__iter__", [](py::object s) { return PySequenceIterator(s.cast<const Sequence &>(), s); })
#endif

    // test_python_iterator_in_cpp
    m.def("object_to_list", [](py::object o) {
        auto l = py::list();
        for (auto item : o) {
            l.append(item);
        }
        return l;
    });

    m.def("iterator_to_list", [](py::iterator it) {
        auto l = py::list();
        while (it != py::iterator::sentinel()) {
            l.append(*it);
            ++it;
        }
        return l;
    });

    // Make sure that py::iterator works with std algorithms
    m.def("count_none", [](py::object o) {
        return std::count_if(o.begin(), o.end(), [](py::handle h) { return h.is_none(); });
    });

    m.def("find_none", [](py::object o) {
        auto it = std::find_if(o.begin(), o.end(), [](py::handle h) { return h.is_none(); });
        return it->is_none();
    });

    m.def("count_nonzeros", [](py::dict d) {
       return std::count_if(d.begin(), d.end(), [](std::pair<py::handle, py::handle> p) {
           return p.second.cast<int>() != 0;
       });
    });

    m.def("tuple_iterator", &test_random_access_iterator<py::tuple>);
    m.def("list_iterator", &test_random_access_iterator<py::list>);
    m.def("sequence_iterator", &test_random_access_iterator<py::sequence>);

    // test_iterator_passthrough
    // #181: iterator passthrough did not compile
    m.def("iterator_passthrough", [](py::iterator s) -> py::iterator {
        return py::make_iterator(std::begin(s), std::end(s));
    });

    // test_iterator_rvp
    // #388: Can't make iterators via make_iterator() with different r/v policies
    static std::vector<int> list = { 1, 2, 3 };
    m.def("make_iterator_1", []() { return py::make_iterator<py::return_value_policy::copy>(list); });
    m.def("make_iterator_2", []() { return py::make_iterator<py::return_value_policy::automatic>(list); });
}

TEST(sequences_and_iterators, generalized_iterators)
{
    py::scoped_interpreter guard{};
    auto m = py::module::import("sequences_and_iterators");
    std::vector<std::pair<int, int>> a {{1,2}, {3, 4}, {0, 5}};
    CHECK_EQ(py::str(m.attr("iterator_to_list")(
                    m.attr("IntPairs")(a).attr("nonzero")()
                )
            ).cast<std::string>(), "[(1, 2), (3, 4)]");

    CHECK_EQ(py::str(m.attr("iterator_to_list")(
                    m.attr("IntPairs")(a).attr("nonzero_keys")()
                )
            ).cast<std::string>(), "[1, 3]");

    std::vector<std::pair<int, int>> b {{0,0}};
    auto it = m.attr("IntPairs")(b).attr("nonzero")();
    for(auto i : {1, 2, 3})
    {
        it.attr("__next__");
    }
    it = m.attr("IntPairs")(b).attr("nonzero_keys")();
    for(auto i : {1, 2, 3})
    {
        it.attr("__next__");
    }
}

TEST(sequences_and_iterators, sliceable)
{
    py::scoped_interpreter guard{};
    auto m = py::module::import("sequences_and_iterators");
    auto sliceable = m.attr("Sliceable")(100);
    CHECK_EQ(py::str(sliceable[py::slice(0,9999,1)]).cast<std::string>(), "(0, 100, 1)");
    CHECK_EQ(py::str(sliceable[py::slice(10,9999,1)]).cast<std::string>(), "(10, 100, 1)");
    CHECK_EQ(py::str(sliceable[py::slice(0,10,1)]).cast<std::string>(), "(0, 10, 1)");
    CHECK_EQ(py::str(sliceable[py::slice(0,9999,10)]).cast<std::string>(), "(0, 100, 10)");
    CHECK_EQ(py::str(sliceable[py::slice(-10,9999,1)]).cast<std::string>(), "(90, 100, 1)");
    CHECK_EQ(py::str(sliceable[py::slice(0,-10,1)]).cast<std::string>(), "(0, 90, 1)");
    // CHECK_EQ(py::str(sliceable[py::slice(0,9999,-10)]).cast<std::string>(), "(99, -1, -10)");
    CHECK_EQ(py::str(sliceable[py::slice(50,60,1)]).cast<std::string>(), "(50, 60, 1)");
    CHECK_EQ(py::str(sliceable[py::slice(50,60,-1)]).cast<std::string>(), "(50, 60, -1)");
}

auto isclose = [](float a, float b, float rel_tol = 1e-05, float abs_tol=0.0) -> bool {
    auto left = std::abs(a - b);
    if (left <= std::max(rel_tol * std::max(std::abs(a), std::abs(b)), abs_tol)) {
        return true;
    } else {
        return false;
    }
};

template <typename PythonType1, typename PythonType2>
bool allclose(PythonType1 a, PythonType2 b, float rel_tol = 1e-05, float abs_tol=0.0) {
    return std::equal(a.begin(), a.end(), b.begin(),
            [](py::handle a, py::handle b) {
                return isclose(a.cast<float>(), b.cast<float>());
            });
};

TEST(sequences_and_iterators, sequence)
{
    py::scoped_interpreter guard{};
    // auto __main__ = py::module::import("__main__");
    // auto global = py::dict(__main__.attr("__dict__"));
    auto m = py::module::import("sequences_and_iterators");
    auto pySequence = m.attr("Sequence");

    auto pybind11_tests_module = py::module::import("pybind11_tests");
    auto ConstructorStats = pybind11_tests_module.attr("ConstructorStats");
    auto cstats = ConstructorStats.attr("get")(pySequence);
    auto cstats_c = ConstructorStats::get<Sequence>();


    auto s = pySequence(5);
    Sequence *sc = s.cast<Sequence *>();
    // FIXMEM what cstats is incorrect
    py::print(cstats.attr("values")());
    py::print(cstats_c.values());
    // CHECK_EQ(py::str(cstats_c.values()).cast<std::string>(), R"(['of size', '5'])");

    // py::print(s.attr("__repr__")());
    CHECK_EQ(s.attr("__len__")().cast<int>() , 5);
    CHECK_EQ(sc->size() , 5);

    CHECK_EQ(s.attr("__getitem__")(0).cast<float>() , 0);
    CHECK_EQ((*sc)[0] , 0.0);

    CHECK_EQ(s.attr("__getitem__")(3).cast<float>() , 0);
    CHECK_EQ((*sc)[3] , 0.0);

    CHECK_EQ(s.attr("__contains__")(12.34).cast<bool>() , 0);
    CHECK_EQ(sc->contains(12.34), false);

    s.attr("__setitem__")(0, 12.34);
    s.attr("__setitem__")(3, 56.78);
    CHECK_EQ(isclose((*sc)[0], 12.34), true);
    CHECK_EQ(isclose((*sc)[3], 56.78), true);

    auto rev = m.attr("iterator_to_list")(s.attr("__reversed__")().attr("__iter__")());
    py::list expected;
    expected.append(0);
    expected.append(56.78);
    expected.append(0);
    expected.append(0);
    expected.append(12.34);
    CHECK_EQ(allclose(rev, expected), true);

    // CHECK_EQ(py::str(cstats.attr("values")()).cast<std::string>(), R"(['of size', '5'])");
}


TEST(sequences_and_iterators, map_iterator)
{
    py::scoped_interpreter guard{};
    auto m = py::module::import("sequences_and_iterators");
    std::unordered_map<std::string, std::string> a {{"hi", "bye"}, {"black", "white"}};
    auto sm = m.attr("StringMap")(a);
    CHECK_EQ(sm.attr("__getitem__")("hi").cast<std::string>(), "bye");
    StringMap *c_sm = sm.cast<StringMap*>();
}

TEST(sequences_and_iterators, iterator_in_cpp)
{
    py::scoped_interpreter guard{};
    auto m = py::module::import("sequences_and_iterators");
    auto local = py::dict();
    auto result = py::eval<py::eval_single_statement>("t = (1, 2, 3)", py::dict(), local);
    auto t = local["t"];
    CHECK_EQ(py::isinstance<py::sequence>(t), true);
    CHECK_EQ(py::str(m.attr("object_to_list")(t)).cast<std::string>(), R"([1, 2, 3])");
    CHECK_EQ(py::str(m.attr("object_to_list")(t.begin())).cast<std::string>(), R"([1, 2, 3])");
    CHECK_EQ(py::str(m.attr("iterator_to_list")(t.begin())).cast<std::string>(), R"([1, 2, 3])");
    // py::sequence s(py::int_(1), 2, 3);
    // py::sequence s(py::int_(3));
    // py::print(s.size());
    // m.attr("Sequence")({})
    // s.append(1);
}

TEST(sequences_and_iterators, iterator_rvp)
{
    py::scoped_interpreter guard{};
    auto m = py::module::import("sequences_and_iterators");
    std::vector<int> l {1, 2, 3};
    py::print(m.attr("iterator_to_list")(py::make_iterator(l)));
}
