#include "gtest/gtest.h"
#include "dmlc/logging.h"
#include "pybind11_tests.h"
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
template <typename T>
class TD;


PYBIND11_EMBEDDED_MODULE(test_embed, m)
{
    m.def("add", [](int i, int j) { return i + j; });
    m.def("__repr__", []() { return "this is a __repr__"; });
}
/*
// copy from test_numpy_dtypes.cpp
template <typename S>
py::list print_recarray(py::array_t<S, 0> arr)
{
    const auto req = arr.request();
    const auto ptr = static_cast<S*>(req.ptr);
    auto l = py::list();
    for (ssize_t i = 0; i < req.size; i++) {
        std::stringstream ss;
        ss << ptr[i];
        l.append(py::str(ss.str()));
    }
    return l;
}

// copy from test_numpy_array.cpp
using arr = py::array;
using arr_t = py::array_t<uint16_t, 0>;

template <typename... Ix>
arr data(const arr& a, Ix... index)
{
    return arr(a.nbytes() - a.offset_at(index...), (const uint8_t*)a.data(index...));
}

template <typename... Ix>
arr data_t(const arr_t& a, Ix... index)
{
    return arr(a.size() - a.index_at(index...), a.data(index...));
}

template <typename... Ix>
arr& mutate_data(arr& a, Ix... index)
{
    auto ptr = (uint8_t*)a.mutable_data(index...);
    for (ssize_t i = 0; i < a.nbytes() - a.offset_at(index...); i++)
        ptr[i] = (uint8_t)(ptr[i] * 2);
    return a;
}

template <typename... Ix>
arr_t& mutate_data_t(arr_t& a, Ix... index)
{
    auto ptr = a.mutable_data(index...);
    for (ssize_t i = 0; i < a.size() - a.index_at(index...); i++)
        ptr[i]++;
    return a;
}

template <typename... Ix>
ssize_t index_at(const arr& a, Ix... idx) { return a.index_at(idx...); }
template <typename... Ix>
ssize_t index_at_t(const arr_t& a, Ix... idx) { return a.index_at(idx...); }
template <typename... Ix>
ssize_t offset_at(const arr& a, Ix... idx) { return a.offset_at(idx...); }
template <typename... Ix>
ssize_t offset_at_t(const arr_t& a, Ix... idx) { return a.offset_at(idx...); }
template <typename... Ix>
ssize_t at_t(const arr_t& a, Ix... idx) { return a.at(idx...); }
template <typename... Ix>
arr_t& mutate_at_t(arr_t& a, Ix... idx)
{
    a.mutable_at(idx...)++;
    return a;
}

#define def_index_fn(name, type)                                      \
    sm.def(#name, [](type a) { return name(a); });                     \
    sm.def(#name, [](type a, int i) { return name(a, i); });           \
    sm.def(#name, [](type a, int i, int j) { return name(a, i, j); }); \
    sm.def(#name, [](type a, int i, int j, int k) { return name(a, i, j, k); });

template <typename T, typename T2>
py::handle auxiliaries(T&& r, T2&& r2)
{
    if (r.ndim() != 2)
        throw std::domain_error("error: ndim != 2");
    py::list l;
    l.append(*r.data(0, 0));
    l.append(*r2.mutable_data(0, 0));
    l.append(r.data(0, 1) == r2.mutable_data(0, 1));
    l.append(r.ndim());
    l.append(r.itemsize());
    l.append(r.shape(0));
    l.append(r.shape(1));
    l.append(r.size());
    l.append(r.nbytes());
    return l.release();
}

double my_func(int x, float y, double z)
{
    py::print("my_func(x:int={}, y:float={:.0f}, z:float={:.0f})"_s.format(x, y, z));
    return (float)x * y * z;
}


void test_eval()
{
    // py::module::import("sys").attr("argv") = py::make_tuple("test.py", "embed.cpp");
    // py::eval_file(test_py_file, py::globals());

    py::scoped_ostream_redirect redir(std::cout, py::module::import("sys").attr("stdout"));
    std::cout << "Hello py::sys::stdout" << std::flush;

    // Copy from test_eval.cpp
    auto global = py::dict(py::module::import("__main__").attr("__dict__"));

    auto test_eval_statements = [global]() {
        auto local = py::dict();
        local["call_test"] = py::cpp_function([&]() -> int {
            return 42;
        });
        // Regular string literal
        py::exec(
            "message = 'Hello World!'\n"
            "x = call_test()",
            global, local);

        // Multi-line raw string literal
        py::exec(R"(
            if x == 42:
                print(message)
            else:
                raise RuntimeError
            )",
            global, local);
        auto x = local["x"].cast<int>();

        return x == 42;
    };

    std::cout << "test_eval_statements result:" << test_eval_statements() << std::endl;

    auto test_eval_single_statements = []() {
        auto local = py::dict();
        local["call_test"] = py::cpp_function([&]() -> int {
            return 42;
        });

        auto result = py::eval<py::eval_single_statement>("x = call_test()", py::dict(), local);
        auto x = local["x"].cast<int>();
        return result.is_none() && x == 42;
    };
    std::cout << "test_eval_single_statements result:" << test_eval_single_statements() << std::endl;

}

PYBIND11_EMBEDDED_MODULE(embed_numpy_array, sm)
{
    try {
        py::module::import("numpy");
    } catch (...) {
        std::cout << "fail import numpy" << std::endl;
        exit(0);
    }


    // Below COpy from test_numpy_array.cpp
    //
    // test_array_attributes
    sm.def("ndim", [](const arr& a) { return a.ndim(); });
    sm.def("shape", [](const arr& a) { return arr(a.ndim(), a.shape()); });
    sm.def("shape", [](const arr& a, ssize_t dim) { return a.shape(dim); });
    sm.def("strides", [](const arr& a) { return arr(a.ndim(), a.strides()); });
    sm.def("strides", [](const arr& a, ssize_t dim) { return a.strides(dim); });
    sm.def("writeable", [](const arr& a) { return a.writeable(); });
    sm.def("size", [](const arr& a) { return a.size(); });
    sm.def("itemsize", [](const arr& a) { return a.itemsize(); });
    sm.def("nbytes", [](const arr& a) { return a.nbytes(); });
    sm.def("owndata", [](const arr& a) { return a.owndata(); });


    // test_index_offset
    def_index_fn(index_at, const arr&);
    def_index_fn(index_at_t, const arr_t&);
    def_index_fn(offset_at, const arr&);
    def_index_fn(offset_at_t, const arr_t&);
    // test_data
    def_index_fn(data, const arr&);
    def_index_fn(data_t, const arr_t&);
    // test_mutate_data, test_mutate_readonly
    def_index_fn(mutate_data, arr&);
    def_index_fn(mutate_data_t, arr_t&);
    def_index_fn(at_t, const arr_t&);
    def_index_fn(mutate_at_t, arr_t&);

    // test_make_c_f_array
    sm.def("make_f_array", [] { return py::array_t<float>({ 2, 2 }, { 4, 8 }); });
    sm.def("make_c_array", [] { return py::array_t<float>({ 2, 2 }, { 8, 4 }); });

    // test_wrap
    sm.def("wrap", [](py::array a) {
        return py::array(
            a.dtype(),
            {a.shape(), a.shape() + a.ndim()},
            {a.strides(), a.strides() + a.ndim()},
            a.data(),
            a
        );
    });

    // test_numpy_view
    struct ArrayClass {
        int data[2] = { 1, 2 };
        ArrayClass() { py::print("ArrayClass()"); }
        ~ArrayClass() { py::print("~ArrayClass()"); }
    };
    py::class_<ArrayClass>(sm, "ArrayClass")
        .def(py::init<>())
        .def("numpy_view", [](py::object &obj) {
            py::print("ArrayClass::numpy_view()");
            ArrayClass &a = obj.cast<ArrayClass&>();
            return py::array_t<int>({2}, {4}, a.data, obj);
        }
    );

    // test_cast_numpy_int64_to_uint64
    sm.def("function_taking_uint64", [](uint64_t) { });

    // test_isinstance
    sm.def("isinstance_untyped", [](py::object yes, py::object no) {
        return py::isinstance<py::array>(yes) && !py::isinstance<py::array>(no);
    });
    sm.def("isinstance_typed", [](py::object o) {
        return py::isinstance<py::array_t<double>>(o) && !py::isinstance<py::array_t<int>>(o);
    });

    // test_constructors
    sm.def("default_constructors", []() {
        return py::dict(
            "array"_a=py::array(),
            "array_t<int32>"_a=py::array_t<std::int32_t>(),
            "array_t<double>"_a=py::array_t<double>()
        );
    });
    sm.def("converting_constructors", [](py::object o) {
        return py::dict(
            "array"_a=py::array(o),
            "array_t<int32>"_a=py::array_t<std::int32_t>(o),
            "array_t<double>"_a=py::array_t<double>(o)
        );
    });

    // test_overload_resolution
    sm.def("overloaded", [](py::array_t<double>) { return "double"; });
    sm.def("overloaded", [](py::array_t<float>) { return "float"; });
    sm.def("overloaded", [](py::array_t<int>) { return "int"; });
    sm.def("overloaded", [](py::array_t<unsigned short>) { return "unsigned short"; });
    sm.def("overloaded", [](py::array_t<long long>) { return "long long"; });
    sm.def("overloaded", [](py::array_t<std::complex<double>>) { return "double complex"; });
    sm.def("overloaded", [](py::array_t<std::complex<float>>) { return "float complex"; });

    sm.def("overloaded2", [](py::array_t<std::complex<double>>) { return "double complex"; });
    sm.def("overloaded2", [](py::array_t<double>) { return "double"; });
    sm.def("overloaded2", [](py::array_t<std::complex<float>>) { return "float complex"; });
    sm.def("overloaded2", [](py::array_t<float>) { return "float"; });

    // Only accept the exact types:
    sm.def("overloaded3", [](py::array_t<int>) { return "int"; }, py::arg().noconvert());
    sm.def("overloaded3", [](py::array_t<double>) { return "double"; }, py::arg().noconvert());

    // Make sure we don't do unsafe coercion (e.g. float to int) when not using forcecast, but
    // rather that float gets converted via the safe (conversion to double) overload:
    sm.def("overloaded4", [](py::array_t<long long, 0>) { return "long long"; });
    sm.def("overloaded4", [](py::array_t<double, 0>) { return "double"; });

    // But we do allow conversion to int if forcecast is enabled (but only if no overload matches
    // without conversion)
    sm.def("overloaded5", [](py::array_t<unsigned int>) { return "unsigned int"; });
    sm.def("overloaded5", [](py::array_t<double>) { return "double"; });

    // test_greedy_string_overload
    // Issue 685: ndarray shouldn't go to std::string overload
    sm.def("issue685", [](std::string) { return "string"; });
    sm.def("issue685", [](py::array) { return "array"; });
    sm.def("issue685", [](py::object) { return "other"; });

    // test_array_unchecked_fixed_dims
    sm.def("proxy_add2", [](py::array_t<double> a, double v) {
        auto r = a.mutable_unchecked<2>();
        for (ssize_t i = 0; i < r.shape(0); i++)
            for (ssize_t j = 0; j < r.shape(1); j++)
                r(i, j) += v;
    }, py::arg().noconvert(), py::arg());

    sm.def("proxy_init3", [](double start) {
        py::array_t<double, py::array::c_style> a({ 3, 3, 3 });
        auto r = a.mutable_unchecked<3>();
        for (ssize_t i = 0; i < r.shape(0); i++)
        for (ssize_t j = 0; j < r.shape(1); j++)
        for (ssize_t k = 0; k < r.shape(2); k++)
            r(i, j, k) = start++;
        return a;
    });
    sm.def("proxy_init3F", [](double start) {
        py::array_t<double, py::array::f_style> a({ 3, 3, 3 });
        auto r = a.mutable_unchecked<3>();
        for (ssize_t k = 0; k < r.shape(2); k++)
        for (ssize_t j = 0; j < r.shape(1); j++)
        for (ssize_t i = 0; i < r.shape(0); i++)
            r(i, j, k) = start++;
        return a;
    });
    sm.def("proxy_squared_L2_norm", [](py::array_t<double> a) {
        auto r = a.unchecked<1>();
        double sumsq = 0;
        for (ssize_t i = 0; i < r.shape(0); i++)
            sumsq += r[i] * r(i); // Either notation works for a 1D array
        return sumsq;
    });

    sm.def("proxy_auxiliaries2", [](py::array_t<double> a) {
        auto r = a.unchecked<2>();
        auto r2 = a.mutable_unchecked<2>();
        return auxiliaries(r, r2);
    });

    // test_array_unchecked_dyn_dims
    // Same as the above, but without a compile-time dimensions specification:
    sm.def("proxy_add2_dyn", [](py::array_t<double> a, double v) {
        auto r = a.mutable_unchecked();
        if (r.ndim() != 2) throw std::domain_error("error: ndim != 2");
        for (ssize_t i = 0; i < r.shape(0); i++)
            for (ssize_t j = 0; j < r.shape(1); j++)
                r(i, j) += v;
    }, py::arg().noconvert(), py::arg());
    sm.def("proxy_init3_dyn", [](double start) {
        py::array_t<double, py::array::c_style> a({ 3, 3, 3 });
        auto r = a.mutable_unchecked();
        if (r.ndim() != 3) throw std::domain_error("error: ndim != 3");
        for (ssize_t i = 0; i < r.shape(0); i++)
        for (ssize_t j = 0; j < r.shape(1); j++)
        for (ssize_t k = 0; k < r.shape(2); k++)
            r(i, j, k) = start++;
        return a;
    });
    sm.def("proxy_auxiliaries2_dyn", [](py::array_t<double> a) {
        return auxiliaries(a.unchecked(), a.mutable_unchecked());
    });

    sm.def("array_auxiliaries2", [](py::array_t<double> a) {
        return auxiliaries(a, a);
    });

    // test_array_failures
    // Issue #785: Uninformative "Unknown internal error" exception when constructing array from empty object:
    sm.def("array_fail_test", []() { return py::array(py::object()); });
    sm.def("array_t_fail_test", []() { return py::array_t<double>(py::object()); });
    // Make sure the error from numpy is being passed through:
    sm.def("array_fail_test_negative_size", []() { int c = 0; return py::array(-1, &c); });

    // test_initializer_list
    // Issue (unnumbered; reported in #788): regression: initializer lists can be ambiguous
    sm.def("array_initializer_list1", []() { return py::array_t<float>(1); }); // { 1 } also works, but clang warns about it
    sm.def("array_initializer_list2", []() { return py::array_t<float>({ 1, 2 }); });
    sm.def("array_initializer_list3", []() { return py::array_t<float>({ 1, 2, 3 }); });
    sm.def("array_initializer_list4", []() { return py::array_t<float>({ 1, 2, 3, 4 }); });

    // test_array_resize
    // reshape array to 2D without changing size
    sm.def("array_reshape2", [](py::array_t<double> a) {
        const ssize_t dim_sz = (ssize_t)std::sqrt(a.size());
        if (dim_sz * dim_sz != a.size())
            throw std::domain_error("array_reshape2: input array total size is not a squared integer");
        a.resize({dim_sz, dim_sz});
    });

    // resize to 3D array with each dimension = N
    sm.def("array_resize3", [](py::array_t<double> a, size_t N, bool refcheck) {
        a.resize({N, N, N}, refcheck);
    });

    // test_array_create_and_resize
    // return 2D array with Nrows = Ncols = N
    sm.def("create_and_resize", [](size_t N) {
        py::array_t<double> a;
        a.resize({N, N});
        std::fill(a.mutable_data(), a.mutable_data() + a.size(), 42.);
        return a;
    });
}


// below copied from test_numpy_dtypes.cpp
#ifdef __GNUC__
#define PYBIND11_PACKED(cls) cls __attribute__((__packed__))
#else
#define PYBIND11_PACKED(cls) __pragma(pack(push, 1)) cls __pragma(pack(pop))
#endif

namespace py = pybind11;

struct SimpleStruct {
    bool bool_;
    uint32_t uint_;
    float float_;
    long double ldbl_;
};

std::ostream& operator<<(std::ostream& os, const SimpleStruct& v) {
    return os << "s:" << v.bool_ << "," << v.uint_ << "," << v.float_ << "," << v.ldbl_;
}

PYBIND11_PACKED(struct PackedStruct {
    bool bool_;
    uint32_t uint_;
    float float_;
    long double ldbl_;
});

std::ostream& operator<<(std::ostream& os, const PackedStruct& v) {
    return os << "p:" << v.bool_ << "," << v.uint_ << "," << v.float_ << "," << v.ldbl_;
}

PYBIND11_PACKED(struct NestedStruct {
    SimpleStruct a;
    PackedStruct b;
});

std::ostream& operator<<(std::ostream& os, const NestedStruct& v) {
    return os << "n:a=" << v.a << ";b=" << v.b;
}

struct PartialStruct {
    bool bool_;
    uint32_t uint_;
    float float_;
    uint64_t dummy2;
    long double ldbl_;
};

struct PartialNestedStruct {
    uint64_t dummy1;
    PartialStruct a;
    uint64_t dummy2;
};

struct UnboundStruct { };

struct StringStruct {
    char a[3];
    std::array<char, 3> b;
};

struct ComplexStruct {
    std::complex<float> cflt;
    std::complex<double> cdbl;
};

std::ostream& operator<<(std::ostream& os, const ComplexStruct& v) {
    return os << "c:" << v.cflt << "," << v.cdbl;
}

struct ArrayStruct {
    char a[3][4];
    int32_t b[2];
    std::array<uint8_t, 3> c;
    std::array<float, 2> d[4];
};

PYBIND11_PACKED(struct StructWithUglyNames {
    int8_t __x__;
    uint64_t __y__;
});

enum class E1 : int64_t { A = -1, B = 1 };
enum E2 : uint8_t { X = 1, Y = 2 };

PYBIND11_PACKED(struct EnumStruct {
    E1 e1;
    E2 e2;
});

std::ostream& operator<<(std::ostream& os, const StringStruct& v) {
    os << "a='";
    for (size_t i = 0; i < 3 && v.a[i]; i++) os << v.a[i];
    os << "',b='";
    for (size_t i = 0; i < 3 && v.b[i]; i++) os << v.b[i];
    return os << "'";
}

std::ostream& operator<<(std::ostream& os, const ArrayStruct& v) {
    os << "a={";
    for (int i = 0; i < 3; i++) {
        if (i > 0)
            os << ',';
        os << '{';
        for (int j = 0; j < 3; j++)
            os << v.a[i][j] << ',';
        os << v.a[i][3] << '}';
    }
    os << "},b={" << v.b[0] << ',' << v.b[1];
    os << "},c={" << int(v.c[0]) << ',' << int(v.c[1]) << ',' << int(v.c[2]);
    os << "},d={";
    for (int i = 0; i < 4; i++) {
        if (i > 0)
            os << ',';
        os << '{' << v.d[i][0] << ',' << v.d[i][1] << '}';
    }
    return os << '}';
}

std::ostream& operator<<(std::ostream& os, const EnumStruct& v) {
    return os << "e1=" << (v.e1 == E1::A ? "A" : "B") << ",e2=" << (v.e2 == E2::X ? "X" : "Y");
}

template <typename T>
py::array mkarray_via_buffer(size_t n) {
    return py::array(py::buffer_info(nullptr, sizeof(T),
                                     py::format_descriptor<T>::format(),
                                     1, { n }, { sizeof(T) }));
}





PYBIND11_EMBEDDED_MODULE(embed_numpy_array, m)
{
    try { py::module::import("numpy");
    } catch (...) {
        std::cout << "fail import numpy" << std::endl;
        exit(0);
    }

    // typeinfo may be registered before the dtype descriptor for scalar casts to work...
    py::class_<SimpleStruct>(m, "SimpleStruct");

    PYBIND11_NUMPY_DTYPE(SimpleStruct, bool_, uint_, float_, ldbl_);
    PYBIND11_NUMPY_DTYPE(PackedStruct, bool_, uint_, float_, ldbl_);
    PYBIND11_NUMPY_DTYPE(NestedStruct, a, b);
    PYBIND11_NUMPY_DTYPE(PartialStruct, bool_, uint_, float_, ldbl_);
    PYBIND11_NUMPY_DTYPE(PartialNestedStruct, a);
    PYBIND11_NUMPY_DTYPE(StringStruct, a, b);
    PYBIND11_NUMPY_DTYPE(ArrayStruct, a, b, c, d);
    PYBIND11_NUMPY_DTYPE(EnumStruct, e1, e2);
    PYBIND11_NUMPY_DTYPE(ComplexStruct, cflt, cdbl);
    // ... or after
    py::class_<PackedStruct>(m, "PackedStruct");

    PYBIND11_NUMPY_DTYPE_EX(StructWithUglyNames, __x__, "x", __y__, "y");

    // If uncommented, this should produce a static_assert failure telling the user that the struct
    // is not a POD type
//    struct NotPOD { std::string v; NotPOD() : v("hi") {}; };
//    PYBIND11_NUMPY_DTYPE(NotPOD, v);

    // test_recarray, test_scalar_conversion
    m.def("create_rec_simple", &create_recarray<SimpleStruct>);
    m.def("create_rec_packed", &create_recarray<PackedStruct>);
    m.def("create_rec_nested", [](size_t n) { // test_signature
        py::array_t<NestedStruct, 0> arr = mkarray_via_buffer<NestedStruct>(n);
        auto req = arr.request();
        auto ptr = static_cast<NestedStruct*>(req.ptr);
        for (size_t i = 0; i < n; i++) {
            SET_TEST_VALS(ptr[i].a, i);
            SET_TEST_VALS(ptr[i].b, i + 1);
        }
        return arr;
    });
    m.def("create_rec_partial", &create_recarray<PartialStruct>);
    m.def("create_rec_partial_nested", [](size_t n) {
        py::array_t<PartialNestedStruct, 0> arr = mkarray_via_buffer<PartialNestedStruct>(n);
        auto req = arr.request();
        auto ptr = static_cast<PartialNestedStruct*>(req.ptr);
        for (size_t i = 0; i < n; i++) {
            SET_TEST_VALS(ptr[i].a, i);
        }
        return arr;
    });
    m.def("print_rec_simple", &print_recarray<SimpleStruct>);
    m.def("print_rec_packed", &print_recarray<PackedStruct>);
    m.def("print_rec_nested", &print_recarray<NestedStruct>);

    // test_format_descriptors
    m.def("get_format_unbound", []() { return py::format_descriptor<UnboundStruct>::format(); });
    m.def("print_format_descriptors", []() {
        py::list l;
        for (const auto &fmt : {
            py::format_descriptor<SimpleStruct>::format(),
            py::format_descriptor<PackedStruct>::format(),
            py::format_descriptor<NestedStruct>::format(),
            py::format_descriptor<PartialStruct>::format(),
            py::format_descriptor<PartialNestedStruct>::format(),
            py::format_descriptor<StringStruct>::format(),
            py::format_descriptor<ArrayStruct>::format(),
            py::format_descriptor<EnumStruct>::format(),
            py::format_descriptor<ComplexStruct>::format()
        }) {
            l.append(py::cast(fmt));
        }
        return l;
    });

    // test_dtype
    m.def("print_dtypes", []() {
        py::list l;
        for (const py::handle &d : {
            py::dtype::of<SimpleStruct>(),
            py::dtype::of<PackedStruct>(),
            py::dtype::of<NestedStruct>(),
            py::dtype::of<PartialStruct>(),
            py::dtype::of<PartialNestedStruct>(),
            py::dtype::of<StringStruct>(),
            py::dtype::of<ArrayStruct>(),
            py::dtype::of<EnumStruct>(),
            py::dtype::of<StructWithUglyNames>(),
            py::dtype::of<ComplexStruct>()
        })
            l.append(py::str(d));
        return l;
    });
    m.def("test_dtype_ctors", &test_dtype_ctors);
    m.def("test_dtype_methods", []() {
        py::list list;
        auto dt1 = py::dtype::of<int32_t>();
        auto dt2 = py::dtype::of<SimpleStruct>();
        list.append(dt1); list.append(dt2);
        list.append(py::bool_(dt1.has_fields())); list.append(py::bool_(dt2.has_fields()));
        list.append(py::int_(dt1.itemsize())); list.append(py::int_(dt2.itemsize()));
        return list;
    });
    struct TrailingPaddingStruct {
        int32_t a;
        char b;
    };
    PYBIND11_NUMPY_DTYPE(TrailingPaddingStruct, a, b);
    m.def("trailing_padding_dtype", []() { return py::dtype::of<TrailingPaddingStruct>(); });

    // test_string_array
    m.def("create_string_array", [](bool non_empty) {
        py::array_t<StringStruct, 0> arr = mkarray_via_buffer<StringStruct>(non_empty ? 4 : 0);
        if (non_empty) {
            auto req = arr.request();
            auto ptr = static_cast<StringStruct*>(req.ptr);
            for (ssize_t i = 0; i < req.size * req.itemsize; i++)
                static_cast<char*>(req.ptr)[i] = 0;
            ptr[1].a[0] = 'a'; ptr[1].b[0] = 'a';
            ptr[2].a[0] = 'a'; ptr[2].b[0] = 'a';
            ptr[3].a[0] = 'a'; ptr[3].b[0] = 'a';

            ptr[2].a[1] = 'b'; ptr[2].b[1] = 'b';
            ptr[3].a[1] = 'b'; ptr[3].b[1] = 'b';

            ptr[3].a[2] = 'c'; ptr[3].b[2] = 'c';
        }
        return arr;
    });
    m.def("print_string_array", &print_recarray<StringStruct>);

    // test_array_array
    m.def("create_array_array", [](size_t n) {
        py::array_t<ArrayStruct, 0> arr = mkarray_via_buffer<ArrayStruct>(n);
        auto ptr = (ArrayStruct *) arr.mutable_data();
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < 3; j++)
                for (size_t k = 0; k < 4; k++)
                    ptr[i].a[j][k] = char('A' + (i * 100 + j * 10 + k) % 26);
            for (size_t j = 0; j < 2; j++)
                ptr[i].b[j] = int32_t(i * 1000 + j);
            for (size_t j = 0; j < 3; j++)
                ptr[i].c[j] = uint8_t(i * 10 + j);
            for (size_t j = 0; j < 4; j++)
                for (size_t k = 0; k < 2; k++)
                    ptr[i].d[j][k] = float(i) * 100.0f + float(j) * 10.0f + float(k);
        }
        return arr;
    });
    m.def("print_array_array", &print_recarray<ArrayStruct>);

    // test_enum_array
    m.def("create_enum_array", [](size_t n) {
        py::array_t<EnumStruct, 0> arr = mkarray_via_buffer<EnumStruct>(n);
        auto ptr = (EnumStruct *) arr.mutable_data();
        for (size_t i = 0; i < n; i++) {
            ptr[i].e1 = static_cast<E1>(-1 + ((int) i % 2) * 2);
            ptr[i].e2 = static_cast<E2>(1 + (i % 2));
        }
        return arr;
    });
    m.def("print_enum_array", &print_recarray<EnumStruct>);

    // test_complex_array
    m.def("create_complex_array", [](size_t n) {
        py::array_t<ComplexStruct, 0> arr = mkarray_via_buffer<ComplexStruct>(n);
        auto ptr = (ComplexStruct *) arr.mutable_data();
        for (size_t i = 0; i < n; i++) {
            ptr[i].cflt.real(float(i));
            ptr[i].cflt.imag(float(i) + 0.25f);
            ptr[i].cdbl.real(double(i) + 0.5);
            ptr[i].cdbl.imag(double(i) + 0.75);
        }
        return arr;
    });
    m.def("print_complex_array", &print_recarray<ComplexStruct>);

    // test_array_constructors
    m.def("test_array_ctors", &test_array_ctors);

    // test_compare_buffer_info
    struct CompareStruct {
        bool x;
        uint32_t y;
        float z;
    };
    PYBIND11_NUMPY_DTYPE(CompareStruct, x, y, z);
    m.def("compare_buffer_info", []() {
        py::list list;
        list.append(py::bool_(py::detail::compare_buffer_info<float>::compare(py::buffer_info(nullptr, sizeof(float), "f", 1))));
        list.append(py::bool_(py::detail::compare_buffer_info<unsigned>::compare(py::buffer_info(nullptr, sizeof(int), "I", 1))));
        list.append(py::bool_(py::detail::compare_buffer_info<long>::compare(py::buffer_info(nullptr, sizeof(long), "l", 1))));
        list.append(py::bool_(py::detail::compare_buffer_info<long>::compare(py::buffer_info(nullptr, sizeof(long), sizeof(long) == sizeof(int) ? "i" : "q", 1))));
        list.append(py::bool_(py::detail::compare_buffer_info<CompareStruct>::compare(py::buffer_info(nullptr, sizeof(CompareStruct), "T{?:x:3xI:y:f:z:}", 1))));
        return list;
    });
    m.def("buffer_to_dtype", [](py::buffer& buf) { return py::dtype(buf.request()); });

    // test_scalar_conversion
    m.def("f_simple", [](SimpleStruct s) { return s.uint_ * 10; });
    m.def("f_packed", [](PackedStruct s) { return s.uint_ * 10; });
    m.def("f_nested", [](NestedStruct s) { return s.a.uint_ * 10; });

    // test_register_dtype
    m.def("register_dtype", []() { PYBIND11_NUMPY_DTYPE(SimpleStruct, bool_, uint_, float_, ldbl_); });


}

void setup_numpy_vectorize()
{
    ////////////////////////////////////////////////////////
    //    below copy from test_numpy_vectorize.cpp
    //
    // test_vectorize, test_docs, test_array_collapse
    // Vectorize all arguments of a function (though non-vector arguments are also allowed)
    auto vectorized_func = py::vectorize(my_func);

    // Vectorize a lambda function with a capture object (e.g. to exclude some arguments from the vectorization)
    auto vectorized_func2 =
        [](py::array_t<int> x, py::array_t<float> y, float z) {
            return py::vectorize([z](int x, float y) { return my_func(x, y, z); })(x, y);
        };

    // Vectorize a complex-valued function
    auto vectorized_func3 = py::vectorize(
        [](std::complex<double> c) { return c * std::complex<double>(2.f); });

    // test_type_selection
    // Numpy function which only accepts specific data types
    m.def("selective_func", [](py::array_t<int, py::array::c_style>) { return "Int branch taken."; });
    m.def("selective_func", [](py::array_t<float, py::array::c_style>) { return "Float branch taken."; });
    m.def("selective_func", [](py::array_t<std::complex<float>, py::array::c_style>) { return "Complex float branch taken."; });

    // test_passthrough_arguments
    // Passthrough test: references and non-pod types should be automatically passed through (in the
    // function definition below, only `b`, `d`, and `g` are vectorized):
    struct NonPODClass {
        NonPODClass(int v)
            : value { v }
        {
        }
        int value;
    };

    py::class_<NonPODClass>(m, "NonPODClass").def(py::init<int>());
    m.def("vec_passthrough", py::vectorize([](double* a, double b, py::array_t<double> c, const int& d, int& e, NonPODClass f, const double g) {
        return *a + b + c.at(0) + d + e + f.value + g;
    }));

    // test_method_vectorization
    struct VectorizeTestClass {
        VectorizeTestClass(int v)
            : value { v } {};
        float method(int x, float y) { return y + (float)(x + value); }
        int value = 0;
    };
    py::class_<VectorizeTestClass> vtc(m, "VectorizeTestClass");
    vtc.def(py::init<int>())
        .def_readwrite("value", &VectorizeTestClass::value);

    // Automatic vectorizing of methkds
    vtc.def("method", py::vectorize(&VectorizeTestClass::method));

    // test_trivial_broadcasting
    // Internal optimization test for whether the input is trivially broadcastable:
    py::enum_<py::detail::broadcast_trivial>(m, "trivial")
        .value("f_trivial", py::detail::broadcast_trivial::f_trivial)
        .value("c_trivial", py::detail::broadcast_trivial::c_trivial)
        .value("non_trivial", py::detail::broadcast_trivial::non_trivial);
    m.def("vectorized_is_trivial", [](py::array_t<int, py::array::forcecast> arg1, py::array_t<float, py::array::forcecast> arg2, py::array_t<double, py::array::forcecast> arg3) {
        ssize_t ndim;
        std::vector<ssize_t> shape;
        std::array<py::buffer_info, 3> buffers { { arg1.request(), arg2.request(), arg3.request() } };
        return py::detail::broadcast(buffers, ndim, shape);
    });
}

void test_numpy_array()
{

    auto m = py::module::import("embed_numpy_array");
    auto pytest = py::module::import("pytest");

    ////////////////// test_numpy_array.py
    // test_array_attributes
    {
        auto a = py::array_t<float, 8>(0);
        assert(m.attr("ndim")(a).cast<int>() == 1);
        // assert(m.attr("shape")(a).cast<py::array>() == arr(0,0));

        auto global = py::dict(pytest.attr("__dict__"));
        auto local = py::dict();
        local["call_test"] = py::cpp_function([&]() -> auto {
            return m.attr("shape")(a);
        });

        py::exec(R"(
            assert all(call_test() == [])
            )",
            global, local);

        assert(m.attr("writeable")(a));
        // assert(m.attr("writeable")(a).cast<bool>());
        std::cout << m.attr("size")(a).cast<int>() ;
        std::cout << m.attr("itemsize")(a).cast<int>() ;
        std::cout << m.attr("nbytes")(a).cast<int>() ;
        assert(m.attr("size")(a).cast<int>() == 0);
        assert(m.attr("itemsize")(a).cast<int>() == 4);
        assert(m.attr("nbytes")(a).cast<int>() == 8);
        assert(m.attr("owndata")(a));

        // auto a_shape = m.attr("shape")(a);
        // auto a_tem1 = std::array();
        // auto a_temp = arr();
        // auto a_shape.cast<std::array>() == {};
        // auto is_true = a_shape.is(a_temp);
        // assert(m.attr("shape")(a).is(arr()));
        // py::print("py_print {}"_s.format(std::type_info(m.attr("shape")(a))));
        std::cout << "test_array test done\n";
    }
}
*/

// int main(int argc, char* argv[])
// {
    // if (argc != 2)
    //     throw std::runtime_error("Expected test.py file as the first argument");
    // auto test_py_file = argv[1];
TEST(test_embed, test_embed)
{
    py::scoped_interpreter guard{};

    auto m = py::module::import("test_embed");
    auto o = m.attr("__repr__")();

    py::print(o);
    CHECK_EQ(m.attr("add")(1, 2).cast<int>(), 3);
}
    // test_eval();

    // test_numpy_array();
// }
TEST(test_factory_constructors, test_factory_constructors)
{
    py::scoped_interpreter guard{};
    auto m = py::module::import("__main__");
    auto global = py::dict(m.attr("__dict__"));
    global["m"] = py::module::import("factory_constructors");
    // global["pytest"] = py::module::import("pytest");
    auto pybind11_tests_module = py::module::import("pybind11_tests");
    global["ConstructorStats"] = pybind11_tests_module.attr("ConstructorStats");
    global["tag"] = global["m"].attr("tag");

    m.def("test_eval_file", [global](py::str filename) {
        auto local = py::dict();
        auto result = py::eval_file(filename, global, local);
        // return result.is_none();
    });

    auto sys_path = py::module::import("os").attr("path");
    auto filename = sys_path.attr("join")(sys_path.attr("dirname")(__FILE__), "test_factory_constructors.py");
    py::print(filename);
    CHECK_EQ(m.attr("test_eval_file")(filename).cast<bool>(), true);

}
