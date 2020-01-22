#include <pybind11/embed.h>
#include "pybind11_tests.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "gtest/gtest.h"
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <dmlc/logging.h>
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/device_api.h"

using namespace tvm;
using namespace tvm::runtime;


std::vector<TVMContext> enabled_ctx_list()
{
    std::map<std::string,TVMContext> ctx_list = 
                    {{"cpu", TVMContext{kDLCPU, 0}},
                     {"gpu", TVMContext{kDLGPU, 0}},
                     {"cl", TVMContext{kDLOpenCL, 0}},
                     {"metal", TVMContext{kDLMetal, 0}},
                     {"rocm", TVMContext{kDLROCM, 0}},
                     {"vulkan", TVMContext{kDLVulkan, 0}},
                     {"vpi", TVMContext{kDLVPI, 0}}};
    std::vector<TVMContext> lst;
    for ( auto &item : ctx_list) {
        TVMRetValue val;
        TVMContext tvm_ctx = item.second;
        DeviceAPI *ptr = tvm::runtime::DeviceAPI::Get(tvm_ctx, true);
        if (ptr!= nullptr) {
            ptr->GetAttr(tvm_ctx, tvm::runtime::kExist, &val);
            if (val.operator int() == 1) {
                lst.push_back(tvm_ctx);
            }
        }
    }
    return lst;
}

DLDataType dtype2dltype(py::dtype d) {
    uint8_t bits = d.attr("itemsize").cast<int>();
    if (d.kind() == 'f') {
        return {kDLFloat, bits*8, 1};
    } else if (d.kind() == 'u') {
        return {kDLUInt, bits*8, 1};
    } else if (d.kind() == 'i') {
        return {kDLInt, bits*8, 1};
    } else {
        assert("Cant's convert\n");
    }
}

py::dtype dltype2dtype(DLDataType d) {
    uint itemsize = d.bits;

    if (d.code == kDLFloat) {
        return py::dtype(("float"+std::to_string(itemsize)));
    } else if (d.code == kDLUInt) {
        return py::dtype(("uint"+std::to_string(itemsize)));
    } else if (d.code == kDLUInt) {
        return py::dtype(("int"+std::to_string(itemsize)));
    } else {
        assert("Cant's convert\n");
    }
}
/*
py::array dltensor2pyarray(py::dtype d) {
    py::buffer_info buf_ndim()
}
*/

TEST(RUNTIME, test_nd_create)
{
    py::scoped_interpreter guard{};
    py::module np = py::module::import("numpy");
    std::vector<TVMContext> ctx_list = enabled_ctx_list();
    for(auto &ctx : ctx_list) {
        LOG(INFO) << "ctx: " << ctx ;
        for (auto &dtype : {py::dtype("uint8"),
                            py::dtype("int8"),
                            py::dtype("uint8"),
                            py::dtype("int16"),
                            py::dtype("uint32"),
                            py::dtype("int32"),
                            py::dtype("float32")}
                ) {
            // x = np.random.randint(0, 10, size=(3, 4))
            auto x = np.attr("random").attr("randint")(0, 10, std::vector<int>{3, 4});
            // x = np.array(x, dtype=dtype)
            auto xx = np.attr("array")(x, "dtype"_a=dtype);
            // y = tvm.nd.array(x, ctx=ctx)
            auto y = runtime::NDArray::Empty(xx.attr("shape").cast<std::vector<long int>>(),
                    dtype2dltype(dtype), ctx);

/* in c_runtime_api.h
typedef DLTensor TVMArray;
typedef TVMArray* TVMArrayHandle;
*/
            py::array a_in = xx;
            py::print(py::str(a_in.shape()));
            py::print(py::str(a_in.ndim()));
            py::print(py::str(a_in.shape() + a_in.ndim()));
            // TVMArrayCopyFromBytes(runtime::NDArray::Internal::MoveAsDLTensor(y), a.data(), a.nbytes());
            TVMArrayCopyFromBytes(&(y.ToDLPack()->dl_tensor), const_cast<void*>(a_in.data()), a_in.nbytes());

            auto shape = y.ToDLPack()->dl_tensor.shape;
            py::dtype a_out_dtype = dltype2dtype(y.ToDLPack()->dl_tensor.dtype);
            py::array a_out = py::array(dtype, shape);
            // auto r = a_out.mutable_unchecked<2>();
            py::buffer_info info = a_out.request();

            TVMArrayCopyToBytes(&(y.ToDLPack()->dl_tensor), info.ptr, a_out.nbytes());

            py::print(dtype);
            py::print(a_in);
            py::print(a_out);
            // np.attr("testing").attr("assert_equal")(a_in, a_out);
        }
    }
}

