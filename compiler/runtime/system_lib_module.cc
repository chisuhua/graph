/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2017 by Contributors
 * \file system_lib_module.cc
 * \brief SystemLib module.
 */
#include "module_util.h"
#include <mutex>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

    class SystemLibModuleNode : public ModuleNode {
    public:
        SystemLibModuleNode() = default;

        const char* type_key() const final
        {
            return "system_lib";
        }

        PackedFunc GetFunction(
            const std::string& name,
            const std::shared_ptr<ModuleNode>& sptr_to_self) final
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (module_blob_ != nullptr) {
                // If we previously recorded submodules, load them now.
                ImportModuleBlob(reinterpret_cast<const char*>(module_blob_), &imports_);
                module_blob_ = nullptr;
            }

            auto it = tbl_.find(name);
            if (it != tbl_.end()) {
                return WrapPackedFunc(
                    reinterpret_cast<BackendPackedCFunc>(it->second), sptr_to_self);
            } else {
                return PackedFunc();
            }
        }

        void RegisterSymbol(const std::string& name, void* ptr)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (name == symbol::tvm_module_ctx) {
                void** ctx_addr = reinterpret_cast<void**>(ptr);
                *ctx_addr = this;
            } else if (name == symbol::tvm_dev_mblob) {
                // Record pointer to content of submodules to be loaded.
                // We defer loading submodules to the first call to GetFunction().
                // The reason is that RegisterSymbol() gets called when initializing the
                // syslib (i.e. library loading time), and the registeries aren't ready
                // yet. Therefore, we might not have the functionality to load submodules
                // now.
                CHECK(module_blob_ == nullptr) << "Resetting mobule blob?";
                module_blob_ = ptr;
            } else {
                auto it = tbl_.find(name);
                if (it != tbl_.end() && ptr != it->second) {
                    LOG(WARNING) << "SystemLib symbol " << name
                                 << " get overriden to a different address "
                                 << ptr << "->" << it->second;
                }
                tbl_[name] = ptr;
            }
        }

        static const std::shared_ptr<SystemLibModuleNode>& Global()
        {
            static std::shared_ptr<SystemLibModuleNode> inst = std::make_shared<SystemLibModuleNode>();
            return inst;
        }

    private:
        // Internal mutex
        std::mutex mutex_;
        // Internal symbol table
        std::unordered_map<std::string, void*> tbl_;
        // Module blob to be imported
        void* module_blob_ { nullptr };
    };

    TVM_REGISTER_GLOBAL("module._GetSystemLib")
        .set_body([](TVMArgs args, TVMRetValue* rv) {
            *rv = runtime::Module(SystemLibModuleNode::Global());
        });
} // namespace runtime
} // namespace tvm

int TVMBackendRegisterSystemLibSymbol(const char* name, void* ptr)
{
    tvm::runtime::SystemLibModuleNode::Global()->RegisterSymbol(name, ptr);
    return 0;
}
