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
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/op/type_relations.h
 * \brief A set of utilities and common functionality
 * for type relations.
 */
#pragma once

#include <tvm/relay/error.h>
#include <tvm/relay/type.h>
#include <string>

namespace tvm {
namespace relay {
    /*!
 * \brief The identity type relation, all the types are equal.
 *
 * \param types The input and output types to the relation.
 * \param num_inputs The number of input arguments.
 * \param attrs The attributes
 * \param reporter The reporter.
 * \return true whether relation has been resolved.
 */
    bool IdentityRel(const Array<Type>& types,
        int num_inputs,
        const Attrs& attrs,
        const TypeReporter& reporter);

    /*!
 * \brief The broadcast type relation, implements the broadcasting
 * rule over the two input types producing the broadcasted type.
 *
 * \param types The input and output types to the relation.
 * \param num_inputs The number of input arguments.
 * \param attrs The attributes
 * \param reporter The reporter.
 * \return true whether relation has been resolved.
 */
    bool BroadcastRel(const Array<Type>& types,
        int num_inputs,
        const Attrs& attrs,
        const TypeReporter& reporter);

    /*!
 * \brief The broadcast type relation, implements the broadcasting
 *  rule over the two input types producing the broadcasted type.
 *
 * This differs from BroadcastRel in the return dtype,
 * it instead returns bool(uint8), for use in comparsion operators
 * such as equal, not_equal, lt, and so on.
 *
 * \param types The input and output types to the relation.
 * \param num_inputs The number of input arguments.
 * \param attrs The attributes
 * \param reporter The reporter.
 * \return true whether relation has been resolved.
 */
    bool BroadcastCompRel(const Array<Type>& types,
        int num_inputs,
        const Attrs& attrs,
        const TypeReporter& reporter);

} // namespace relay
} // namespace tvm
