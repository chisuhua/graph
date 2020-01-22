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
 * \file op_util.h
 * \brief Common utility used in operator construction.
 */
#pragma once

#include "../pass/arg_binder.h"
#include "../pass/ir_util.h"
#include "../schedule/message_passing.h"
#include <tvm/expr.h>
#include <tvm/schedule.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace op {

    using ir::MergeNest;

    /*!
 * \brief Build loop nest for stage.
 *
 * \param stage The stage to create a loop nest.
 * \param dom_map The range of each iter var.
 * \param begin_iter_pos The beginning position of leaf_iter_vars to generate loop.
 * \param new_loop_var Whether create new loop variable.
 * \param skip_iter Whether skip certain iteration.
 * \param p_value_map The result value of each IterVar.
 * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
 */
    std::vector<std::vector<Stmt>>
    MakeLoopNest(const Stage& stage,
        const std::unordered_map<IterVar, Range>& dom_map,
        size_t begin_iter_pos,
        bool new_loop_var,
        const std::unordered_set<IterVar>& skip_iter,
        std::unordered_map<IterVar, Expr>* p_value_map,
        bool debug_keep_trivial_loop);

    /*!
 * \brief Create a nest of if checking the predicates.
 *
 * \param predicates The predicates to be checked.
 * \return List of If nest that checks the predicates.
 */
    std::vector<Stmt> MakeIfNest(const std::vector<Expr>& predicates);

    /*!
 * \brief Replace the tensor reference (especially in Call's) in stmt by the replace map.
 * \param stmt The statement to be processed.
 * \param replace The replacement rule.
 */
    Stmt ReplaceTensor(Stmt stmt,
        const std::unordered_map<Tensor, Tensor>& replace);
    /*!
 * \brief Replace the tensor reference (especially in Call's) in stmt by the replace map.
 * \param expr The expression to be processed.
 * \param replace The replacement rule.
 */
    Expr ReplaceTensor(Expr expr,
        const std::unordered_map<Tensor, Tensor>& replace);

    /*!
 * \brief Substitute the variables of stmt by value map.
 * \param stmt the statment
 * \param value_map The value map.
 * \return Substituted result.
 */
    Stmt Substitute(Stmt stmt,
        const std::unordered_map<IterVar, Expr>& value_map);

    /*!
 * \brief Converts Halide ForType to its corresponding IterVarType
 * \param for_type The ForType to be converted
 */
    IterVarType ForTypeToIterVarType(ir::ForType for_type);

    /*!
 * \brief Converts IterVarType to its corresponding Halide ForType
 * \param iter_type The IterVarType to be converted
 */
    ir::ForType IterVarTypeToForType(IterVarType iter_type);

} // namespace op
} // namespace tvm
