/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// One host call for a fixed sequence of generated Cake bindings
// (flashinfer/jit/cake_program_sequence.py).
//
// make_sequence(runs, args, slots) returns a function that calls runs[0], runs[1], ... back to
// back from C++, each with its argument list args[i], in which the positions named by slots[i]
// (pairs of a position and a call-argument index) take the sequence call's own arguments. Every
// binding keeps its device guard, its argument checks and its launch; the sequence only removes
// the Python round trip between the launches of one operation call. A binding that raises stops
// the sequence, and the error reaches the caller unchanged.
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace cake_program_sequence {

using tvm::ffi::Any;
using tvm::ffi::AnyView;
using tvm::ffi::Array;
using tvm::ffi::Function;
using tvm::ffi::PackedArgs;

// The most arguments one binding call of a sequence takes, and the most call arguments.
constexpr size_t kMaxArgs = 96;
constexpr int64_t kMaxCallArgs = 64;

struct Step {
  Function run;
  std::vector<Any> args;
  // (position in args, index of the sequence call's argument)
  std::vector<std::pair<int32_t, int32_t>> slots;
};

Function MakeSequence(Array<Function> runs, Array<Array<Any>> args, Array<Array<int64_t>> slots) {
  TVM_FFI_CHECK(runs.size() >= 1 && runs.size() == args.size() && runs.size() == slots.size(),
                ValueError)
      << "make_sequence takes one argument list and one slot list per binding, got " << runs.size()
      << ", " << args.size() << " and " << slots.size();
  auto steps = std::make_shared<std::vector<Step>>();
  int32_t arity = 0;
  for (size_t i = 0; i < runs.size(); ++i) {
    Step step;
    step.run = runs[i];
    for (const Any& value : args[i]) step.args.push_back(value);
    TVM_FFI_CHECK(step.args.size() <= kMaxArgs, ValueError)
        << "binding " << i << " takes " << step.args.size() << " arguments, more than " << kMaxArgs;
    const Array<int64_t> pairs = slots[i];
    TVM_FFI_CHECK(pairs.size() % 2 == 0, ValueError)
        << "slot list " << i << " must hold (position, index) pairs";
    for (size_t j = 0; j < pairs.size(); j += 2) {
      const int64_t position = pairs[j];
      const int64_t index = pairs[j + 1];
      TVM_FFI_CHECK(position >= 0 && position < static_cast<int64_t>(step.args.size()) &&
                        index >= 0 && index < kMaxCallArgs,
                    ValueError)
          << "slot (" << position << ", " << index << ") of binding " << i << " is out of range";
      step.slots.emplace_back(static_cast<int32_t>(position), static_cast<int32_t>(index));
      arity = std::max(arity, static_cast<int32_t>(index + 1));
    }
    steps->push_back(std::move(step));
  }
  return Function::FromPacked([steps, arity](PackedArgs call, Any* result) {
    TVM_FFI_CHECK(call.size() == arity, TypeError)
        << "the sequence takes " << arity << " arguments, got " << call.size();
    AnyView argv[kMaxArgs];
    for (const Step& step : *steps) {
      const size_t n = step.args.size();
      for (size_t j = 0; j < n; ++j) argv[j] = step.args[j];
      for (const auto& slot : step.slots) argv[slot.first] = call[slot.second];
      Any ignored;
      step.run.CallPacked(argv, static_cast<int32_t>(n), &ignored);
    }
  });
}

}  // namespace cake_program_sequence

TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_sequence, cake_program_sequence::MakeSequence);
