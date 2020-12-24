/**
 * @brief python wrappers for ragged_ops.h
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang
 *                                                  Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/ragged_ops.h"
#include "k2/python/csrc/torch/ragged_ops.h"
#include "k2/python/csrc/torch/torch_util.h"

namespace k2 {

template <typename T>
static void PybindRaggedRemoveAxis(py::module &m, const char *name) {
  m.def(name, &RemoveAxis<T>, py::arg("src"), py::arg("axis"));
}

template <typename T>
static void PybindRemoveValuesLeq(py::module &m, const char *name) {
  m.def(name, &RemoveValuesLeq<T>, py::arg("src"), py::arg("cutoff"));
}

template <typename T>
static void PybindRemoveValuesEq(py::module &m, const char *name) {
  m.def(name, &RemoveValuesEq<T>, py::arg("src"), py::arg("target"));
}

// Recursive implementation function used inside PybindToLists().
// Returns a list containing elements `begin` through `end-1` on
// axis `axis` of `r`, with 0 <= axis < r.NumAxes(), and
// 0 <= begin <= end <= r.TotSize(axis).
static py::list RaggedInt32ToList(Ragged<int32_t> &r, int32_t axis,
                                  int32_t begin, int32_t end) {
  K2_CHECK_LT(static_cast<uint32_t>(axis), static_cast<uint32_t>(r.NumAxes()));
  K2_CHECK_LE(end, r.TotSize(axis));
  py::list ans(end - begin);
  int32_t num_axes = r.NumAxes();
  int32_t *data;
  if (axis == num_axes - 1)
    data = r.values.Data();
  else
    data = r.RowSplits(axis + 1).Data();
  for (int32_t i = begin; i < end; i++) {
    if (axis == num_axes - 1) {
      ans[i - begin] = data[i];
    } else {
      int32_t row_begin = data[i], row_end = data[i + 1];
      ans[i - begin] = RaggedInt32ToList(r, axis + 1, row_begin, row_end);
    }
  }
  return ans;
};

static void PybindRaggedIntToList(py::module &m, const char *name) {
  m.def(
      name,
      [](Ragged<int32_t> &src) -> py::list {
        Ragged<int32_t> r = src.To(GetCpuContext());
        return RaggedInt32ToList(r, 0, 0, r.Dim0());
      },
      py::arg("src"));
}

template <typename T>
static void PybindNormalizePerSublist(py::module &m, const char *name) {
  m.def(name, &NormalizePerSublist<T>, py::arg("src"));
}

template <typename T>
static void PybindNormalizePerSublistBackward(py::module &m, const char *name) {
  m.def(
      name,
      /*
         @param [in] out      It is the output of `NormalizePerSublist(src)`.
         @param [in] out_grad The gradient for `out`.
         @return  Return the gradient for `src`.
       */
      [](Ragged<T> &out, torch::Tensor out_grad) -> torch::Tensor {
        Array1<T> out_grad_array = FromTensor<T>(out_grad);
        K2_CHECK_EQ(out.values.Dim(), out_grad_array.Dim());

        ContextPtr context = GetContext(out, out_grad_array);
        Ragged<T> out_grad_ragged(out.shape, out_grad_array);

        int32_t num_axes = out.NumAxes();
        Array1<T> out_grad_sum(context, out.TotSize(num_axes - 2));
        SumPerSublist<T>(out_grad_ragged, 0, &out_grad_sum);
        const T *out_grad_sum_data = out_grad_sum.Data();

        Array1<T> ans_grad_array(context, out_grad_array.Dim());
        const T *out_data = out.values.Data();
        const T *out_grad_data = out_grad_array.Data();
        T *ans_grad_data = ans_grad_array.Data();
        const int32_t *row_ids_data = out.RowIds(num_axes - 1).Data();
        int32_t num_elements = ans_grad_array.Dim();

        if (std::is_same<T, float>::value) {
          // use `expf` for float
          K2_EVAL(
              context, num_elements, lambda_set_ans_grad, (int32_t i)->void {
                int32_t row = row_ids_data[i];
                T scale = out_grad_sum_data[row];
                ans_grad_data[i] = out_grad_data[i] - expf(out_data[i]) * scale;
              });
        } else {
          // use `exp` for double
          K2_EVAL(
              context, num_elements, lambda_set_ans_grad, (int32_t i)->void {
                int32_t row = row_ids_data[i];
                T scale = out_grad_sum_data[row];
                ans_grad_data[i] = out_grad_data[i] - exp(out_data[i]) * scale;
              });
        }
        return ToTensor(ans_grad_array);
      },
      py::arg("out"), py::arg("out_grad"));
}

}  // namespace k2

void PybindRaggedOps(py::module &m) {
  using namespace k2;  // NOLINT
  PybindRaggedRemoveAxis<int32_t>(m, "ragged_int_remove_axis");
  PybindRemoveValuesLeq<int32_t>(m, "ragged_int_remove_values_leq");
  PybindRemoveValuesEq<int32_t>(m, "ragged_int_remove_values_eq");
  PybindRaggedIntToList(m, "ragged_int_to_list");
  PybindNormalizePerSublist<float>(m, "normalize_per_sublist");
  PybindNormalizePerSublistBackward<float>(m, "normalize_per_sublist_backward");
}
