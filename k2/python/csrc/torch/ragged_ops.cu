/**
 * @brief python wrappers for ragged_ops.h
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corp.       (authors: Fangjun Kuang
 *                                                   Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <tuple>
#include <utility>
#include <vector>

#include "k2/csrc/ragged_ops.h"
#include "k2/python/csrc/torch/ragged_ops.h"
#include "k2/python/csrc/torch/torch_util.h"

namespace k2 {

template <typename T>
static void PybindRaggedRemoveAxis(py::module &m) {
  // src is a Ragged<T>
  //  there is another `remove_axis` in k2/python/csrc/torch/ragged.cu
  //  taking a RaggedShape as input.
  m.def("remove_axis", &RemoveAxis<T>, py::arg("src"), py::arg("axis"));
}

template <typename T>
static void PybindRaggedArange(py::module &m, const char *name) {
  m.def(
      name,
      [](Ragged<T> &src, int32_t axis, int32_t begin,
         int32_t end) -> Ragged<T> { return Arange<T>(src, axis, begin, end); },
      py::arg("src"), py::arg("axis"), py::arg("begin"), py::arg("end"));
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
  m.def(name, &NormalizePerSublist<T>, py::arg("src"), py::arg("use_log"));
}

/* Backward propagation for NormalizePerSublist.

   @param [in] out      It is the output of `NormalizePerSublist(src)`.
   @param [in] use_log  It indicates which kind of normalization was used.
   @param [in] out_grad The gradient for `out`; must have same type as `out`
                        (float or double), and shape (out.NumElements(),).
   @return  Return the gradient for `src`.  A torch.Tensor with shape
                        (out.NumElements(),).
 */
template <typename T>
static torch::Tensor NormalizePerSublistBackward(Ragged<T> &out, bool use_log,
                                                 torch::Tensor out_grad) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(out_grad.dim(), 1)
      << "Expected dim: 1. Given: " << out_grad.dim();
  K2_CHECK_EQ(out_grad.scalar_type(), ToScalarType<T>::value)
      << "Expected scalar type: " << ToScalarType<T>::value
      << ". Given: " << out_grad.scalar_type();
  K2_CHECK(use_log) << "It supports only use_log==True at present";

  ContextPtr context = GetContext(out_grad);
  K2_CHECK(context->IsCompatible(*out.Context()));

  int32_t num_axes = out.NumAxes();
  Array1<T> out_grad_sum(context, out.TotSize(num_axes - 2));
  T *out_grad_sum_data = out_grad_sum.Data();
  const T *out_grad_data = out_grad.data_ptr<T>();

  int64_t stride = out_grad.strides()[0];
  if (stride != 0) {
    Array1<T> out_grad_array = FromTensor<T>(out_grad);
    K2_CHECK_EQ(out.values.Dim(), out_grad_array.Dim());

    Ragged<T> out_grad_ragged(out.shape, out_grad_array);
    SumPerSublist<T>(out_grad_ragged, 0, &out_grad_sum);
  } else {
    // stride is 0;
    // the sum is the number_of_elements_in_the_sublist * out_grad[0]
    const int32_t *row_splits_data = out.RowSplits(num_axes - 1).Data();
    K2_EVAL(
        context, out_grad_sum.Dim(), lambda_compute_out_grad_sum,
        (int32_t i)->void {
          int32_t begin = row_splits_data[i];
          int32_t end = row_splits_data[i + 1];
          out_grad_sum_data[i] = (end - begin) * out_grad_data[0];
        });
  }

  Array1<T> ans_grad_array(context, out.NumElements());
  T *ans_grad_data = ans_grad_array.Data();
  const T *out_data = out.values.Data();
  const int32_t *row_ids_data = out.RowIds(num_axes - 1).Data();
  int32_t num_elements = ans_grad_array.Dim();

  if (std::is_same<T, float>::value) {
    // use `expf` for float
    K2_EVAL(
        context, num_elements, lambda_set_ans_grad, (int32_t i)->void {
          int32_t row = row_ids_data[i];
          T scale = out_grad_sum_data[row];
          ans_grad_data[i] =
              out_grad_data[i * stride] - expf(out_data[i]) * scale;
        });
  } else {
    // use `exp` for double
    K2_EVAL(
        context, num_elements, lambda_set_ans_grad, (int32_t i)->void {
          int32_t row = row_ids_data[i];
          T scale = out_grad_sum_data[row];
          ans_grad_data[i] =
              out_grad_data[i * stride] - exp(out_data[i]) * scale;
        });
  }
  return ToTensor(ans_grad_array);
}

template <typename T>
static void PybindNormalizePerSublistBackward(py::module &m, const char *name) {
  m.def(name, NormalizePerSublistBackward<T>, py::arg("out"),
        py::arg("use_log"), py::arg("out_grad"));
}

template <typename T, typename Op>
static void PybindOpPerSublist(py::module &m, Op op, const char *name) {
  m.def(
      name,
      [op](Ragged<T> &src, T initial_value) -> torch::Tensor {
        Array1<T> values(src.Context(), src.TotSize(src.NumAxes() - 2));
        op(src, initial_value, &values);
        return ToTensor(values);
      },
      py::arg("src"), py::arg("initial_value"));
}

template <typename T>
static void PybindAppend(py::module &m) {
  // py::list is more efficient, but it requires more code
  m.def(
      "append",
      [](std::vector<Ragged<T>> &srcs, int32_t axis) -> Ragged<T> {
        return Append(axis, srcs.size(), &srcs[0]);
      },
      py::arg("srcs"), py::arg("axis"));
}

template <typename T>
static void PybindCreateRagged2(py::module &m) {
  m.def(
      "create_ragged2",
      [](const std::vector<std::vector<T>> &vecs) -> Ragged<T> {
        return CreateRagged2(vecs);
      },
      py::arg("vecs"));
}

static void PybindGetLayer(py::module &m) {
  m.def("get_layer", &GetLayer, py::arg("src"), py::arg("layer"));
}

static void PybindUniqueSequences(py::module &m) {
  m.def(
      "unique_sequences",
      [](Ragged<int32_t> &src, bool need_num_repeats = true,
         bool need_new2old_indexes = false)
          -> std::tuple<Ragged<int32_t>, torch::optional<Ragged<int32_t>>,
                        torch::optional<torch::Tensor>> {
        Ragged<int32_t> num_repeats;
        Array1<int32_t> new2old_indexes;
        Ragged<int32_t> ans =
            UniqueSequences(src, need_num_repeats ? &num_repeats : nullptr,
                            need_new2old_indexes ? &new2old_indexes : nullptr);

        torch::optional<Ragged<int32_t>> num_repeats_tensor;
        if (need_num_repeats) num_repeats_tensor = num_repeats;

        torch::optional<torch::Tensor> new2old_indexes_tensor;
        if (need_new2old_indexes)
          new2old_indexes_tensor = ToTensor(new2old_indexes);

        return std::make_tuple(ans, num_repeats_tensor, new2old_indexes_tensor);
      },
      py::arg("src"), py::arg("need_num_repeats") = true,
      py::arg("need_new2old_indexes") = false);
}

static void PybindIndex(py::module &m) {
  // Note there are several overloads of `index`
  // in k2/python/csrc/torch/ragged.cu

  // return a pair:
  //  - ans (RaggedShape)
  //  - value_indexes (optional)
  //
  m.def(
      "index",
      [](RaggedShape &src, int32_t axis, torch::Tensor indexes,
         bool need_value_indexes =
             true) -> std::pair<RaggedShape, torch::optional<torch::Tensor>> {
        Array1<int32_t> indexes_array = FromTensor<int32_t>(indexes);
        Array1<int32_t> value_indexes;
        RaggedShape ans = Index(src, axis, indexes_array,
                                need_value_indexes ? &value_indexes : nullptr);

        torch::optional<torch::Tensor> value_indexes_tensor;
        if (need_value_indexes) value_indexes_tensor = ToTensor(value_indexes);

        return std::make_pair(ans, value_indexes_tensor);
      },
      py::arg("src"), py::arg("axis"), py::arg("indexes"),
      py::arg("need_value_indexes") = true);
}

static void PybindRegularRaggedShape(py::module &m) {
  // TODO(fangjun): pass a torch.device to specify the context
  //
  // As a workaround, the user can use
  // _k2.regular_ragged_shape(...).to(torch.device)
  // to move it to a given device
  m.def(
      "regular_ragged_shape",
      [](int32_t dim0, int32_t dim1) -> RaggedShape {
        ContextPtr c = GetCpuContext();
        return RegularRaggedShape(c, dim0, dim1);
      },
      py::arg("dim0"), py::arg("dim1"));
}

template <typename T>
static void PybindArgMaxPerSublist(py::module &m) {
  m.def(
      "argmax_per_sublist",
      [](Ragged<T> &src, T initial_value) -> torch::Tensor {
        int32_t last_axis = src.NumAxes() - 1;
        const Array1<int32_t> &row_splits_array = src.RowSplits(last_axis);
        int32_t num_rows = row_splits_array.Dim() - 1;

        Array1<int32_t> indexes(src.Context(), num_rows);
        ArgMaxPerSublist(src, initial_value, &indexes);

        return ToTensor(indexes);
      },
      py::arg("src"), py::arg("initial_value"));
}

template <typename T>
static void PybindMaxPerSublist(py::module &m) {
  m.def(
      "max_per_sublist",
      [](Ragged<T> &src, T initial_value) -> torch::Tensor {
        int32_t last_axis = src.NumAxes() - 1;
        const Array1<int32_t> &row_splits_array = src.RowSplits(last_axis);
        int32_t num_rows = row_splits_array.Dim() - 1;

        Array1<T> max_values(src.Context(), num_rows);
        MaxPerSublist(src, initial_value, &max_values);

        return ToTensor(max_values);
      },
      py::arg("src"), py::arg("initial_value"));
}


}  // namespace k2

void PybindRaggedOps(py::module &m) {
  using namespace k2;  // NOLINT
  PybindRaggedRemoveAxis<int32_t>(m);
  PybindRaggedArange<int32_t>(m, "ragged_int_arange");
  PybindRemoveValuesLeq<int32_t>(m, "ragged_int_remove_values_leq");
  PybindRemoveValuesEq<int32_t>(m, "ragged_int_remove_values_eq");
  PybindRaggedIntToList(m, "ragged_int_to_list");
  PybindNormalizePerSublist<float>(m, "normalize_per_sublist");
  PybindNormalizePerSublistBackward<float>(m, "normalize_per_sublist_backward");
  PybindOpPerSublist<float>(m, SumPerSublist<float>, "sum_per_sublist");
  PybindAppend<int32_t>(m);
  PybindCreateRagged2<int32_t>(m);
  PybindCreateRagged2<float>(m);
  PybindGetLayer(m);
  PybindUniqueSequences(m);
  PybindIndex(m);
  PybindRegularRaggedShape(m);
  PybindArgMaxPerSublist<float>(m);
  PybindMaxPerSublist<float>(m);
}
