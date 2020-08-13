// k2/python/csrc/fsa_equivalent.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/fsa_equivalent.h"

#include "k2/csrc/array.h"
#include "k2/csrc/fsa_equivalent.h"
#include "k2/csrc/weights.h"

template <k2::FbWeightType Type>
void PyBindIsRandEquivalentTpl(py::module &m, const char *name) {
  m.def(
      name,
      [](const k2::Fsa &a, k2::Array1<float *> *a_weights, const k2::Fsa &b,
         k2::Array1<float *> *b_weights, float beam = k2::kFloatInfinity,
         float delta = 1e-6, bool top_sorted = true,
         std::size_t npath = 100) -> bool {
        return k2::IsRandEquivalent<Type>(a, a_weights->data, b,
                                          b_weights->data, beam, delta,
                                          top_sorted, npath);
      },
      py::arg("fsa_a"), py::arg("a_weights"), py::arg("fsa_b"),
      py::arg("b_weights"), py::arg("beam") = k2::kFloatInfinity,
      py::arg("delta") = 1e-6, py::arg("top_sorted") = true,
      py::arg("npath") = 100);
}

void PyBindRandPath(py::module &m) {
  using PyClass = k2::RandPath;
  py::class_<PyClass>(m, "_RandPath")
      .def(py::init<const k2::Fsa &, bool, int32_t>(), py::arg("fsa"),
           py::arg("no_eps_arc"), py::arg("eps_arc_tries") = 50)
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"))
      .def(
          "get_output",
          [](PyClass &self, k2::Fsa *fsa_out,
             k2::Array1<int32_t *> *arc_map = nullptr) -> bool {
            return self.GetOutput(fsa_out,
                                  arc_map == nullptr ? nullptr : arc_map->data);
          },
          py::arg("fsa_out"), py::arg("arc_map").none(true));
}

void PybindFsaEquivalent(py::module &m) {
  m.def("_is_rand_equivalent",
        (bool (*)(const k2::Fsa &, const k2::Fsa &, std::size_t)) &
            k2::IsRandEquivalent,
        py::arg("fsa_a"), py::arg("fsa_b"), py::arg("npath") = 100);

  PyBindIsRandEquivalentTpl<k2::kMaxWeight>(m,
                                            "_is_rand_equivalent_max_weight");
  PyBindIsRandEquivalentTpl<k2::kLogSumWeight>(
      m, "_is_rand_equivalent_logsum_weight");

  // maybe we don't need this version in Python code.
  m.def(
      "_is_rand_equivalent_after_rmeps_pruned_logsum",
      [](const k2::Fsa &a, k2::Array1<float *> *a_weights, const k2::Fsa &b,
         k2::Array1<float *> *b_weights, float beam, bool top_sorted = true,
         std::size_t npath = 100) -> bool {
        return k2::IsRandEquivalentAfterRmEpsPrunedLogSum(
            a, a_weights->data, b, b_weights->data, beam, top_sorted, npath);
      },
      py::arg("fsa_a"), py::arg("a_weights"), py::arg("fsa_b"),
      py::arg("b_weights"), py::arg("beam"), py::arg("top_sorted") = true,
      py::arg("npath") = 100);

  PyBindRandPath(m);
}
