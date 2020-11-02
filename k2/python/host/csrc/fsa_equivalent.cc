// k2/python/host/csrc/fsa_equivalent.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/host/csrc/fsa_equivalent.h"

#include "k2/csrc/host/array.h"
#include "k2/csrc/host/fsa_equivalent.h"

template <k2host::FbWeightType Type>
void PyBindIsRandEquivalentTpl(py::module &m, const char *name) {
  m.def(
      name,
      [](const k2host::Fsa &a, const k2host::Fsa &b,
         float beam = k2host::kFloatInfinity,
         bool treat_epsilons_specially = true, float delta = 1e-6,
         bool top_sorted = true, std::size_t npath = 100) -> bool {
        return k2host::IsRandEquivalent<Type>(
            a, b, beam, treat_epsilons_specially, delta, top_sorted, npath);
      },
      py::arg("fsa_a"), py::arg("fsa_b"),
      py::arg("beam") = k2host::kFloatInfinity,
      py::arg("treat_epsilons_specially") = true, py::arg("delta") = 1e-6,
      py::arg("top_sorted") = true, py::arg("npath") = 100);
}

void PyBindRandPath(py::module &m) {
  using PyClass = k2host::RandPath;
  py::class_<PyClass>(m, "_RandPath")
      .def(py::init<const k2host::Fsa &, bool, int32_t>(), py::arg("fsa"),
           py::arg("no_eps_arc"), py::arg("eps_arc_tries") = 50)
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"))
      .def(
          "get_output",
          [](PyClass &self, k2host::Fsa *fsa_out,
             k2host::Array1<int32_t *> *arc_map = nullptr) -> bool {
            return self.GetOutput(fsa_out,
                                  arc_map == nullptr ? nullptr : arc_map->data);
          },
          py::arg("fsa_out"), py::arg("arc_map").none(true));
}

void PybindFsaEquivalent(py::module &m) {
  m.def("_is_rand_equivalent",
        (bool (*)(const k2host::Fsa &, const k2host::Fsa &,
                  bool treat_epsilons_specially, std::size_t)) &
            k2host::IsRandEquivalent,
        py::arg("fsa_a"), py::arg("fsa_b"),
        py::arg("treat_epsilons_specially") = true, py::arg("npath") = 100);

  PyBindIsRandEquivalentTpl<k2host::kMaxWeight>(
      m, "_is_rand_equivalent_max_weight");
  PyBindIsRandEquivalentTpl<k2host::kLogSumWeight>(
      m, "_is_rand_equivalent_logsum_weight");

  // maybe we don't need this version in Python code.
  m.def(
      "_is_rand_equivalent_after_rmeps_pruned_logsum",
      [](const k2host::Fsa &a, const k2host::Fsa &b, float beam,
         bool top_sorted = true, std::size_t npath = 100) -> bool {
        return k2host::IsRandEquivalentAfterRmEpsPrunedLogSum(
            a, b, beam, top_sorted, npath);
      },
      py::arg("fsa_a"), py::arg("fsa_b"), py::arg("beam"),
      py::arg("top_sorted") = true, py::arg("npath") = 100);

  PyBindRandPath(m);
}
