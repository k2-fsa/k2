// k2/python/csrc/fsa.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/fsa.h"

#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_util.h"
#include "k2/python/csrc/dlpack.h"

using k2::Arc;
using k2::Cfsa;
using k2::CfsaVec;
using k2::Fsa;

// refer to
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L375
// https://github.com/microsoft/onnxruntime-tvm/blob/master/python/tvm/_ffi/_ctypes/ndarray.py#L28
// https://github.com/cupy/cupy/blob/master/cupy/core/dlpack.pyx#L66
// PyTorch, TVM and CuPy name the created dltensor to be `dltensor`
static const char *kDLPackTensorName = "dltensor";

// refer to
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L402
// https://github.com/apache/incubator-tvm/blob/master/python/tvm/_ffi/_ctypes/ndarray.py#L29
// https://github.com/cupy/cupy/blob/master/cupy/core/dlpack.pyx#L62
// PyTorch, TVM and CuPy name the used dltensor to be `used_dltensor`
static const char *kDLPackUsedTensorName = "used_dltensor";

/*
static void DLPackDeleter(void *p) {
  auto dl_managed_tensor = reinterpret_cast<DLManagedTensor *>(p);

  if (dl_managed_tensor && dl_managed_tensor->deleter)
    dl_managed_tensor->deleter(dl_managed_tensor);

  // this will be invoked if you uncomment it, which
  // means Python will indeed free the memory returned by the subsequent
  // `CfsaVecFromDLPack()`.
  //
  // LOG(INFO) << "freed!";
}

// the returned pointer is freed by Python
static CfsaVec *CfsaVecFromDLPack(py::capsule *capsule,
                                  const std::vector<Cfsa> *cfsas = nullptr) {
  // the following error message is modified from
  // https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L384
  CHECK_EQ(strcmp(kDLPackTensorName, capsule->name()), 0)
      << "Expected capsule name: " << kDLPackTensorName << "\n"
      << "But got: " << capsule->name() << "\n"
      << "Note that DLTensor capsules can be consumed only once,\n"
      << "so you might have already constructed a tensor from it once.";

  PyCapsule_SetName(capsule->ptr(), kDLPackUsedTensorName);

  DLManagedTensor *managed_tensor = *capsule;
  // (fangjun): the above assignment will either throw or succeed with a
  // non-null ptr; so no need to check for nullptr below

  auto tensor = &managed_tensor->dl_tensor;
  CHECK_EQ(tensor->ndim, 1) << "Expect 1-D tensor";
  CHECK_EQ(tensor->dtype.code, kDLInt);
  CHECK_EQ(tensor->dtype.bits, 32);
  CHECK_EQ(tensor->dtype.lanes, 1);
  CHECK_EQ(tensor->strides[0], 1);  // memory should be contiguous

  auto ctx = &tensor->ctx;
  // TODO(fangjun): enable GPU once k2 supports GPU.
  CHECK_EQ(ctx->device_type, kDLCPU);

  auto start_ptr = reinterpret_cast<char *>(tensor->data) + tensor->byte_offset;
  CHECK_EQ((intptr_t)start_ptr % sizeof(int32_t), 0);

  if (cfsas)
    CreateCfsaVec(*cfsas, start_ptr, tensor->shape[0] * sizeof(int32_t));

  // no memory leak here; python will deallocate it
  auto cfsa_vec = new CfsaVec(tensor->shape[0], start_ptr);
  cfsa_vec->SetDeleter(&DLPackDeleter, managed_tensor);

  return cfsa_vec;
}


static void PybindCfsaVec(py::module &m) {
  m.def("get_cfsa_vec_size",
        overload_cast_<const Cfsa &>()(&k2::GetCfsaVecSize), py::arg("cfsa"));

  m.def("get_cfsa_vec_size",
        overload_cast_<const std::vector<Cfsa> &>()(&k2::GetCfsaVecSize),
        py::arg("cfsas"));

  py::class_<CfsaVec>(m, "CfsaVec")
      .def("num_fsas", &CfsaVec::NumFsas)
      .def("__getitem__", [](const CfsaVec &self, int i) { return self[i]; },
           py::keep_alive<0, 1>());

  m.def("create_cfsa_vec",
        [](py::capsule *capsule, const std::vector<Cfsa> *cfsas = nullptr) {
          return CfsaVecFromDLPack(capsule, cfsas);
        },
        py::arg("dlpack"), py::arg("cfsas") = nullptr,
        py::return_value_policy::take_ownership);
}
*/

void PybindFsa(py::module &m) {
  /*
py::class_<Arc>(m, "Arc")
    .def(py::init<>())
    .def(py::init<int32_t, int32_t, int32_t>(), py::arg("src_state"),
         py::arg("dest_state"), py::arg("label"))
    .def_readwrite("src_state", &Arc::src_state)
    .def_readwrite("dest_state", &Arc::dest_state)
    .def_readwrite("label", &Arc::label)
    .def("__str__", [](const Arc &self) {
      std::ostringstream os;
      os << self;
      return os.str();
    });

py::class_<Fsa>(m, "Fsa")
    .def(py::init<>())
    .def("num_states", &Fsa::NumStates)
    .def("final_state", &Fsa::FinalState)
    .def("__str__", [](const Fsa &self) { return FsaToString(self); })
    .def_readwrite("arc_indexes", &Fsa::arc_indexes)
    .def_readwrite("arcs", &Fsa::arcs);

py::class_<std::vector<Fsa>>(m, "FsaVec")
    .def(py::init<>())
    .def("clear", &std::vector<Fsa>::clear)
    .def("__len__", [](const std::vector<Fsa> &self) { return self.size(); })
    .def("push_back",
         [](std::vector<Fsa> *self, const Fsa &fsa) { self->push_back(fsa); })
    .def("__iter__",
         [](const std::vector<Fsa> &self) {
           return py::make_iterator(self.begin(), self.end());
         },
         py::keep_alive<0, 1>());
// py::keep_alive<Nurse, Patient>
// 0 is the return value and 1 is the first argument.
// Keep the patient (i.e., `self`) alive as long as the Nurse (i.e., the
// return value) is not freed.

py::class_<std::vector<Arc>>(m, "ArcVec")
    .def(py::init<>())
    .def("clear", &std::vector<Arc>::clear)
    .def("__len__", [](const std::vector<Arc> &self) { return self.size(); })
    .def("__iter__",
         [](const std::vector<Arc> &self) {
           return py::make_iterator(self.begin(), self.end());
         },
         py::keep_alive<0, 1>());

py::class_<Cfsa>(m, "Cfsa")
    .def(py::init<>())
    .def(py::init<const Fsa &>(), py::arg("fsa"), py::keep_alive<1, 2>())
    .def("num_states", &Cfsa::NumStates)
    .def("num_arcs", &Cfsa::NumArcs)
    .def("arc",
         [](Cfsa *self, int s) {
           DCHECK_GE(s, 0);
           DCHECK_LT(s, self->NumStates());
           auto begin = self->arc_indexes[s];
           auto end = self->arc_indexes[s + 1];
           return py::make_iterator(self->arcs + begin, self->arcs + end);
         },
         py::keep_alive<0, 1>())
    .def("__str__",
         [](const Cfsa &self) {
           std::ostringstream os;
           os << self;
           return os.str();
         })
    .def("__eq__",  // for test only
         [](const Cfsa &self, const Cfsa &other) { return self == other; });

py::class_<std::vector<Cfsa>>(m, "CfsaStdVec")
    .def(py::init<>())
    .def("clear", &std::vector<Cfsa>::clear)
    .def("push_back", [](std::vector<Cfsa> *self,
                         const Cfsa &cfsa) { self->push_back(cfsa); })
    .def("__len__", [](const std::vector<Cfsa> &self) { return self.size(); })
    .def("__iter__",
         [](const std::vector<Cfsa> &self) {
           return py::make_iterator(self.begin(), self.end());
         },
         py::keep_alive<0, 1>());
  PybindCfsaVec(m);
*/
}
