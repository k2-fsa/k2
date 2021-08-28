

#include "k2/python/csrc/torch/doc/doc.h"

namespace k2 {

void SetMethodDoc(py::object *cls, const char *name, const char *doc) {
  py::function method = cls->attr(name);
  if (!method) {
    K2_LOG(FATAL) << "No such method: " << name;
  }

  if (!method.is_cpp_function()) {
    K2_LOG(FATAL) << "name: " << name << " is not a method";
  }

  py::handle h = method.cpp_function();
  auto f = reinterpret_cast<PyCFunctionObject *>(h.ptr());

  // Don't assume "doc" is statically allocated.
  // So we duplicate it here.
  f->m_ml->ml_doc = strdup(doc);
}

}  // namespace k2
