

#ifndef K2_PYTHON_CSRC_TORCH_DOC_DOC_H_
#define K2_PYTHON_CSRC_TORCH_DOC_DOC_H_
#include "k2/python/csrc/torch.h"

namespace k2 {

/* Set the documentation of a method inside a class.

   @param cls The class whose method's doc we are going to set
   @param name  The name of the method
   @param doc  The doc string to set.
 */
void SetMethodDoc(py::object *cls, const char *name, const char *doc);

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_DOC_DOC_H_
