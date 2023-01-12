/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>
#include <mutex>  // NOLINT
#include <unordered_set>
#include <utility>
#include <vector>

#include "caffe2/serialize/file_adapter.h"
#include "caffe2/serialize/inline_container.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/ragged.h"
#include "k2/torch/csrc/deserialization.h"
#include "k2/torch/csrc/utils.h"
#include "torch/csrc/jit/serialization/import_source.h"
#if K2_TORCH_VERSION_MAJOR > 1 || \
    (K2_TORCH_VERSION_MAJOR == 1 && K2_TORCH_VERSION_MINOR >= 9)
// for torch::jit::readArchiveAndTensors
#include "torch/csrc/jit/serialization/import_read.h"
#endif

namespace k2 {

// A helper class to construct a Ragged<int32_t> from an archive
struct RaggedIntHelper : public Ragged<int32_t>,
                         public torch::CustomClassHolder {
  using k2::Ragged<int32_t>::Ragged;
  explicit RaggedIntHelper(const Ragged<int32_t> &ragged)
      : Ragged<int32_t>(ragged) {}
};

/** Whether the torch IValue contains a Ragged<int32_t> instance.

    @param value  The given torch IValue.
    @return Return true if the given value contains a Ragged<int32_t> instance,
            otherwise false.
 */
static bool IsRaggedInt(torch::IValue value) {
  return value.type() ==
         torch::getCustomClassType<torch::intrusive_ptr<RaggedIntHelper>>();
}

/// Convert an IValue to a Ragged<int32_t>
/// It is not static as it's used in deserialization_test.cu
/*static*/ Ragged<int32_t> ToRaggedInt(torch::IValue value) {
  auto ragged_int_holder = value.toCustomClass<RaggedIntHelper>();
  return *ragged_int_holder;
}

static void RegisterRaggedInt();

struct RaggedRegister {
  RaggedRegister() { RegisterRaggedInt(); }
};

// Register Ragged<int32_t> as a custom class of torch, so that we can wrap
// it to torch IValue and do serialization & deserialization thing.
static RaggedRegister ragged_register;

namespace {

// copied & modified from torch/csrc/jit/serialization/unpickler.cpp
void restoreAccurateTypeTags(const torch::IValue &root,
                             const torch::jit::TypePtr &type_tag) {
  struct Work {
    torch::jit::TypePtr static_type;
    torch::IValue value;
  };
  std::vector<Work> to_process = {{type_tag, root}};
  std::unordered_set<const void *> scanned;
  while (!to_process.empty()) {
    Work w = std::move(to_process.back());
    to_process.pop_back();
    // ensure we only scan each pointer value once, otherwise this
    // can become exponential (and if we allow recursive data in the future,
    // it would not terminiate).
    if (w.value.isPtrType()) {
      const void *key = w.value.internalToPointer();
      auto it = scanned.find(key);
      if (it != scanned.end()) {
        continue;
      }
      scanned.emplace_hint(it, key);
    }
    switch (w.static_type->kind()) {
      case torch::jit::TensorType::Kind:
      case torch::jit::NumberType::Kind:
      case torch::jit::FloatType::Kind:
      case torch::jit::IntType::Kind:
      case torch::jit::NoneType::Kind:
      case torch::jit::GeneratorType::Kind:
      case torch::jit::BoolType::Kind:
      case torch::jit::VarType::Kind:
      case torch::jit::CapsuleType::Kind:
      case torch::jit::PyObjectType::Kind:
      case torch::jit::StringType::Kind:
      case torch::jit::FunctionType::Kind:
      case torch::jit::DeviceObjType::Kind:
      case torch::jit::QSchemeType::Kind:
      case torch::jit::LayoutType::Kind:
      case torch::jit::ScalarTypeType::Kind:
      case torch::jit::RRefType::Kind:
      case torch::jit::AnyType::Kind:
      case torch::jit::AnyListType::Kind:
      case torch::jit::AnyTupleType::Kind:
      case torch::jit::AnyClassType::Kind:
#if K2_TORCH_VERSION_MAJOR > 1 || \
    (K2_TORCH_VERSION_MAJOR == 1 && K2_TORCH_VERSION_MINOR >= 7)
      case torch::jit::AnyEnumType::Kind:
      case torch::jit::QuantizerType::Kind:
#endif
        // no op, there is nothing to tag
        break;
#if K2_TORCH_VERSION_MAJOR > 1 || \
    (K2_TORCH_VERSION_MAJOR == 1 && K2_TORCH_VERSION_MINOR >= 7)
      case torch::jit::EnumType::Kind:
        // TODO(gmagogsfm): Implement serialization/deserialization of Enum.
        AT_ASSERT(false);
#endif
      case torch::jit::TupleType::Kind: {
        auto t = w.value.toTuple();
        auto ttype = w.static_type->expect<torch::jit::TupleType>();
        for (size_t i = 0; i < ttype->containedTypes().size(); ++i) {
          Work elem = {ttype->containedTypes().at(i), t->elements().at(i)};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case torch::jit::FutureType::Kind: {
        auto f = w.value.toFuture();
        auto t = w.static_type->expect<torch::jit::FutureType>();
        if (f->completed()) {
          Work elem = {t->getElementType(), f->value()};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case torch::jit::OptionalType::Kind: {
        if (!w.value.isNone()) {
          auto t = w.static_type->expect<torch::jit::OptionalType>();
          Work elem = {t->getElementType(), w.value};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case torch::jit::ListType::Kind: {
        // specialized lists do not need their type refined, so we can exit
        // early here
        if (!w.value.isList()) {
          break;
        }
        auto elem_type =
            w.static_type->cast<torch::jit::ListType>()->getElementType();
        auto lst = w.value.toList();
        lst.unsafeSetElementType(elem_type);
        for (const torch::IValue &item : lst) {
          Work elem = {elem_type, item};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case torch::jit::DictType::Kind: {
        auto dt = w.static_type->cast<torch::jit::DictType>();
        auto d = w.value.toGenericDict();
        d.unsafeSetKeyType(dt->getKeyType());
        d.unsafeSetValueType(dt->getValueType());
        for (const auto &item : d) {
          Work kelem = {dt->getKeyType(), item.key()};
          Work velem = {dt->getValueType(), item.value()};
          to_process.emplace_back(std::move(kelem));
          to_process.emplace_back(std::move(velem));
        }
      } break;
      // in both cases the dynamic type is a class, and we are going to tag with
      // the dynamic type
      case torch::jit::InterfaceType::Kind:
      case torch::jit::ClassType::Kind: {
        auto obj = w.value.toObject();
        auto typ = obj->type();  // note: intentionally using the dynamic type,
                                 // the static type is potentially less accurate
        for (size_t i = 0; i < typ->numAttributes(); ++i) {
          Work elem = {typ->getAttribute(i), obj->getSlot(i)};
          to_process.emplace_back(std::move(elem));
        }
      }
    }
  }
}

// modified from torch/csrc/jit/serialization/pickler.cpp
bool checkHasValidSetGetState(const std::shared_ptr<c10::ClassType> &cls) {
  // Check that the schemas for __getstate__ and __setstate__ are correct
  auto getstate = cls->findMethod("__getstate__");
  if (getstate == nullptr) {
    return false;
  }
  auto get_schema = getstate->getSchema();

  // Check __getstate__
  //   __getstate__ is expected to be (self) -> T
  K2_CHECK_EQ(get_schema.arguments().size(), 1)
      << "'__getstate__' must have 'self' as its only argument, but found "
      << get_schema.arguments().size() << " arguments";

  K2_CHECK_EQ(get_schema.returns().size(), 1)
      << "'__getstate__' must return 1 value, but found "
      << get_schema.returns().size();

  // Check __setstate__ if the method exists
  //   __setstate__ is expected to be (self, T) -> None
  auto setstate = cls->findMethod("__setstate__");
  if (!setstate) {
    return false;
  }
  auto set_schema = setstate->getSchema();

  K2_CHECK_EQ(set_schema.arguments().size(), 2)
      << "'__setstate__' must have 'self' and the state as its "
         "only arguments, but found "
      << set_schema.arguments().size() << " arguments";

  K2_CHECK_EQ(set_schema.returns().size(), 1)
      << "'__setstate__' must return None, but found "
      << set_schema.returns().size() << " return values";

  K2_CHECK(set_schema.returns().at(0).type()->isSubtypeOf(
      torch::jit::NoneType::get()))
      << "'__setstate__' must return None, but found value of type "
      << set_schema.returns().at(0).type()->annotation_str();

  // Check that the return type of __getstate__ matches the input to
  // __setstate__
  auto get_type = get_schema.returns().at(0).type();
  auto set_type = set_schema.arguments().at(1).type();

  K2_CHECK(get_type->isSubtypeOf(set_type))
      << "'__getstate__'s return type (" << get_type->annotation_str()
      << ") does not match '__setstate__'s argument type ("
      << set_type->annotation_str() << ")";

  return true;
}

// modified from torch/csrc/jit/serialization/import.cpp
// The code style in this function is also kept.
void postSetStateValidate(const torch::IValue &v) {
  auto obj = v.toObject();
  const auto &objType = obj->type();
  for (size_t i = 0; i < objType->numAttributes(); i++) {
    const auto &attrType = objType->getAttribute(i);
    const auto &attrName = objType->getAttributeName(i);
    const auto &slot = obj->getSlot(i);
    // const auto attrType = objType->getAttribute(i);
    // Verify that all the non-optional attributes have been initialized
    // TODO: Issue #20497
    if (attrType->kind() != torch::jit::TypeKind::OptionalType) {
      K2_CHECK(!slot.isNone())
          << "The field '" << attrName
          << "' was left uninitialized after '__setstate__',"
             "but expected a value of type '"
          << attrType->repr_str() << "'";
    }
  }
}

}  // namespace

static void RegisterRaggedInt() {
  // Register a custom class so that PyTorch knows how to parse
  // the value from the archive.
  //
  // TODO: to support other types other than Ragged<int32_t>
  torch::class_<RaggedIntHelper>("_k2", "RaggedTensor")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<RaggedIntHelper> &self) {
            std::vector<torch::IValue> v;
            c10::intrusive_ptr<torch::ivalue::Tuple> ans =
                torch::ivalue::Tuple::create(v);
            return torch::IValue(ans);
          },
          // __setstate__
          [](torch::IValue states) {
            K2_CHECK(states.isTuple());
            auto tuple = states.toTuple();
            auto &elements = tuple->elements();
            K2_CHECK(elements.size() == 3u || elements.size() == 5u)
                << "actual size: " << elements.size();

            // TODO: handle the case when size is 5
            K2_CHECK_EQ(elements.size(), 3u);

            k2::Array1<int32_t> row_splits =
                k2::Array1FromTorch<int32_t>(elements[0].toTensor());
            k2::Array1<int32_t> values =
                k2::Array1FromTorch<int32_t>(elements[2].toTensor());
            K2_CHECK_EQ(elements[1].toStringRef(), "row_ids1");

            k2::RaggedShape shape =
                k2::RaggedShape2(&row_splits, nullptr, values.Dim());

            return c10::make_intrusive<RaggedIntHelper>(shape, values);
          });

  // the default namespace for custom classes is __torch__.torch.classes
  // but `RaggedTensor` is serialized to the namespace _k2.ragged,
  // so we need to change it to `_k2.ragged`
  torch::ClassTypePtr p =
      torch::getCustomClassType<torch::intrusive_ptr<RaggedIntHelper>>();
  const_cast<torch::QualifiedName &>(p->name().value()) =
      torch::QualifiedName("_k2.ragged.RaggedTensor");
  // We need to re-register the class type since we changed its name.
  torch::registerCustomClass(p);
}

// This function is modified from torch::jit::load()
// See torch/csrc/jit/serialization/import.cpp
//
torch::IValue Load(
    const std::string &filename,
    torch::optional<torch::Device> map_location /*= torch::nullopt*/) {
  auto rai = std::make_unique<caffe2::serialize::FileAdapter>(filename);

  // Verify that we're loading a zip archive and not a torch.save pickle archive
  // (marked by the 0x80 0x02 bytes at the start)
  // i.e., _use_new_zipfile_serialization is False when torch.save was invoked
  uint8_t first_short[2];
  rai->read(
      /*pos=*/0,
      /*buf=*/&first_short,
      /*n=*/2,
      /*what=*/"checking archive");
  if (first_short[0] == 0x80 && first_short[1] == 0x02) {
    // NB: zip files by spec can start with any data, so technically they might
    // start with 0x80 0x02, but in practice zip files start with a file entry
    // which begins with 0x04034b50. Furthermore, PyTorch will never produce zip
    // files that do not start with the file entry, so it is relatively safe to
    // perform this check.
    K2_LOG(FATAL) << "Please set _use_new_zipfile_serialization to True "
                     "when invoking torch.save()";
  }

  auto reader = torch::make_unique<caffe2::serialize::PyTorchStreamReader>(
      std::move(rai));

  auto cu = std::make_shared<torch::jit::CompilationUnit>();
  torch::jit::SourceImporter source_importer(cu, nullptr, nullptr,
                                             reader->version());

  auto type_resolver = [&](const c10::QualifiedName &qn) {
    auto cls = source_importer.loadType(qn);
    return c10::StrongTypePtr(cu, std::move(cls));
  };

  // Decouple how to get obj from type.
  // For bytecode import we need to decouple these dependencies.
  auto obj_loader = [&](at::StrongTypePtr type, torch::IValue input) {
    auto cls = type.type_->expect<at::ClassType>();
    auto qn = cls->name();
    size_t n = cls->numAttributes();
    if (checkHasValidSetGetState(cls)) {
      auto obj = c10::ivalue::Object::create(type, n);
      // XXX: Do not optimize __setstate__, so that we don't try to
      // specialize the class before it is initialized.
      torch::jit::GraphOptimizerEnabledGuard guard(false);
      torch::jit::Function &set_state = cls->getMethod("__setstate__");
      // since we are in the middle of unpickling we might still have lists and
      // dicts that do not have accurate tags (e.g. they report they are
      // List[Any]). But we need to run __setstate__ which will check the input
      // type and may access the tags. Since setstate has a known input type, we
      // can correctly restore the tags now by apply the input type of set_state
      // to the state object being passed.
      // TODO: Remove once [serialization type tags] is landed
      restoreAccurateTypeTags(input,
                              set_state.getSchema().arguments().at(1).type());
      set_state({obj, input});
      postSetStateValidate(obj);
      return obj;
    } else {
      auto dict = std::move(input).toGenericDict();
      auto obj = c10::ivalue::Object::create(type, n);
      for (size_t i = 0; i < n; ++i) {
        obj->setSlot(i, dict.at(cls->getAttributeName(i)));
      }
      return obj;
    }
  };

#if K2_TORCH_VERSION_MAJOR > 1 || \
    (K2_TORCH_VERSION_MAJOR == 1 && K2_TORCH_VERSION_MINOR >= 9)
  torch::IValue ivalue = torch::jit::readArchiveAndTensors(
      "data", "", "", type_resolver, obj_loader,
      /*device=*/map_location, *reader);

#else
  torch::IValue ivalue =
      torch::jit::readArchiveAndTensors("data", type_resolver, obj_loader,
                                        /*device=*/map_location, *reader);
#endif
  return ivalue;
}

k2::FsaClass LoadFsa(
    const std::string &filename,
    torch::optional<torch::Device> map_location /*= torch::nullopt*/) {
  auto ivalue = Load(filename, map_location);
  K2_CHECK(ivalue.isGenericDict())
      << "Expect a dict. Given: " << ivalue.tagKind();

  torch::Dict<torch::IValue, torch::IValue> dict = ivalue.toGenericDict();
  K2_CHECK(dict.contains("arcs")) << "Expect to contain 'arcs' in the dict";

  Tensor arcs = TensorFromTorch(dict.at("arcs").toTensor());

  bool error = false;
  Fsa fsa;
  if (arcs.NumAxes() == 2) {
    fsa = FsaFromTensor(arcs, &error);
  } else if (arcs.NumAxes() == 1) {
    fsa = FsaVecFromTensor(arcs, &error);
  }
  K2_CHECK_EQ(error, false);

  FsaClass ans(fsa);

  (void)dict.erase(torch::IValue("arcs"));
  for (const auto &p : dict) {
    const auto &name = p.key().toStringRef();
    auto v = p.value();
    if (v.isTensor()) {
      ans.SetTensorAttr(name, v.toTensor());
    } else if (IsRaggedInt(v)) {
      ans.SetRaggedTensorAttr(name, ToRaggedInt(v));
    } else {
      K2_LOG(WARNING) << "Ignore non tensor attribute: '" << name
                      << "' of type: " << v.tagKind();
    }
  }

  return ans;
}

}  // namespace k2
