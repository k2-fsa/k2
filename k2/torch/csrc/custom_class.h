// This file is copied & modified from
// https://github.com/pytorch/pytorch/blob/master/torch/custom_class.h
// (We use the code from torch 1.7.1)
//
// The reason to make our own copy is that custom classes are registered
// inside the namespace "__torch__.torch.classes", which is hard coded.
//
// Classes in k2 are inside namespace `_k2` when wrapped to Python, so we need
// to make a copy and remove that constraint.
//
// If at some point, PyTorch supports to change the namespace where
// a custom class can be, we can get rid of this file.
//
// The new class is called torch::class2_, while the original class is
// torch::class_.
//
// Note: The code style of this file is different from other files in k2.
// We don't reformat it so that you can view the `diff` easily.
//
#pragma once

#include <ATen/core/stack.h>
#include <ATen/core/builtin_function.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/op_registration/infer_schema.h>
#include <ATen/core/stack.h>
#include <c10/util/C++17.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <c10/util/TypeTraits.h>
#include <torch/library.h>
#include <torch/custom_class_detail.h>
#include <iostream>
#include <sstream>

namespace torch {

/// Entry point for custom C++ class registration. To register a C++ class
/// in PyTorch, instantiate `torch::class_` with the desired class as the
/// template parameter. Typically, this instantiation should be done in
/// the initialization of a global variable, so that the class will be
/// made available on dynamic library loading without any additional API
/// calls needed. For example, to register a class named Foo, you might
/// create a global variable like so:
///
///     static auto register_foo = torch::class_<Foo>("myclasses", "Foo")
///       .def("myMethod", &Foo::myMethod)
///       .def("lambdaMethod", [](const c10::intrusive_ptr<Foo>& self) {
///         // Do something with `self`
///       });
///
/// In addition to registering the class, this registration also chains
/// `def()` calls to register methods. `myMethod()` is registered with
/// a pointer to the Foo class's `myMethod()` method. `lambdaMethod()`
/// is registered with a C++ lambda expression.
template <class CurClass>
class class2_ {
  static_assert(std::is_base_of<CustomClassHolder, CurClass>::value,
    "torch::class2_<T> requires T to inherit from CustomClassHolder");

 public:
  /// This constructor actually registers the class type.
  /// String argument `namespaceName` is an identifier for the
  /// namespace you would like this class to appear in.
  /// String argument `className` is the name you would like to
  /// see this class exposed as in Python and TorchScript. For example, if
  /// you pass `foo` as the namespace name and `Bar` as the className, the
  /// class will appear as `prefix.foo.Bar` in Python and TorchScript
  explicit class2_(const std::string &namespaceName,
                   const std::string &className,
                   const std::string &prefix = "") {
    detail::checkValidIdent(namespaceName, "Namespace name");
    detail::checkValidIdent(className, "Class name");
    // qualClassName = std::string("__torch__.torch.classes.") + namespaceName + "." + className;
    qualClassName = prefix + namespaceName + "." + className;

    classTypePtr = at::ClassType::create(
        c10::QualifiedName(qualClassName),
        std::weak_ptr<jit::CompilationUnit>());
    classTypePtr->addAttribute("capsule", at::CapsuleType::get());

    c10::getCustomClassTypeMap().insert(
        {std::type_index(typeid(c10::intrusive_ptr<CurClass>)), classTypePtr});
    c10::getCustomClassTypeMap().insert(
        {std::type_index(typeid(c10::tagged_capsule<CurClass>)), classTypePtr});

    registerCustomClass(classTypePtr);
  }

  /// def() can be used in conjunction with `torch::init()` to register
  /// a constructor for a given C++ class type. For example, passing
  /// `torch::init<int, std::string>()` would register a two-argument constructor
  /// taking an `int` and a `std::string` as argument.
  template <typename... Types>
  class2_& def(detail::types<void, Types...>) { // Used in combination with
                                               // torch::init<...>()
    auto func = [](c10::tagged_capsule<CurClass> self, Types... args) {
      auto classObj = c10::make_intrusive<CurClass>(args...);
      auto object = self.ivalue.toObject();
      object->setSlot(0, c10::IValue::make_capsule(std::move(classObj)));
    };

    defineMethod("__init__", std::move(func));
    return *this;
  }

  /// This is the normal method registration API. `name` is the name that
  /// the method will be made accessible by in Python and TorchScript.
  /// `f` is a callable object that defines the method. Typically `f`
  /// will either be a pointer to a method on `CurClass`, or a lambda
  /// expression that takes a `c10::intrusive_ptr<CurClass>` as the first
  /// argument (emulating a `this` argument in a C++ method.)
  ///
  /// Examples:
  ///
  ///     // Exposes method `foo` on C++ class `Foo` as `call_foo()` in
  ///     // Python and TorchScript
  ///     .def("call_foo", &Foo::foo)
  ///
  ///     // Exposes the given lambda expression as method `call_lambda()`
  ///     // in Python and TorchScript.
  ///     .def("call_lambda", [](const c10::intrusive_ptr<Foo>& self) {
  ///       // do something
  ///     })
  template <typename Func>
  class2_& def(std::string name, Func f) {
    auto wrapped_f = detail::wrap_func<CurClass, Func>(std::move(f));
    defineMethod(std::move(name), std::move(wrapped_f));
    return *this;
  }

  /// This is an unsafe method registration API added for adding custom JIT backend support via custom
  /// C++ classes. It is not for general purpose use.
  class2_& _def_unboxed(std::string name, std::function<void(jit::Stack&)> func, c10::FunctionSchema schema) {
    auto qualMethodName = qualClassName + "." + name;
    auto method = std::make_unique<jit::BuiltinOpFunction>(
        qualMethodName, std::move(schema), std::move(func));
    classTypePtr->addMethod(method.get());
    registerCustomClassMethod(std::move(method));
    return *this;
  }

  /// def_pickle() is used to define exactly what state gets serialized
  /// or deserialized for a given instance of a custom C++ class in
  /// Python or TorchScript. This protocol is equivalent to the Pickle
  /// concept of `__getstate__` and `__setstate__` from Python
  /// (https://docs.python.org/2/library/pickle.html#object.__getstate__)
  ///
  /// Currently, both the `get_state` and `set_state` callables must be
  /// C++ lambda expressions. They should have the following signatures,
  /// where `CurClass` is the class you're registering and `T1` is some object
  /// that encapsulates the state of the object.
  ///
  ///     __getstate__(intrusive_ptr<CurClass>) -> T1
  ///     __setstate__(T2) -> intrusive_ptr<CurClass>
  ///
  /// `T1` must be an object that is convertable to IValue by the same rules
  /// for custom op/method registration.
  ///
  /// For the common case, T1 == T2. T1 can also be a subtype of T2. An
  /// example where it makes sense for T1 and T2 to differ is if __setstate__
  /// handles legacy formats in a backwards compatible way.
  ///
  /// Example:
  ///
  ///     .def_pickle(
  ///         // __getstate__
  ///         [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
  ///           return self->stack_;
  ///         },
  ///         [](std::vector<std::string> state) { // __setstate__
  ///            return c10::make_intrusive<MyStackClass<std::string>>(
  ///               std::vector<std::string>{"i", "was", "deserialized"});
  ///         })
  template <typename GetStateFn, typename SetStateFn>
  class2_& def_pickle(GetStateFn&& get_state, SetStateFn&& set_state) {
    static_assert(
        c10::guts::is_stateless_lambda<std::decay_t<GetStateFn>>::value &&
            c10::guts::is_stateless_lambda<std::decay_t<SetStateFn>>::value,
        "def_pickle() currently only supports lambdas as "
        "__getstate__ and __setstate__ arguments.");
    def("__getstate__", std::forward<GetStateFn>(get_state));

    // __setstate__ needs to be registered with some custom handling:
    // We need to wrap the invocation of of the user-provided function
    // such that we take the return value (i.e. c10::intrusive_ptr<CurrClass>)
    // and assign it to the `capsule` attribute.
    using SetStateTraits =
        c10::guts::infer_function_traits_t<std::decay_t<SetStateFn>>;
    using SetStateArg = typename c10::guts::typelist::head_t<
        typename SetStateTraits::parameter_types>;
    auto setstate_wrapper = [set_state = std::move(set_state)](
                                c10::tagged_capsule<CurClass> self,
                                SetStateArg&& arg) {
      c10::intrusive_ptr<CurClass> classObj =
          at::guts::invoke(set_state, std::forward<SetStateArg>(arg));
      auto object = self.ivalue.toObject();
      object->setSlot(0, c10::IValue::make_capsule(classObj));
    };
    defineMethod(
        "__setstate__",
        detail::wrap_func<CurClass, decltype(setstate_wrapper)>(
            std::move(setstate_wrapper)));

    // type validation
    auto getstate_schema = classTypePtr->getMethod("__getstate__").getSchema();
    auto format_getstate_schema = [&getstate_schema]() {
      std::stringstream ss;
      ss << getstate_schema;
      return ss.str();
    };
    TORCH_CHECK(
        getstate_schema.arguments().size() == 1,
        "__getstate__ should take exactly one argument: self. Got: ",
        format_getstate_schema());
    auto first_arg_type = getstate_schema.arguments().at(0).type();
    TORCH_CHECK(
        *first_arg_type == *classTypePtr,
        "self argument of __getstate__ must be the custom class type. Got ",
        first_arg_type->repr_str());
    TORCH_CHECK(
        getstate_schema.returns().size() == 1,
        "__getstate__ should return exactly one value for serialization. Got: ",
        format_getstate_schema());

    auto ser_type = getstate_schema.returns().at(0).type();
    auto setstate_schema = classTypePtr->getMethod("__setstate__").getSchema();
    auto arg_type = setstate_schema.arguments().at(1).type();
    TORCH_CHECK(
        ser_type->isSubtypeOf(arg_type),
        "__getstate__'s return type should be a subtype of "
        "input argument of __setstate__. Got ",
        ser_type->repr_str(),
        " but expected ",
        arg_type->repr_str());

    return *this;
  }

 private:
  template <typename Func>
  void defineMethod(std::string name, Func func) {
    auto qualMethodName = qualClassName + "." + name;
    auto schema = c10::inferFunctionSchemaSingleReturn<Func>(std::move(name), "");

    auto wrapped_func = [func = std::move(func)](jit::Stack& stack) mutable -> void {
      // TODO: we need to figure out how to profile calls to custom functions
      // like this! Currently can't do it because the profiler stuff is in
      // libtorch and not ATen
      using RetType =
          typename c10::guts::infer_function_traits_t<Func>::return_type;
      detail::BoxedProxy<RetType, Func>()(stack, func);
    };
    auto method = std::make_unique<jit::BuiltinOpFunction>(
        qualMethodName, std::move(schema), std::move(wrapped_func));

    // Register the method here to keep the Method alive.
    // ClassTypes do not hold ownership of their methods (normally it
    // those are held by the CompilationUnit), so we need a proxy for
    // that behavior here.
    classTypePtr->addMethod(method.get());
    registerCustomClassMethod(std::move(method));
  }

  std::string qualClassName;
  at::ClassTypePtr classTypePtr;
};

}  // namespace torch
