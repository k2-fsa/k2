#!/usr/bin/env python3

import inspect
from typing import List

import k2


def get_function_names(module) -> List[str]:
    ans = []
    for m in inspect.getmembers(module):
        if inspect.isroutine(m[1]):
            ans.append(m[0])
    ans.sort()
    return ans


def get_class_names(module) -> List[str]:
    ans = []
    for m in inspect.getmembers(module):
        if inspect.isclass(m[1]):
            ans.append(m[0])
    ans.sort()
    return ans


def generate_doc_for_functions(names: List[str]) -> str:
    ans = ""
    for n in names:
        ans += n
        ans += "\n"
        ans += "-" * len(n)
        ans += "\n\n"
        ans += f".. autofunction:: {n}\n\n"
    return ans


def generate_doc_for_classes(module, names: List[str]) -> str:
    ans = ""
    for n in names:
        ans += n
        ans += "\n"
        ans += "-" * len(n)
        ans += "\n\n"
        cls = getattr(module, n)
        if hasattr(cls, "forward"):
            method_names = ["forward"]
        else:
            method_names = get_function_names(getattr(module, n))

        for m in method_names:
            m_obj = getattr(cls, m)
            if not hasattr(m_obj, "__doc__"):
                continue
            if m_obj.__doc__ is None:
                continue
            if m_obj.__doc__.count("\n") < 7 and m != "__str__":
                continue

            if m_obj.__doc__.count("\n") < 2:
                continue

            ans += m
            ans += "\n"
            ans += "^" * len(m)
            ans += "\n\n"

            ans += f".. automethod:: {module.__name__}.{cls.__name__}.{m}"
            ans += "\n\n"

    return ans


def main():
    doc = "k2\n"
    doc += "=" * len("k2")
    doc += "\n\n"

    doc += ".. currentmodule:: k2"
    doc += "\n\n"

    function_names = get_function_names(k2)
    doc += generate_doc_for_functions(function_names)

    class_names = get_class_names(k2)
    doc += generate_doc_for_classes(k2, class_names)

    doc += "k2.ragged\n"
    doc += "=" * len("k2.ragged")
    doc += "\n\n"

    doc += ".. currentmodule:: k2.ragged"
    doc += "\n\n"

    function_names = get_function_names(k2.ragged)
    doc += generate_doc_for_functions(function_names)

    # Note: We have exported k2.ragged.RaggedTensor and k2.ragged.RaggedShape
    class_names = get_class_names(k2.ragged)
    doc += generate_doc_for_classes(k2.ragged, class_names)

    print(doc)


if __name__ == "__main__":
    main()
