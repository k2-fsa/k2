#!/usr/bin/env python3
#
# Copyright (c)  2020  Fangjun Kuang
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# k2 tutorial lesson 1.
#
# This lesson demonstrates the basic usage of k2.
#
# You can run this lesson using
#
#   ctest --verbose -R lesson_01_basics

import k2


def main():
    # first, let us build an fsa from a string
    transtion_rules = r'''
    0 1 2
    0 1 3
    1 2 1
    2
    '''
    # Each line (except the last line) in the transition rules has three fields
    #
    #   current_state  next_state  label
    #
    # The last line indicates the final state. Note that the final state
    # has to have the largest state number. Although it can be inferred from
    # the transition rules, k2 requires that you have to provide it so that
    # the format is compatible with OpenFST.

    fsa = k2.string_to_fsa(transtion_rules)

    # Now that we have the fsa, we can visualize it
    renderer = k2.FsaRenderer(fsa)
    dot = renderer.render()
    print(dot)

    # You can install graphviz to visualize the fsa.
    #
    # (1) Install graphviz using pip:
    #       pip install graphviz
    #
    # (2) then add the following to the python code
    #       import graphviz
    #       source = graphviz.Source(dot)
    #       source.render('/tmp/lesson_01', format='png', cleanup=True)
    #
    # (3) You will get `tmp/lesson_01.png`.


if __name__ == '__main__':
    main()
