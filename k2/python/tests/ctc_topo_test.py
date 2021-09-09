#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corporation      (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To run this single test, use
#
#  ctest --verbose -R ctc_topo_test_py

import unittest

import k2
import torch


class TestCtcTopo(unittest.TestCase):

    @staticmethod
    def visualize_ctc_topo():
        '''This function shows how to visualize
        standard/modified ctc topologies. It's for
        demonstration only, not for testing.
        '''
        max_token = 2
        labels_sym = k2.SymbolTable.from_str('''
            <blk> 0
            z 1
            o 2
        ''')
        aux_labels_sym = k2.SymbolTable.from_str('''
            z 1
            o 2
        ''')

        word_sym = k2.SymbolTable.from_str('''
            zoo 1
        ''')

        standard = k2.ctc_topo(max_token, modified=False)
        modified = k2.ctc_topo(max_token, modified=True)
        standard.labels_sym = labels_sym
        standard.aux_labels_sym = aux_labels_sym

        modified.labels_sym = labels_sym
        modified.aux_labels_sym = aux_labels_sym

        standard.draw('standard_topo.svg', title='standard CTC topo')
        modified.draw('modified_topo.svg', title='modified CTC topo')
        fsa = k2.linear_fst([1, 2, 2], [1, 0, 0])
        fsa.labels_sym = labels_sym
        fsa.aux_labels_sym = word_sym
        fsa.draw('transcript.svg', title='transcript')

        standard_graph = k2.compose(standard, fsa)
        modified_graph = k2.compose(modified, fsa)
        standard_graph.draw('standard_graph.svg', title='standard graph')
        modified_graph.draw('modified_graph.svg', title='modified graph')

        # z z <blk> <blk> o o <blk> o <blk>
        inputs = k2.linear_fsa([1, 1, 0, 0, 2, 2, 0, 2, 0])
        inputs.labels_sym = labels_sym
        inputs.draw('inputs.svg', title='inputs')
        standard_lattice = k2.intersect(standard_graph,
                                        inputs,
                                        treat_epsilons_specially=False)
        standard_lattice.draw('standard_lattice.svg', title='standard lattice')

        modified_lattice = k2.intersect(modified_graph,
                                        inputs,
                                        treat_epsilons_specially=False)
        modified_lattice = k2.connect(modified_lattice)
        modified_lattice.draw('modified_lattice.svg', title='modified lattice')

        # z z <blk> <blk> o o o <blk>
        inputs2 = k2.linear_fsa([1, 1, 0, 0, 2, 2, 2, 0])
        inputs2.labels_sym = labels_sym
        inputs2.draw('inputs2.svg', title='inputs2')
        standard_lattice2 = k2.intersect(standard_graph,
                                         inputs2,
                                         treat_epsilons_specially=False)
        standard_lattice2 = k2.connect(standard_lattice2)
        # It's empty since the topo requires that there must be a blank
        # between the two o's in zoo
        assert standard_lattice2.num_arcs == 0
        standard_lattice2.draw('standard_lattice2.svg',
                               title='standard lattice2')

        modified_lattice2 = k2.intersect(modified_graph,
                                         inputs2,
                                         treat_epsilons_specially=False)
        modified_lattice2 = k2.connect(modified_lattice2)
        modified_lattice2.draw('modified_lattice2.svg',
                               title='modified lattice2')

    def test_no_repeated(self):
        # standard ctc topo and modified ctc topo
        # should be equivalent if there are no
        # repeated neighboring symbols in the transcript
        max_token = 3
        standard = k2.ctc_topo(max_token, modified=False)
        modified = k2.ctc_topo(max_token, modified=True)
        transcript = k2.linear_fsa([1, 2, 3])
        standard_graph = k2.compose(standard, transcript)
        modified_graph = k2.compose(modified, transcript)

        input1 = k2.linear_fsa([1, 1, 1, 0, 0, 2, 2, 3, 3])
        input2 = k2.linear_fsa([1, 1, 0, 0, 2, 2, 0, 3, 3])
        inputs = [input1, input2]
        for i in inputs:
            lattice1 = k2.intersect(standard_graph,
                                    i,
                                    treat_epsilons_specially=False)
            lattice2 = k2.intersect(modified_graph,
                                    i,
                                    treat_epsilons_specially=False)
            lattice1 = k2.connect(lattice1)
            lattice2 = k2.connect(lattice2)

            aux_labels1 = lattice1.aux_labels[lattice1.aux_labels != 0]
            aux_labels2 = lattice2.aux_labels[lattice2.aux_labels != 0]
            aux_labels1 = aux_labels1[:-1]  # remove -1
            aux_labels2 = aux_labels2[:-1]
            assert torch.all(torch.eq(aux_labels1, aux_labels2))
            assert torch.all(torch.eq(aux_labels2, torch.tensor([1, 2, 3])))

    def test_with_repeated(self):
        max_token = 2
        standard = k2.ctc_topo(max_token, modified=False)
        modified = k2.ctc_topo(max_token, modified=True)
        transcript = k2.linear_fsa([1, 2, 2])
        standard_graph = k2.compose(standard, transcript)
        modified_graph = k2.compose(modified, transcript)

        # There is a blank separating 2 in the input
        # so standard and modified ctc topo should be equivalent
        input = k2.linear_fsa([1, 1, 2, 2, 0, 2, 2, 0, 0])
        lattice1 = k2.intersect(standard_graph,
                                input,
                                treat_epsilons_specially=False)
        lattice2 = k2.intersect(modified_graph,
                                input,
                                treat_epsilons_specially=False)
        lattice1 = k2.connect(lattice1)
        lattice2 = k2.connect(lattice2)

        aux_labels1 = lattice1.aux_labels[lattice1.aux_labels != 0]
        aux_labels2 = lattice2.aux_labels[lattice2.aux_labels != 0]
        aux_labels1 = aux_labels1[:-1]  # remove -1
        aux_labels2 = aux_labels2[:-1]
        assert torch.all(torch.eq(aux_labels1, aux_labels2))
        assert torch.all(torch.eq(aux_labels1, torch.tensor([1, 2, 2])))

        # There are no blanks separating 2 in the input.
        # The standard ctc topo requires that there must be a blank
        # separating 2, so lattice1 in the following is empty
        input = k2.linear_fsa([1, 1, 2, 2, 0, 0])
        lattice1 = k2.intersect(standard_graph,
                                input,
                                treat_epsilons_specially=False)
        lattice2 = k2.intersect(modified_graph,
                                input,
                                treat_epsilons_specially=False)
        lattice1 = k2.connect(lattice1)
        lattice2 = k2.connect(lattice2)
        assert lattice1.num_arcs == 0

        # Since there are two 2s in the input and there are also two 2s
        # in the transcript, the final output contains only one path.
        # If there were more than two 2s in the input, the output
        # would contain more than one path
        aux_labels2 = lattice2.aux_labels[lattice2.aux_labels != 0]
        aux_labels2 = aux_labels2[:-1]
        assert torch.all(torch.eq(aux_labels1, torch.tensor([1, 2, 2])))


# TODO(fangjun): Add test for CUDA.

if __name__ == '__main__':
    #  TestCtcTopo.visualize_ctc_topo()
    unittest.main()
