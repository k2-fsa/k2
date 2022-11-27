#!/usr/bin/env python3

# Construct a CTC graph by intersection

import k2

isym = k2.SymbolTable.from_str('''
blk 0
a 1
b 2
c 3
''')

osym = k2.SymbolTable.from_str('''
a 1
b 2
c 3
''')

linear_fsa = k2.linear_fsa([1, 2, 2, 3])
linear_fsa.labels_sym = isym

ctc_topo = k2.ctc_topo(max_token=3, modified=False)
ctc_topo_modified = k2.ctc_topo(max_token=3, modified=True)

ctc_topo.labels_sym = isym
ctc_topo.aux_labels_sym = osym

ctc_topo_modified.labels_sym = isym
ctc_topo_modified.aux_labels_sym = osym

ctc_graph = k2.compose(ctc_topo, linear_fsa)
ctc_graph_modified = k2.compose(ctc_topo_modified, linear_fsa)

linear_fsa.draw('linear_fsa.svg', title='Linear FSA of the string "abbc"')
ctc_topo.draw('ctc_topo.svg', title='CTC topology')
ctc_topo_modified.draw('ctc_topo_modified.svg', title='Modified CTC topology')

ctc_graph.draw('ctc_topo_compose_linear_fsa.svg',
               title='k2.compose(ctc_topo, linear_fsa)')

ctc_graph_modified.draw('ctc_topo_modified_compose_linear_fsa.svg',
                        title='k2.compose(ctc_topo_modified, linear_fsa)')
