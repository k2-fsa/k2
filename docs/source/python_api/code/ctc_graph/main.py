#!/usr/bin/env python3

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

fsa = k2.ctc_graph([[1, 2, 2, 3]], modified=False)
fsa_modified = k2.ctc_graph([[1, 2, 2, 3]], modified=True)

fsa.labels_sym = isym
fsa.aux_labels_sym = osym

fsa_modified.labels_sym = isym
fsa_modified.aux_labels_sym = osym

# fsa is an FsaVec, so we use fsa[0] to visualize the first Fsa
fsa[0].draw('ctc_graph.svg',
            title='CTC graph for the string "abbc" (modified=False)')
fsa_modified[0].draw('modified_ctc_graph.svg',
                     title='CTC graph for the string "abbc" (modified=True)')
