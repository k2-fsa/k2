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

fsa = k2.ctc_topo(max_token=3, modified=False)
fsa_modified = k2.ctc_topo(max_token=3, modified=True)

fsa.labels_sym = isym
fsa.aux_labels_sym = osym

fsa_modified.labels_sym = isym
fsa_modified.aux_labels_sym = osym

fsa.draw('ctc_topo.svg',
         title='CTC topology with max_token=3 (modified=False)')
fsa_modified.draw('modified_ctc_topo.svg',
                  title='CTC topology with max_token=3 (modified=True)')
