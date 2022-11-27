#!/usr/bin/env python3

import k2

s = '''
0 1 1 4 0.1
0 1 3 5 0.2
0 1 2 3 0.3
0 2 5 2 0.4
0 2 4 1 0.5
1 2 2 3 0.6
1 2 3 1 0.7
1 2 1 2 0.8
2 3 -1 -1 0.9
3
'''
fsa = k2.Fsa.from_str(s, acceptor=False)
fsa.draw('arc_sort_single_before.svg', title='Before k2.arc_sort')
sorted_fsa = k2.arc_sort(fsa)
sorted_fsa.draw('arc_sort_single_after.svg', title='After k2.arc_sort')

# If you want to sort by aux_labels, you can use
inverted_fsa = k2.invert(fsa)
sorted_fsa_2 = k2.arc_sort(inverted_fsa)
sorted_fsa_2 = k2.invert(sorted_fsa_2)
sorted_fsa_2.draw('arc_sort_single_after_aux_labels.svg',
                  title='After k2.arc_sort by aux_labels')
