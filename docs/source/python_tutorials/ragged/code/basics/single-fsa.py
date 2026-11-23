#!/usr/bin/env python3
import k2

s = """
0 1 1 0.1
0 1 2 0.2
1 2 -1 0.3
2
"""
fsa = k2.Fsa.from_str(s)
print(fsa.arcs)
"""
[ [ 0 1 1 0.1 0 1 2 0.2 ] [ 1 2 -1 0.3 ] [ ] ]
"""

sym_str = """
a 1
b 2
"""

#  fsa.labels_sym = k2.SymbolTable.from_str(sym_str)
#  fsa.draw("images/simple-fsa.svg")
#  print(k2.to_dot(fsa))
