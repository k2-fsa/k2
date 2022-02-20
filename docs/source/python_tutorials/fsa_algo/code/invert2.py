import k2
s = '''
0 1 2 0.1
1 2 -1 0.2
2
'''
fsa = k2.Fsa.from_str(s)
fsa.aux_labels = k2.RaggedTensor('[ [10 20] [-1] ]')
inverted_fsa = k2.invert(fsa)
fsa.draw('before_invert_aux.svg',
         title='before invert with ragged tensors as aux_labels')
inverted_fsa.draw('after_invert_aux.svg', title='after invert')
