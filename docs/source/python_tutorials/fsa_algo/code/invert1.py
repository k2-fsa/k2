import k2
s = '''
0 1 2 10 0.1
1 2 -1 -1 0.2
2
'''
fsa = k2.Fsa.from_str(s, acceptor=False)
inverted_fsa = k2.invert(fsa)
fsa.draw('before_invert.svg', title='before invert')
inverted_fsa.draw('after_invert.svg', title='after invert')
