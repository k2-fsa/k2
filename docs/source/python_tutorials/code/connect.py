import k2
s1 = '''
0 0 1 0.1
0 1 2 0.2
1 2 -1 0.3
2
'''
s2 = '''
0 1 1 1
0 1 2 2
1 2 -1 3
2
'''
a_fsa = k2.Fsa.from_str(s1)
b_fsa = k2.Fsa.from_str(s2)
c_fsa = k2.intersect(a_fsa, b_fsa)
connected = k2.connect(c_fsa)
a_fsa.draw('a_fsa_1.svg', title='a_fsa')
b_fsa.draw('b_fsa_1.svg', title='b_fsa')
c_fsa.draw('before_connect.svg', title='intersect(a_fsa, b_fsa)')
connected.draw('after_connect.svg', title='after connect')
