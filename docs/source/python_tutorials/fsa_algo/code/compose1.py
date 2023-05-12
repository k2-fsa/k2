import k2
s1 = '''
0 1 2 20 0.1
0 0 1 10 0.2
1 1 3 30 0.3
1 2 -1 -1 0.4
2
'''

s2 = '''
0 1 10 1 0.1
0 1 10 2 0.2
0 1 20 3 0.3
1 2 10 1 0.3
1 2 20 2 0.2
1 2 30 3 0.1
2 3 20 1 0.2
2 3 30 2 0.1
2 3 30 3 0.3
3 4 -1 -1 0.4
4
'''

a_fsa = k2.Fsa.from_str(s1, acceptor=False)
b_fsa = k2.Fsa.from_str(s2, acceptor=False)
c_fsa = k2.compose(a_fsa, b_fsa)

a_fsa.draw('a_fsa_compose.svg', title='a_fsa')
b_fsa.draw('b_fsa_compose.svg', title='b_fsa')
c_fsa.draw('c_fsa_compose.svg', title='c_fsa')
