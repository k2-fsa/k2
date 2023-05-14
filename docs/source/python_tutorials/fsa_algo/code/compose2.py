import k2
s1 = '''
0 1 1 1 0
1 2 2 2 0
2 3 -1 -1 0
3
'''

a_fsa = k2.Fsa.from_str(s1, acceptor=False)
b_fsa = k2.ctc_topo(max_token=2, modified=False) 
c_fsa = k2.compose(a_fsa, b_fsa, treat_epsilons_specially=False)

a_fsa.draw('a_fsa_compose2.svg', title='a_fsa')
b_fsa.draw('b_fsa_compose2.svg', title='b_fsa')
c_fsa.draw('c_fsa_compose2.svg', title='c_fsa')
