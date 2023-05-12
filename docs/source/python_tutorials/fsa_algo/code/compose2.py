import k2
s = '''
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

a_fsa = k2.ctc_topo(max_token=1) 
b_fsa = k2.Fsa.from_str(s, acceptor=False)
c_fsa = k2.compose(a_fsa, b_fsa, treat_epsilons_specially=False)

a_fsa.draw('a_fsa_compose2.svg', title='a_fsa')
b_fsa.draw('b_fsa_compose2.svg', title='b_fsa')
c_fsa.draw('c_fsa_compose2.svg', title='c_fsa')
