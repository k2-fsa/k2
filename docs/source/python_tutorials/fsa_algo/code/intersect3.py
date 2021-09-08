import k2
s1 = '''
0 1 0 0.1
0 1 1 0.2
1 1 2 0.3
1 2 -1 0.4
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
b_fsa = k2.add_epsilon_self_loops(b_fsa)
c_fsa = k2.intersect(a_fsa, b_fsa, treat_epsilons_specially=False)

a_fsa.draw('a_fsa_intersect3.svg', title='a_fsa')
b_fsa.draw('b_fsa_intersect3.svg', title='b_fsa')
c_fsa.draw('c_fsa_intersect3.svg', title='c_fsa')
