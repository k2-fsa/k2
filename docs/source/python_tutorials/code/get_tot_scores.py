import k2
s = '''
0 1 1 1.2
0 1 3 0.8
0 2 2 0.5
1 2 5 0.1
1 3 -1 0.6
2 3 -1 0.4
3
'''
fsa = k2.Fsa.from_str(s)
fsa.draw('get_tot_scores.svg', title='get_tot_scores example')
fsa_vec = k2.create_fsa_vec([fsa])
log_semiring = fsa_vec.get_tot_scores(use_double_scores=True,
                                      log_semiring=True)
tropical_semiring = fsa_vec.get_tot_scores(use_double_scores=True,
                                           log_semiring=False)
print('get_tot_scores for log semiring:', log_semiring)
print('get_tot_scores for tropical semiring:', tropical_semiring)
