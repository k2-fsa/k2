import k2
s = '''
0 2 -1  0.0
0 1 2 0.2
0 1 1 0.3
1 2 -1 0.4
2
'''
fsa = k2.Fsa.from_str(s)
sorted_fsa = k2.arc_sort(fsa)
fsa.draw('before_sort.svg', title='before sort')
sorted_fsa.draw('after_sort.svg', title='after sort')
