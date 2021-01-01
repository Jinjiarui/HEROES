import os

path = './../alicpp/'
mod = 'test'
with open(os.path.join(path, mod, 'sample_skeleton_%s.csv' % (mod)), 'r', encoding='utf8') as f:
    with open(os.path.join(path, mod, 'small_sample_skeleton_%s.csv' % (mod)), 'w', encoding='utf8') as opf:
        for i in range(1000000):
            opf.write(f.readline())
