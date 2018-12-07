import numpy as np
import dataPreprocessing
import sys

instr = sys.argv[1]
filename = sys.argv[2]

gen = np.load(filename)
p = dataPreprocessing.matrix_to_midi(gen, instr)
s = dataPreprocessing.stream.Stream(p)
s.write('midi', fp='{}.midi'.format(filename.split('.')[0]))
