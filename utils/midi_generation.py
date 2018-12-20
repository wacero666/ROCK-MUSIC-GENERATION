import numpy as np
import data_preprocessing
import sys

instr = sys.argv[1]
filename = sys.argv[2]

gen = np.load(filename)
p = data_preprocessing.matrix_to_midi(gen, instr)
s = data_preprocessing.stream.Stream(p)
s.write('midi', fp='{}.midi'.format(filename.split('.')[0]))
