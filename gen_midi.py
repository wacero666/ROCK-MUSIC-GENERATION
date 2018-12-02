import numpy as np
import dataPreprocessing
import sys

instr = sys.argv[1]

gen = np.load('./try_{}.npy'.format(instr.lower()))
p = dataPreprocessing.matrix_to_midi(gen, instr)
s = dataPreprocessing.stream.Stream(p)
s.write('midi', fp='try_{}.midi'.format(instr.lower()))
