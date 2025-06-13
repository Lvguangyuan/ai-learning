from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()