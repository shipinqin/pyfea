import utils
from femodel import FEModel

func = lambda xi: [(1-xi[1]*(2*xi[0]+xi[1]))/4, (1-xi[0]*(xi[0]+2*xi[1]))/4]

print(func([0.5, 0.5]))