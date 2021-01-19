from QPy import psiTools as psi
from QPy import *

def wf(x):
    return np.sin(PI*x)
a = psi.normalize(wf, 0.5, 0.5)

