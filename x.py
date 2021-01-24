from QPy import psiTools as psit
import numpy as np 
from scipy.integrate import quad
inf = np.inf

def pos(wf, lBound = -inf, rBound = inf):
    a = psit.normalize(wf, lBound, rBound)
    pos = lambda x: a*x*wf(x)
    return pos


class x():
        
        def expVal(psi, lBound = -inf, rBound = inf):
            a = psit.normalize(psi, lBound, rBound)
            exp = quad(lambda x: (a**2)*np.conj(psi(x))*x*psi(x), lBound, rBound)
            return exp[0]
        
        def sigma(psi):
            return print(" This Feature is under Construction")




def y(wf, lBound = -inf, rBound = inf):
    a = psi.normalize(wf, lBound, rBound)
    pos = lambda y: a*y*wf(y)
    return pos

def z(wf, lBound = -inf, rBound = inf):
    a = psi.normalize(wf, lBound, rBound)
    pos = lambda z: a*z*wf(z)
    return pos


