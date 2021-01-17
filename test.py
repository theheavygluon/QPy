from QPy import *
from QPy import psiTools as psi 

print(psi.normalize(lambda x,y,z: np.sin(PI*x)*np.sin(PI*y)*np.sin(PI*z), [0,0,0], [1,1,1]))

