from matplotlib import pyplot as plt 
import numpy as np 
import scipy
from scipy import integrate
from scipy.integrate import quad

PLANCK = 6.6*(10**(-34))
C = 299792458
E = 2.71828
KAPPA = 1.38064852*(10**(-23))
PI = np.pi 
inf = np.inf

def sin(x):
    return np.sin(x)
def cos(x):
    return np.cos(x)
def tan(x):
    return np.tan(x)
def sec(x):
    return np.sec(x)
def csc(x):
    return 1/sin(x)
def cot(x):
    return 1/tan(x)
def exp(x):
    return E**(x)
def e(x):
    return E**(x)



#Algebraic Solver
#Things to work on
#blackBody (verify eqn.)
#Compton 



class Solver():
    class photoelectric():
        def energy(f,phi):
            energy = PLANCK*f - phi
            return energy
    
        def frequency(e,phi):
            freq = (e + phi)/PLANCK
            return freq
    
        def bindingE(e,phi):
            workfunc = PLANCK*f - e
            return workfunc

        def threshold(phi):
            thresh = phi/PLANCK
            return thresh

    class deBroglie():
        def wavelength(p):
            lamda = PLANCK/p
            return lamda
        def momentum(lamda):
            p = PLANCK/lamda
            return p

    class blackBody():
        def intensity(freq,temp):
            u = (2*PI*PLANCK*(freq**5))/((C**3)*(E**((PLANCK*freq)/(KAPPA*temp))-1))
            return u

'''
def eigsTry(potential, E_level):
    from back import numerovEigs as algo
    sol = algo.Numerov(potential,E_level)
    return sol
'''


#Algebraic Plotter
#Things to work on: 
#Finish blackBody(). Fix the rJean's law, do wein, do all. Write equations for rJeans and Wein
#Compton Wavelength 
                                                                                                                                   
class Plotter():
    class photoelectric():        
        
        def energy(phi,lim1,lim2, title="Dragonflycatboi"):
            freq = [f for f in np.linspace(lim1,lim2, 10)]
            energy = [PLANCK*f - phi for f in freq]
            plt.plot(freq,energy, label='KE = hf - ' + str(int(phi)))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Kinetic Energy (Jouls)')
            if title=="Dragonflycatboi":
                plt.title("Kinetic Energy vs. Freq when \u03C6 = " + str(phi)+ "\n in range " + str(lim1) + " to " + str(lim2))
            else:
                plt.title(title)
            plt.legend()
            return plt.show()

        def frequency(phi,lim1,lim2, title='Dragonflycatboi'):
            freq = [f for f in np.linspace(lim1,lim2, 10)]
            energy = [PLANCK*f - phi for f in freq]
            plt.plot(energy,freq, label='KE = hf - ' + str(int(phi)))
            plt.xlabel('Kinetic Energy (Jouls)')
            plt.ylabel('Frequency (Hz)')
            if title=="Dragonflycatboi":
                plt.title("Freq.  vs. Kinetic Energy when \u03C6 = " + str(phi)+ "\n in range " + str(lim1) + " to " + str(lim2))
            else:
                plt.title(title)
            plt.legend()
            return plt.show()

    class deBroglie():
        def momentum(lim1,lim2, title='Dragonflycatboi'):
            lamda = [i for i in np.linspace(lim1,lim2, (lim2-lim1)*5000)]
            p = [PLANCK/i for i in lamda]
            plt.plot(lamda,p, label='p = h/\u03BB')
            plt.xlabel('Wavelength (m)')
            plt.ylabel('Momentum N.s')
            if title=="Dragonflycatboi":
                plt.title("Momentum vs. Wavelength from " + str(lim1) +  " to " + str(lim2))
            else:
                plt.title(title)
            plt.legend()
            return plt.show()

        def wavelength(lim1, lim2, title='Dragonflycatboi'):
            lamda = [i for i in np.linspace(lim1,lim2, (lim2-lim1)*500)]
            p = [PLANCK/i for i in lamda]
            plt.plot(lamda,p, label='\u03BB = h/p')
            plt.ylabel('Wavelength (m)')
            plt.xlabel('Momentum N.s')
            if title=="Dragonflycatboi":
                plt.title("Wavelength vs. Momentum from " + str(lim1) +  " to " + str(lim2))
            else:
                plt.title(title)
            plt.legend()
            return plt.show()
        

    def blackBody(temp,lim1=0,lim2="default", law="planck", title='Power Density Distribution vs Wavelength'):
            
        if law == 'planck':

            if lim2 == "default":
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(lim1, np.mean(temp)/2.5,500000)]
                        y = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*temp[n])) - 1))*10**(-13) for i in x]
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, ((np.mean(temp)/2.5)),500000)]
                    y = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*temp)) - 1))*10**(-13) for i in x]
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")
            else:
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(lim1, lim2,500000)]
                        y = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*temp[n])) - 1))*10**(-13) for i in x]
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, lim2, 500000)]
                    y = [(2*PI*PLANCK*((10**9)*C/i)**5)/(C**3*(E**((PLANCK*(10**9)*(C/i))/(KAPPA*temp)) - 1))*10**(-13) for i in x]
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")

        if law == 'wein':

            if lim2 == "default":
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(lim1, np.mean(temp)/2.5,500000)]
                        y = [2*i for i in x]
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, ((np.mean(temp)/2.5)),500000)]
                    y = [2*i for i in x]
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")
            else:
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(lim1, lim2,500000)]
                        y = [2*i for i in x]
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, lim2, 500000)]
                    y = [2*i for i in x]
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")


        if law == 'rJeans':

            if lim2 == "default":
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(0.0001, np.mean(temp)/2500,500000)]
                        y = [2*PI*(((10**9)*C/i)**4)*KAPPA*temp[n]/(C**3) for i in x]
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, ((np.mean(temp)/2500)),500000)]
                    y = [2*PI*(((10**9)*C/i)**4)*KAPPA*temp/(C**3) for i in x]
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")
            else:
                if type(temp) == list:
                    n = 0
                    while n < len(temp):
                        x = [i for i in np.linspace(lim1, lim2,500000)]
                        y = [2*PI*(((10**9)*C/i)**4)*KAPPA*temp[n]/(C**3) for i in x]
                        plt.plot(x,y, label='Temp = ' + str(temp[n]) + "K")
                        n += 1 
                else:
                    x = [i for i in np.linspace(lim1, lim2, 500000)]
                    y = [2*PI*(((10**9)*C/i)**4)*KAPPA*temp/(C**3) for i in x]
                    plt.plot(x,y, label='Temp = ' + str(temp) + "K")

        
        
        return plt.title(title), plt.xlabel("Wavelength (nm)"), plt.ylabel("Power Density (10^13)"), plt.legend(), plt.show()

#Time Independent Schrodinger Equation
#Things to do
#Make specific functions like "infSquare()" and "Constant Potential"

class tise:
    def solve(potential, E_level, alg='numerov'):
        if alg.lower() == "numerov":
            from back import numerovMethod as algo
            sol = algo.Numerov(potential,E_level)
        return sol

    
    def harmOsc(k,n, alg='numerov'):
        if alg.lower() == "numerov":
            from back import numerovMethod as algo
            sol = algo.Numerov("0.5*" + str(k) + "*x**2", n)
            return sol


    def eigs(potential, E_level, showWork=False):
        from back import numerovMethod as algo
        sol = algo.eigs(potential,E_level, showWork)
        return sol



def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def nParam(func):
    val = func.__code__.co_argcount
    return val 

#Wavefunction features
#Ideas: Normalize (), Find probability between 2 points, Operators, Expectation 
#For multi-dim normalization, change the architecture in a way where the conditions are based on the number of variables in the lambda function 
#Add useless variables instead of list

#To do list: Normalize complex functions, expectation values, sigma and shit 

class psiTools():
    

    def normalize(psi, x1 = -inf, x2 = inf, y1 = -inf, y2 = inf, z1 = -inf, z2 = inf, funcType='real'):
        if funcType.lower() == 'real':
            if nParam(psi) == 1:
                a = integrate.quad(lambda x: psi(x)**2, x1, x2)
                return 1/np.sqrt(float(a[0]))
            if nParam(psi) == 2:
                a = integrate.dblquad(lambda x,y: psi(x,y)**2, x1,x2,y1,y2)
                return 1/np.sqrt(float(a[0]))
            if nParam(psi) == 3: 
                a = integrate.tplquad(lambda x,y,z: psi(x,y,z)**2, x1,x2,y1,y2,z1,z2)
                return 1/np.sqrt(float(a[0]))
        if funcType.lower() == 'complex':
            if nParam(psi) == 1:
                a = integrate.quad(lambda x: psi(x)**2, x1, x2)
                return 1/np.sqrt(float(a[0]))
            if nParam(psi) == 2:
                a = integrate.dblquad(lambda x,y: psi(x,y)**2, x1,x2,y1,y2)
                return 1/np.sqrt(float(a[0]))
            if nParam(psi) == 3: 
                a = integrate.tplquad(lambda x,y,z: psi(x,y,z)**2, x1,x2,y1,y2,z1,z2)
                return 1/np.sqrt(float(a[0]))


    def prob(psi,lBound, rBound, lNorm=-inf, rNorm=inf):
        if nParam(psi) == 1:
            b = psiTools.normalize(lambda x: psi(x), lNorm,rNorm)
        if nParam(psi) == 2:
            b = psiTools.normalize(lambda x,y: psi(x,y), lNorm, rNorm)
        if nParam(psi) == 3:
            b = psiTools.normalize(lambda x,y,z: psi(x,y,z), lNorm, rNorm)
        if nParam(psi) == 1:
            a = quad(lambda x: (b*psi(x))**2,lBound, rBound)
        if nParam(psi) == 2: 
            a = integrate.dblquad(lambda x,y: (b*psi(x,y))**2,lBound[0], rBound[0], lBound[1], rBound[1])
        if nParam(psi) == 3:
            a = integrate.tplquad(lambda x,y,z: (b*psi(x,y,z))**2,lBound[0], rBound[0], lBound[1], rBound[1], lBound[2], rBound[2])
        return a[0]


    
