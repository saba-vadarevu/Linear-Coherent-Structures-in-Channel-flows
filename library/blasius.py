import scipy as sp
from scipy.integrate import odeint


def odefun(F,x):
	f1,f2,f3 = F
	return [f2,
		f3,
		-0.5*f1*f3]



def blasius(eta):
	F0 = [0,0,0.33206]
	sol = odeint(odefun, F0, eta)
	U = sol[:,1]
	dU = sol[:,2]
	return U,dU

