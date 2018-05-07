# Modules
Impulse response is computed using the following 4 modules, going from low level to high level:

## pseudo.py
Functions relating to wall-normal differentiation using Chebyshev nodes for channel geometry
Also, semi-infinite domain using exponential mapping


## ops.py
Uses pseudo.py. Defines two classes

### linearize
Framework for linearized flows, currently supporting only channel flows
Extensible to Couette flows; boundary layer is also possible, but needs more effort 
Class instances initialized using flowClass (channel, couette, bl), flowState (lam, turb), Re (interpreted to be ReTau for turb), N (no. of nodes); optionally U, dU, d2U can be supplied 
Class method makeSystem() takes wavenumbers and 'eddy' (=True/False) to create the system matrices A, C, and B, and their adjoints if requested. 

### statComp
For the statistics completion of Zare et. al. 2017, Colour of Turbulence
This class also uses functions from minimize.py


## impulseResponse.py
Uses the linearize function in ops.py. Defines the function timeMap() and H2 norm() among others. 
timeMap() takes a wavenumber pair and a set of times, along with parameters like flowClass, flowState, Re, etc., and returns the energy in this Fourier mode at the requested times. The Fourier coefficients are also returned (by setting timeMap(coeffs=True)), but the default is to not return these coefficients. 
H2norm() returns integral of energy over t>0. 


## flowField.py
Contains two parts: the class 'flowField' and functions related to computing impulse response and populating flowField class instances.

### flowField
A 4-d numpy.ndarray with additional functionality, storing Fourier coefficients at a specific time 't'.
The shape of the array is (n_kx, n_kz, 3, N), for streawise wavenumber, spanwise wavenumber, velocity component, and wall-normal location respectively. 
In principle, any method of numpy.ndarray can be applied on these arrays, but this is not recommended as it could destroy functionality. 
Contains class methods for operations such as differentiation along all 3 directions, computing swirl from velocity, and producing the physical flowfield. 

### impulseResponse(), impulseResponse_split()
Functions that compute impulse response, using impulseResponse.timeMap(), and populate the flowField class instance. For large cases (with a lot of wavenumbers), this task is split into smaller chunks using impulseResponse_split()



# Example usage
See ipython notebook 'exampleScripts.py'

# Important note
The code is written for python version >= 3.5. 
This restriction primary stems from usage of the infix operator '@' for matrix multiplication as A@B instead of np.dot(A,B). 
