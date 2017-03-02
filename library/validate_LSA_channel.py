from pseudo import *
import matplotlib.pyplot as plt
import ops


flowDict = {'a':0.5, 'b':1., 'Re':2000.}
N = 81
y = chebdif(N+2,1)[0]
y = y[1:-1]
U = 1.- y**2; dU = -2.*y; d2U=-2.*np.ones(U.size)
#linOps = ops.linearize(N=N,flowClass="channel", U=U, dU=dU, d2U=d2U )
linOps = ops.linearize(N=N,flowClass="channel")

evals, evecs = linOps.eig(linOps.OSS(**flowDict), weighted=True)
#evals, evecs = linOps.eig(OSS, b=b, weighted=True)
print(evals.shape, evecs.shape, y.shape)

omega = -1.j*evals
c = omega/flowDict['a']

cSH = [0.3723-0.0374j, 0.4994-0.0992j , 0.8877-0.1095j, 0.8880-0.1096j, 0.7953-0.1933j , 0.7265-0.2610j , 0.6478-0.2697j, 0.7047-0.2987j , 0.4332-0.3066j,\
0.9776-0.0236j , 0.9329-0.0683j , 0.8882-0.1131j, 0.8435-0.1578j , 0.3123-0.1699j, 0.3123-0.1699j , 0.7987-0.2025j , 0.7536-0.2470j, 0.5347-0.2714j, 0.5347-0.2715j]

plt.scatter(np.real(cSH),np.imag(cSH),marker="x")
plt.hold(True)
plt.scatter(np.real(c[0:]),np.imag(c[0:]),marker="+")
plt.xlabel('cr')
plt.ylabel('ci')
plt.title('Least-stable eigenvalues for a channel flow, \n with a=0.5,b=1.0,Re=2000 using N=151')
plt.legend(('Schmid & Henningson', 'Current work'),loc='lower left')
plt.xlim([0.,1.5])
plt.ylim([-1.,0.2])
plt.xlabel("$c_r$"); plt.ylabel("$c_i$")

#fig = plt.gcf()
#fig.set_size_inches(6,6)
#plt.savefig('channel_evals.jpeg')
plt.show()

