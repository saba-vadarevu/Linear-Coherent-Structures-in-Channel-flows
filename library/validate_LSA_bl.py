from pseudo import *
import matplotlib.pyplot as plt
import ops


flowDict = {'a':0.5/1.72, 'b':0.1/1.72, 'Re':800./1.72}
N = 151
linOps = ops.linearize(N=N,flowClass="bl",Y=15.)
evals, evecs = linOps.eig(linOps.OSS(**flowDict), weighted=True)

omega = -1.j*evals
c = omega/flowDict['a']

cSH = np.array([0.391929-0.043498j, 0.481315-0.139048j, 0.281945-0.264561j, 0.641946-0.290094j , 0.518690-0.353818j, 0.815075-0.378080j, 0.729010-0.420048j,
	0.18924427-0.109716j, 0.33172068-0.190194j, 0.449629-0.253847j, 0.555771-0.3070491j, 0.56631838-0.351969j, 0.751205-0.389606j, 0.845408-0.420542j])

plt.scatter(cSH.real, cSH.imag,marker="x")
plt.scatter(c.real, c.imag,marker="+")
plt.xlabel('$c_r$',fontsize=15)
plt.ylabel('$c_i$',fontsize=15)
plt.xlim([0.15,0.9])
plt.ylim([-0.5,0.])
plt.title('Low-speed eigenvalues for a boundary-layer, \n with a=0.5,b=0.1,Re=800 using N=%d'%N,fontsize=15)
plt.legend(('Schmid & Henningson', 'Current work'))
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.show()
plt.savefig('BL_evals1.pdf',format='pdf')


flowDict = {'a':0.25/1.72, 'b':0.2/1.72, 'Re':800./1.72}
N = 151
linOps = ops.linearize(N=N,flowClass="bl",Y=15.)
evals, evecs = linOps.eig(linOps.OSS(**flowDict), weighted=True)

omega = -1.j*evals
c = omega/flowDict['a']

cSH = np.array([0.3907+0.0029j, 0.5477-0.2343j, 0.3387-0.3101j, 0.7918-0.3787j, 0.6575-0.4051j, \
	0.2387-0.1377j, 0.4190-0.2375j, 0.5702-0.3136j, 0.7089-0.3734j, 0.8426-0.4194j])

plt.scatter(cSH.real, cSH.imag,marker="x")
plt.scatter(c.real, c.imag,marker="+")
plt.xlabel('$c_r$',fontsize=15)
plt.ylabel('$c_i$',fontsize=15)
plt.xlim([0.15,0.9])
plt.ylim([-0.5,0.])
plt.title('Low-speed eigenvalues for a boundary-layer, \n with a=0.25,b=0.2,Re=800 using N=%d'%N,fontsize=15)
plt.legend(('Schmid & Henningson', 'Current work'))
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.show()
plt.savefig('BL_evals2.pdf',format='pdf')


