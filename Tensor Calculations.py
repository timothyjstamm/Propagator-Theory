import numpy as np
import matplotlib.pyplot as plt


def diffusion_1D(L,N,s):

    #For a length of L with N samples make the tensor corresponding to Gaussian propagators of deviation s

    dx = L/N

    G = np.zeros((N,N))
    norm = 0
    i = 0

    while i < N:
        j = 0
        while j < N:
            #G[i][j] = (dx/(s*np.sqrt(2*np.pi)))*np.exp(-0.5*((dx/s)**2)*((i-j)**2))
            k = dx*np.exp(-0.5 * ((dx / s) ** 2) * ((i - j) ** 2))
            G[i][j] = k
            norm += k
            j+=1

        G[i] = G[i]/norm
        i += 1
        norm = 0

    return G

def Rho(x):

    rh = np.sin(x)
    #rh = np.exp(x)
    #rh = np.cos(x) + np.cos(2*x) + np.cos(3*x) + np.cos(4*x)
    #rh = np.sin(x) + np.sin(2 * x) + np.sin(3 * x) + np.sin(4 * x)
    return rh

def init_rho(L, N):

    dx = L/N
    xlist = np.arange(0, L, dx)
    rh = np.zeros(xlist.size)
    i = 0
    for x in xlist:
        rh[i] = Rho(x)
        i+=1

    return rh


L = 4*np.pi
N = 350
s = 0.1

rh = init_rho(L,N)
G = diffusion_1D(L,N,s)

Y = 5

plt.plot(rh)
plt.xlim([0,N])
plt.ylim([-Y,Y])
plt.show()

t = 0
T = 1000

while t < T:

    rh2 = G.dot(rh)
    if t%30 == 0:
        plt.plot(rh2)
        plt.xlim([0, N])
        plt.ylim([-Y, Y])
        plt.show()
    rh = rh2
    t +=1

#plt.plot(rh2)
#plt.xlim([0,N])
#plt.ylim([-Y,Y])
#plt.show()


#w,v = np.linalg.eig(G)
#print(w)
#print(v)

#b = 245

#plt.plot(v[b].real)
#plt.show()
#plt.plot(v[b].imag)
#plt.show()