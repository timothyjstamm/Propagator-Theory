import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, cm
from mpl_toolkits.mplot3d import Axes3D

def greens(r_i,r_j, s = 10):

    p = 1/(s*np.sqrt(2*np.pi))*np.exp(-np.inner((r_i - r_j),(r_i - r_j))/(2*(s**2)))

    return p


def Rho(x,y):

    #rh = np.cos(x**2 + y**2)/(x**2 + y**2 + 0.1)
    #rh = 1/(x**1 + y**2 + 0.00001)
    rh = x**2 + y**2 - 1
    return rh

def prop(r_i,rho, L, N):

    #r_i is 2d np array vector
    #rho is 2d np.array matrix

    Lx = L
    Ly = L

    dx = Lx/N
    dy = Ly/N

    xlist = np.arange(0,L,dx)
    ylist = np.arange(0, L, dy)
    #rj = np.array([xlist,ylist])
    i = 0
    rho_dt = 0
    g = 0
    while i < N:
        j = 0
        while j < N:
            r_j = np.array([xlist[i],ylist[j]])
            rho_dt += greens(r_i,r_j)*rho[i,j]*dx*dy
            g += greens(r_i,r_j)*dx*dy
            j +=1
        i +=1

    rho_dt = rho_dt/g #Normalize
    return rho_dt

#x = np.arange(0,1.1,0.1)

def init_rho(L, N):

    #L = 1
    #N = 1000
    Lx = L
    Ly = L

    dx = Lx / N
    dy = Ly / N

    xlist = np.arange(0, L, dx)
    ylist = np.arange(0, L, dy)
    rho = np.zeros((N,N))
    i = 0
    for x in xlist:
        j = 0
        for y in ylist:
            rho[i][j] = Rho(x,y)
            #print(i)
            #print(j)
            j += 1
        i += 1

    return rho

def make_rho_dt(rho,L,N):

    Lx = L
    Ly = L

    dx = Lx / N
    dy = Ly / N

    xlist = np.arange(0, L, dx)
    ylist = np.arange(0, L, dy)
    rho_dt = np.zeros((N, N))
    i = 0
    for x in xlist:
        print(i)
        j = 0
        for y in ylist:
            r_i = np.array([x,y])
            rho_dt[i][j] = prop(r_i, rho, L,N)
            # print(i)
            print(j)
            j += 1
        i += 1

    return rho_dt

L = 10
N = 15


rho = init_rho(L,N)
plt.imshow(rho, cmap=cm.jet, origin='lower')
#plt.imshow(rho, origin='lower')
plt.show()

t = 0
T = 10

while t < T:

    rho_dt = make_rho_dt(rho,L,N)
    plt.imshow(rho_dt, cmap=cm.jet, origin='lower')
    #plt.imshow(rho_dt, origin='lower')
    plt.show()
    rho = rho_dt
    t+=1
    print(t)
    #if t%100 == 0:

     #   plt.imshow(rho_dt, cmap=cm.jet, origin='lower')
      #  plt.show()

#plt.imshow(rho_dt, cmap=cm.jet, origin='lower')
#plt.show()