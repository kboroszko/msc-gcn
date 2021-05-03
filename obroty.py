from matplotlib import pyplot as plt
import numpy as np



def plotRing(V, new_fig=True, title="a", label="a"):
    if new_fig:
        plt.figure(figsize=(5,5))
        plt.title(title)
        xmin = np.min(V[:,0])
        xmax = np.max(V[:,0])
        xspread = xmax - xmin
        ymin = np.min(V[:,1])
        ymax = np.max(V[:,1])
        yspread = ymax - ymin
        # plt.xlim([xmin - 0.1 * xspread, xmax + 0.1*xspread])
        # plt.ylim([ymin - 0.1 * yspread, ymax + 0.1*yspread])
    plt.scatter(V[:,0], V[:,1], label=label)

    plt.axis('equal')
#%%
N = 100
Vrt = np.zeros((N, 2))
Vrt[:,0] = 1
Vrt[:,1] = np.arange(N)*2*np.pi/N

Vxy = np.zeros((N,2))
Vxy[:,0] = Vrt[:,0]*np.cos(Vrt[:,1])
Vxy[:,1] = Vrt[:,0]*np.sin(Vrt[:,1])
#%%
plotRing(Vxy, title="normal")

#%%

M = np.array([
    [2., 0.7],
    [0.3, 0.8]
], dtype=np.float)

W = np.matmul( Vxy, M)

#%%
plotRing(Vxy, label="normal")
#%%
plotRing(W, new_fig=False, label="rotated")
#%%

M2 = np.array([
    [0.5, -np.sqrt(3)/2],
    [np.sqrt(3)/2, 0.5]
], dtype=np.float)

Wp = np.matmul( W, M2)

plotRing(Wp, new_fig=False, label="rotated2")
plt.legend()
#%%

def plotVectors(Vs, col='k'):
    ax = plt.axes()
    for v in Vs:
        ax.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.1, fc=col, ec=col)
        plt.show()

#%%
Vectors = np.array([
    [1,0],
    [0,1],
])

plotRing(Vxy, label="normal")
plotVectors(Vectors, col='k')
#%%
s = np.expand_dims(np.sum(Vectors, axis=1),0)
plotVectors(s, col='r')

#%%
plotRing(W, new_fig=False, label="rotated")
Vec2 = np.matmul(Vectors, M)
plotVectors(Vec2, col='g')
s2 = np.expand_dims(np.sum(Vec2, axis=1),0)
plotVectors(s2, col='b')

#%%
def nonl(v):
    x,y = v
    r = np.sqrt(x**2 + y**2)
    t = np.arctan(y/(x+1e-10))
    alpha = np.sin(t*4)
    nr = r + 0.2*r*alpha
    nx = nr*np.cos(t)
    ny = nr*np.sin(t)
    return nx,ny



