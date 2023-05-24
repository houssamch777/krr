import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) # for reproducibility

napp = 500
ntest = 500
x1 = 2*np.random.rand(2, napp) - 1
y1 = np.sign(x1[0])
xt1 = 2*np.random.rand(2, ntest) - 1
yt1 = np.sign(xt1[0])
plt.figure(1)
plt.plot(x1[0, y1==-1], x1[1, y1==-1], 'rx',label="rx")
plt.plot(x1[0, y1==1], x1[1, y1==1], 'bx')
plt.title('Jeu de données 1')
plt.legend()

x2 = np.concatenate([1.5*np.random.randn(2, napp//2) - 0.5, 0.7*np.random.randn(2, napp//2) + 1.5], axis=1)
y2 = np.concatenate([np.ones(napp//2), -np.ones(napp//2)])
xt2 = np.concatenate([1.5*np.random.randn(2, ntest//2) - 0.5, 0.7*np.random.randn(2, ntest//2) + 1.5], axis=1)
yt2 = np.concatenate([np.ones(ntest//2), -np.ones(ntest//2)])
plt.figure(2)
plt.plot(x2[0, y2==-1], x2[1, y2==-1], 'rx')
plt.plot(x2[0, y2==1], x2[1, y2==1], 'bx')
plt.title('Jeu de données 2')

sigma = 0 # increase to mix at the boundaries
nb = napp//16
x3, y3 = [], []
xt3, yt3 = [], []
for i in range(-2, 2+1):
    for j in range(-2, 2+1):
        x3.append([i + (1+sigma)*np.random.rand(nb), j + (1+sigma)*np.random.rand(nb)])
        y3.append((2*((i+j+4)%2)-1)*np.ones(nb))

x3, y3 = np.concatenate(x3, axis=1), np.concatenate(y3)
nb = ntest//16
for i in range(-2, 2+1):
    for j in range(-2, 2+1):
        xt3.append([i + (1+sigma)*np.random.rand(nb), j + (1+sigma)*np.random.rand(nb)])
        yt3.append((2*((i+j+4)%2)-1)*np.ones(nb))
xt3, yt3 = np.concatenate(xt3, axis=1), np.concatenate(yt3)
plt.figure(3)
plt.plot(x3[0, y3==-1], x3[1, y3==-1], 'rx')
plt.plot(x3[0, y3==1], x3[1, y3==1], 'bx')
plt.title('Jeu de données 3')
plt.show()
