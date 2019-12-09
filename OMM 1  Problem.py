from math import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap  #нанести цвет на график
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

#точность, шаги по сетке
epsilon = 0.0001
N = 100; M = 1000
T_begin = 0; T_end = 1
X_begin = 0; X_end = -1

#элементарные шаги
h_x = (X_end - X_begin)/(N)
h_t = (T_end - T_begin)/(M)

#задаем двумерный массив размерами M*N с искомыми значениями
y = np.zeros((M,N))

#НУ+ГУ
for n in np.arange(N):
        y[0][n] = -sin(pi * h_x * n)

for m in np.arange(M):
        y[m][0] = exp(-h_t * m)-1

#задание дополнительных функций
def F(m,n):
    return np.arctan(exp(y[m][n]))

def df(mp1,np1):
    return 1/(2*h_t) - (exp(y[mp1][np1])/(0.5*h_x*(1+exp(2*y[mp1][np1]))))

#разностная схема
def f(mp1, np1):
    n = np1 - 1
    m = mp1 - 1
    return (y[mp1][n] - y[m][n] + y[mp1][np1] - y[m][np1]) / (2.*h_t) - (F(mp1, np1)-F(mp1,n) + F(m, np1)-F(m,n)) / (2.*h_x)

#метод Ньютона

for m in range(M-1):            #пока m в диапазоне [0,98] работаем,в конце каждой итерации неявно m=m+1
    for n in range(N-1):
        ep = 0
        eps = epsilon + 1
        y[m + 1][n + 1] = y[m, n]
        while eps > epsilon:
            ep = f(m+1, n+1) / df(m+1, n+1)
            y[m+1][n+1] = y[m+1][n+1] - ep
            print("kuku", eps)
            print(" ggg")
            eps = abs(ep)

for m in range(M):           #вывод полученной функции.
    for n in range(N):
        print("u(",m,n,")",y[m,n])
#построение решения

tm = np.linspace(T_begin,T_end, num=M)
xn = np.linspace(X_begin, X_end, num=N)

X, T = np.meshgrid(xn, tm)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(X, T, y, rstride=10,cstride=1, cmap=cm.jet)
plt.title('Решение')
plt.xlabel('x')
plt.ylabel('t')
plt.show()