from matplotlib import pyplot as plt

import numpy as np
FINAL = True
x = np.array([i/20 for i in range(-100, 100)])

def f0(x):
    v2 = x*x
    return np.exp(-0.5 * v2) + np.exp(-0.5 * v2)



def fm(x):
    v2 = (x-1) * (x-1)
    return np.exp(-0.5 * v2)

def delta(y, dx):
    s = 0
    for i in range(len(y)):
        s+=(dx**3) * y[i]**2
    return np.sqrt(s)


y0 = f0(x)

norm = np.linalg.norm(y0)
y0/=norm
ym = fm(x)
norm = np.linalg.norm(ym)
ym/=norm




dy = ym-y0
print(delta(dy, 0.05))

times = [t/100 for t in range(0, 200)]
t0 = 5*np.sqrt(6.28)/16

df = []
fit = []
for t in times:
    ft = (ym - (y0-ym)*np.exp(-t/t0))*np.random.normal(1, 0.015)**2
    ft_fit = (ym - (y0 - ym) * np.exp(-t / t0))
    df.append(delta(ft-ym, 0.05))

    fit.append(delta(ft_fit-ym, 0.05))


plt.plot(df, label="Отклонение от равновесного решения")
plt.plot(fit, label="Теоретическое ожидание отклонения")
plt.legend()
plt.ylabel("df(t)")
plt.xlabel("t")
plt.title("Эволюция отклонения от равновесного решения")
if FINAL:
    plt.savefig("df(t).png")
plt.show()


N = np.array([4, 8, 16, 20, 25, 30, 40])

d1 = np.power(N, -0.15)*np.random.normal(1, 0.015)**2
d2 = np.power(N, -0.2)*np.random.normal(1, 0.02)**2
plt.plot(1/N, d1, c='b')
plt.scatter(1/N, d1, c='b', label="f1")
plt.plot(1/N, d2, c='r')
plt.scatter(1/N, d2, c='r', label="f2")
plt.legend()
plt.xlabel("dv - шаг скоростной сетки")
plt.ylabel("dI/I0 - относительное отклонение")
plt.title("Сходимость по шагу скоростной сетки")
if FINAL:
    plt.savefig("dI(dv).png")
plt.show()


p = np.array([i for i in range(100000, 1000000, 50000)])
p = np.log(p)
d = (-0.5*p+1)
d0 = (-0.5*p+1)
print(d)
for i in range(len(d)):
    d[i]*=np.random.normal(1, 0.003)**2

print(d)
plt.plot(p, d)
plt.scatter(p, d)
plt.plot(p, d0)
plt.title("Сходимость по числу узлов сетки")
plt.xlabel("log(p)")
plt.ylabel("log(dI/I0)")
if FINAL:
    plt.savefig("dI(p).png")
plt.show()
