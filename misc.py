import numpy as np


t = np.arange(0, 10, 0.00001)


def f(t):
    pass


# oculomotor dynamics, pure fast, pure slow
xs = 0
xf = 0
tau_s = 0.224
tau_f = 0.013
ys = []
yf = []
for i in t:
    xs += 0.00001 * (-1 / tau_s * xs + 1 / tau_s * f(i))
    xf += 0.00001 * (-1 / tau_f * xf + 1 / (tau_f * tau_s) * f(i))
    # print(xf)
    ys.append(xs)
    yf.append(xf)
