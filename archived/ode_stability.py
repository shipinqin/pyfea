import numpy as np
import matplotlib.pyplot as plt

def solution_exact(t, lambda_h, d0):

    d = d0*np.exp(-lambda_h*t)

    return d

def solution_num(t, lambda_h, d1, dt, t_total, alpha):

    t = 0
    while t<=t_total:
        d2 = d1*(1-(1-alpha)*dt*lambda_h)/(1+alpha*dt*lambda_h)
        yield d2
        d1 = d2
        t += dt

lambda_h = 10
d0 = 0.5

t_total = 1

# Exact solution
t = np.linspace(0, t_total, 200)
d = solution_exact(t, lambda_h, d0)
plt.plot(t, d, label='Exact', color='k')

# numerical solution

alpha = 1
stable_time = 2/((1-2*alpha)*lambda_h)

for n in [51, 21, 11, 7, 6, 5]:
    t = np.linspace(0, t_total, n)
    dt = t[1]-t[0]
    # d = [solution_num(t, lambda_h, d0, dt, alpha) for i in t]

    d = [d0] + list(solution_num(t, lambda_h, d0, dt, t[-1], alpha))
    plt.plot(t, d[:len(t)], label=f'dt={dt}', linestyle='--')

plt.text(0.6, 0.4, f'Stable time: {stable_time}')


plt.xlabel('t')
plt.ylabel('Displacement')
plt.legend()
plt.show()