import numpy as np
import tensorflow as tf


def fun(x, coeffs):

    res = coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]
    # res = x**2 - 3

    return res

def fun_p(x, coeffs):

    y_p = 2*coeffs[0]*x + coeffs[1]
    # y_p = 2*x

    return y_p

if __name__ == '__main__':
    coeffs = (4, 0, -1)
    toler = 1e-6

    # Newton iteration
    x, res  = 0.5, 1
    i, max_iter = 0, 30
    while abs(res) > toler and i < max_iter:
        res = fun(x, coeffs)
        x = x - res/fun_p(x, coeffs)
        print('Iteration %s: x=%.2f, res=%f'%(i,x,res))
        i += 1

    print('The coefficients are:', coeffs)
    print('The Newwton iteration solution is:', x)

    # Gradient descent
    lr = 0.01
    i, max_iter = 0, 100
    x = 0.6

    cost = (fun(x, coeffs))**2
    step = 1
    while abs(step) > toler and i < max_iter:
        step = lr*2*fun(x, coeffs)*fun_p(x, coeffs)
        x = x - step
        print('Iteration %s: x=%.2f, step=%f'%(i,x,step))
        i += 1

    print('The gradient descent solution is:', x)
