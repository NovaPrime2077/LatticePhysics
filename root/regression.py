import numpy as np
import copy,math

def cost_function(x,y,w,b):
    m = x.shape[0]
    f_wb = np.dot(x, w) + b  # Predictions for all samples
    cost = np.sum((f_wb - y) ** 2) / (2 * m)  # Vectorized cost computation
    return cost

def gradient(x,y,w,b):
    m = x.shape[0]
    f_wb = np.dot(x, w) + b 
    error = f_wb - y 
    dj_dw = np.dot(x.T, error) / m 
    dj_db = np.sum(error) / m 
    return dj_dw, dj_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iteration,cost_function,gradient):
    J_hist = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(0,num_iteration):
        dj_dw,dj_db = gradient(x,y,w,b)

        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        if(i%100==0):
            J_hist.append(cost_function(x,y,w,b))
    return w,b,J_hist

def regression_func(x, y,iterations, alpha):
    w = np.random.randn(len(x[0]))
    b_in = 0
    w_final,b_final,hist = gradient_descent(x,y,w,b_in,alpha,iterations,cost_function,gradient)
    return w_final, b_final,hist
