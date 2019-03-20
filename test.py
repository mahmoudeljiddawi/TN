# derviative tanh = 1- x^2
# sigmoid = x(1-x)
import numpy as np


def derivative_sigmoid(x):
    x = np.array(x)
    xdash = 1 - x
    return np.multiply(x,xdash)

Y1 = np.array([1, 0, 0])
Y2 = np.array([0, 1, 0])
Y3 = np.array([0, 0, 1])

Y1 = Y1.reshape((3,1))
Y2 = Y2.reshape((3,1))
Y3 = Y3.reshape((3,1))

w = np.array([[0.99999004, 0.99999336, 0.99999703, 0.99999948],
       [0.99999004, 0.99999336, 0.99999703, 0.99999948]])
i = np.array([[4.7],[3.2],[1.6],[0.2]])

output_weight = np.array([[1.01251583, 1.01251583], [0.90752169 ,0.90752169],[0.90752169 ,0.90752169]])
print('w shape ' , w.shape , 'input shape' , i.shape )
dot = np.dot(w,i)
print(dot)
act = 1 / (1 + np.exp(-dot))
print('act' , act)

Y_hat = np.dot(output_weight,act)

print('Y_hat',Y_hat)

Y_hat = 1 / (1 + np.exp(-Y_hat))

print( 'after activation:' , Y_hat)

gradient = (Y1 - Y_hat) * derivative_sigmoid(Y_hat)
print('gradient',gradient)

print('act',act)
new_gradient = derivative_sigmoid(act) * np.dot(output_weight.T, gradient)
print('n' , new_gradient)

w = w + (np.dot(new_gradient, i.T))
print('update' , w)
output_weight = output_weight + (np.dot(gradient, act.T))
print('out',output_weight)
