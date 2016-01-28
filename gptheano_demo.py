import sys, os
import theano
import theano.tensor as T
import theano.sandbox.linalg as sT
import numpy as np
import cPickle
import time
import matplotlib.pyplot as plt

import pdb


theano.config.mode= 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False

x_val = np.random.rand(100,5)
y_val = np.random.rand(100,1)
x_test_val = np.random.rand(50,5)

SHADEDCOLOR = [0.7539, 0.89453125, 0.62890625, 1.0]
MEANCOLOR = [ 0.2109375, 0.63385, 0.1796875, 1.0]
DATACOLOR = [0.12109375, 0.46875, 1., 1.0]


def plot_regression(x, y, xs, ym, ys2,title):
    # x: training point
    # y: training target
    # xs: test point
    # ym: predictive test mean
    # ys2: predictive test variance
    plt.figure()
    xss  = np.reshape(xs,(xs.shape[0],))
    ymm  = np.reshape(ym,(ym.shape[0],))
    ys22 = np.reshape(ys2,(ys2.shape[0],))
    plt.plot(x, y, color=DATACOLOR, ls='None', marker='+',ms=12, mew=2)
    plt.plot(xs, ym, color=MEANCOLOR, ls='-', lw=3.)
    plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=SHADEDCOLOR,linewidths=0.0)
    plt.grid()
    #if not axisvals is None:
    #    plt.axis(axisvals)
    plt.xlabel('input x')
    plt.ylabel('target y')
    plt.title(title)
    #plt.show()

   

def demo():
    demoData = np.load('regression_data.npz')
    x_val = demoData['x']            # training data
    y_val = demoData['y']            # training target
    x_test_val = demoData['xstar']        # test data
     
    from gptheano_model import GP_Theano
    initial_params = {'sigma_n':np.log(0.1), 'sigma_f':0., 'l_k':0.}
    model = GP_Theano(initial_params)
    outputs = model.get_prediction(x_val, y_val, x_test_val)   
    plot_regression(x_val, y_val, x_test_val, outputs['y_test_mu'],outputs['y_test_var'],'Before Optimization')
    
    model.train(x_val, y_val, num_epoch = 100,lr = 1e-2,decay=None,batch_size=20)
    outputs = model.get_prediction(x_val, y_val, x_test_val)   
    plot_regression(x_val, y_val, x_test_val, outputs['y_test_mu'],outputs['y_test_var'],'After Optimization')
    plt.show()

def test_case(x_val, y_val, x_test_val):
    from gptheano_model import GP_Theano
    initial_params = {'sigma_n':1.0, 'sigma_f':1.2, 'l_k':1.2}
    model = GP_Theano(initial_params)
    outputs = model.get_prediction(x_val, y_val, x_test_val)   
    
    model.train(x_val, y_val, num_epoch = 50)
    pdb.set_trace()


if __name__ == '__main__':
    #test_case(x_val, y_val, x_test_val)
    demo()
