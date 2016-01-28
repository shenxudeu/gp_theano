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


def plot_regression(x, y, xs, ym, ys2):
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
    plt.show()

   

def demo():
    demoData = np.load('regression_data.npz')
    x_val = demoData['x']            # training data
    y_val = demoData['y']            # training target
    x_test_val = demoData['xstar']        # test data
     
    from gptheano_model import GP_Theano
    initial_params = {'sigma_n':np.log(0.1), 'sigma_f':0., 'l_k':0.}
    model = GP_Theano(initial_params)
    outputs = model.get_prediction(x_val, y_val, x_test_val)   
    plot_regression(x_val, y_val, x_test_val, outputs['y_test_mu'],outputs['y_test_var'])
    
    model.train(x_val, y_val, num_epoch = 100,lr = 1e-2,decay=None,batch_size=20)
    outputs = model.get_prediction(x_val, y_val, x_test_val)   
    plot_regression(x_val, y_val, x_test_val, outputs['y_test_mu'],outputs['y_test_var'])
    
    
    pdb.set_trace()

def test_case(x_val, y_val, x_test_val):
    from gptheano_model import GP_Theano
    initial_params = {'sigma_n':1.0, 'sigma_f':1.2, 'l_k':1.2}
    model = GP_Theano(initial_params)
    outputs = model.get_prediction(x_val, y_val, x_test_val)   
    
    model.train(x_val, y_val, num_epoch = 50)
    pdb.set_trace()


def test_dist_func(x1_val, x2_val):
    start = time.time()
    # naive version of distance calculation
    dist_vals = np.zeros((x1_val.shape[0],x2_val.shape[0]))
    for i in range(x2_val.shape[0]):
        dist_vals[:,i] = np.sqrt(np.sum((x1_val - x2_val[i,:])**2,axis=1))
    print '----naive version time cost  = %f\n'%(time.time()-start)

    start = time.time()
    # Matrix Factorized calculation.
    xx_val0 = np.sum(x1_val**2,axis=1).reshape(x1_val.shape[0],1)
    xc_val0 = np.dot(x1_val,x2_val.T)
    cc_val0 = np.sum(x2_val**2,axis=1).reshape(1,x2_val.shape[0])
    dist_vals_f = np.sqrt(xx_val0 - 2*xc_val0 + cc_val0)
    print '----numpy version time cost  = %f\n'%(time.time()-start)

    if np.sum(dist_vals - dist_vals_f) < 1e-5:
        print 'Numpy version of factorization calculation is correct !!'
    else:
        print 'Numpy version of factorization calculation is correct !!'


    # Theano matrix factorized calculation.
    x1,x2 = T.dmatrices('x1','x2')
    xx = T.sum(x1**2,axis=1).reshape((x1.shape[0],1))
    xc = T.dot(x1, x2.T)
    cc = T.sum(x2**2,axis=1).reshape((1,x2.shape[0]))
    dist = T.sqrt(xx - 2*xc + cc)

    fs = zip(['xx','xc','cc','dist'],
             [xx, xc, cc,dist])
    inputs = {'x1':x1,'x2':x2}
    input_vals = {'x1':x1_val, 'x2':x2_val}

    f = {n: theano.function(inputs.values(),f,name=n,on_unused_input='ignore') for n,f in fs}

    start = time.time()
    xx_val1 = f['xx'](*input_vals.values())
    xc_val1 = f['xc'](*input_vals.values())
    cc_val1 = f['cc'](*input_vals.values())
    dist_vals_t = f['dist'](*input_vals.values())
    print '----theano version time cost  = %f\n'%(time.time()-start)

    #f2 = theano.function((x1,x2),xx,name='xx',on_unused_input='ignore')
    #xx_val2 = f2(x1_val,x2_val)

    if np.sum(dist_vals - dist_vals_t) < 1e-5:
        print 'Theano version of factorization calculation is correct !!'
    else:
        print 'Theano version of factorization calculation is correct !!'



if __name__ == '__main__':
    #test_case(x_val, y_val, x_test_val)
    demo()
