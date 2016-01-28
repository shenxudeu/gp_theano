"""
Gaussian Process Implementation using Theano for symbolic gradient computation.
"""

# To speed Theano up, create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/randisk
# Then use flag THEANO_FLAGS=base_compiledir=/mnt/randisk python script.py
import sys, os
import theano
import theano.tensor as T
import theano.sandbox.linalg as sT
import numpy as np
import cPickle
import pdb

print 'Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir

theano.config.mode= 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False


def np_uniform_scalar(scale=1):
    np.random.seed(1984)
    return np.random.uniform(low=-scale,high=scale)

def shared_scalar(val=0., dtype=theano.config.floatX,name=None):
    return theano.shared(np.cast[dtype](val))


class GP_Theano(object):
    def __init__(self,
            initial_params=None):
        print 'Setting up variables ...'
        # Parameters
        if initial_params is None:
            initial_params = {'sigma_n':0.+np_uniform_scalar(0),
                              'sigma_f':0.+np_uniform_scalar(0),
                              'l_k':0.+np.uniform_scalar(0)}
        self.sigma_n = shared_scalar(initial_params['sigma_n'])
        self.sigma_f = shared_scalar(initial_params['sigma_f'])
        self.l_k = shared_scalar(initial_params['l_k'])
        
        # Variables
        X,Y,x_test = T.dmatrices('X','Y','x_test')

        
        print 'Setting up model ...'
        K, Ks, Kss, y_test_mu, y_test_var, log_likelihood,L,alpha,V,fs2,sW = self.get_model(X, Y, x_test)

        print 'Compiling model ...'
        inputs = {'X': X, 'Y': Y, 'x_test': x_test}
        # solve a bug with derivative wrt inputs not in the graph
        z = 0.0*sum([T.sum(v) for v in inputs.values()])
        f = zip(['K', 'Ks', 'Kss', 'y_test_mu', 'y_test_var', 'log_likelihood',
                 'L','alpha','V','fs2','sW'],
                [K, Ks, Kss, y_test_mu, y_test_var, log_likelihood,
                 L, alpha,V,fs2,sW])
        self.f = {n: theano.function(inputs.values(), f+z, name=n, on_unused_input='ignore')
                     for n, f in f}

        wrt = {'sigma_n':self.sigma_n, 'sigma_f':self.sigma_f, 'l_k':self.l_k}
        self.g = {vn: theano.function(inputs.values(), T.grad(log_likelihood+z,vv),
                                      name=vn,on_unused_input='ignore')
                                      for vn, vv in wrt.iteritems()}

    def get_model(self,X, Y, x_test):
        '''
        return posterior, prediction mean, variance, and log marginal likelihood
        '''
        # compute covariance matrices
        K = self.covFunc(X,X,'K')
        Ks = self.covFunc(X,x_test,'Ks')
        # Pay attention, here is the self test cov matrix.
        Kss = T.ones_like(x_test)
        #Kss = self.covFunc(x_test, x_test,'Kss')
        
        # noise variance of likGauss
        sn2 = T.exp(2*self.sigma_n)
        
        # mean func value
        m = T.mean(Y)*T.ones_like(Y)

        # compute prediction mean, variance
        # C.E. Rasmussen, "Gaussian Process for Machine Learning", MIT Press 2006, p19
        # The cov matrix inverse is computed through Cholesky factorization
        # A = LL^T, A^(-1) = (L^-1)^T(L^(-1))
        # https://makarandtapaswi.wordpress.com/2011/07/08/cholesky-decomposition-for-matrix-inversion/
        L = sT.cholesky(K/sn2 + T.identity_like(K))
        sl = sn2
        alpha = T.dot(sT.matrix_inverse(L.T), 
                      T.dot(sT.matrix_inverse(L), (Y-m)) ) / sl
        sW = T.ones_like(T.sum(K,axis=1)).reshape((K.shape[0],1)) / T.sqrt(sl)
        fmu = m + T.dot(Ks.T, alpha) # Prediction Mu fs|f, eq 2.25 book
        tmp = T.extra_ops.repeat(sW,x_test.shape[0],axis=1)

        V = T.dot(sT.matrix_inverse(L),T.extra_ops.repeat(sW,x_test.shape[0],axis=1)*Ks)
        #v = T.dot(sT.matrix_inverse(L),Ks)
        fs2 = Kss - (T.sum(V*V,axis=0)).reshape((1,V.shape[1])).T # Predication Sigma, eq 2.26 book
        fs2 = T.maximum(fs2,0) # remove negative variance noise

        y_test_mu = fmu
        y_test_var = fs2 + sn2

        log_likelihood = -0.5 * (T.dot((Y-m).T, alpha)) - T.sum(T.log(T.diag(L))) - X.shape[0] / 2 * T.log(2.*np.pi*sl)
        
        return K, Ks, Kss, y_test_mu, y_test_var, T.sum(log_likelihood), L, alpha,V, fs2,sW

    
    def covFunc(self,x1,x2,name,method='SE'):
        '''
        Factorization Implementation of distance function.
        https://chrisjmccormick.wordpress.com/2014/08/22/fast-euclidean-distance-calculation-with-matlab-code/
        '''
        if method == 'SE':
            ell = T.exp(self.l_k)
            sf2 = T.exp(2.*self.sigma_f)
            xx = T.sum(x1**2,axis=1).reshape((x1.shape[0],1))
            xc = T.dot(x1, x2.T)
            cc = T.sum(x2**2,axis=1).reshape((1,x2.shape[0]))
            dist = xx - 2*xc + cc
            k = sf2 * T.exp(-dist/2/ell)
            #f_cov = theano.function((x1,x2),k,name=name,on_unused_input='ignore')
        else:
            raise NotImplementedError
        return k

    
    def get_prediction(self, x_val, y_val, x_test_val):
        '''
        Input numpy array, output posterior distributions.
        Note: This function is independent of Theano
        '''
        inputs = {'X':x_val, 'Y':y_val, 'x_test':x_test_val}
        outputs = {n: self.f[n](*inputs.values()) for n in self.f.keys()}
        return outputs


    def get_cost_grads(self, x_val, y_val):
        '''
        get the likelihood and gradients 
        '''
        inputs = {'X':x_val, 'Y':y_val, 'x_test':x_val}
        outputs = {n: self.f[n](*inputs.values()) for n in self.f.keys()}
        grads = {n: self.g[n](*inputs.values()) for n in self.g.keys()}

        return grads, outputs
    

    def train(self, x_val, y_val,
              lr = 0.001, momentum = 0,decay = None,
              nesterov = None,batch_size=None,
              num_epoch = 10):
        '''
        Optimize model's hyper-parameters using SGD
        '''
        params  = {'sigma_n':self.sigma_n, 'sigma_f':self.sigma_f, 'l_k':self.l_k}
        N = x_val.shape[0]
        if batch_size is None:
            batch_size = N
        
        num_batches = N / batch_size
        if N%batch_size!=0:
            num_batches += 1
        train_index = np.arange(0,N)
        
        for epoch in range(num_epoch):
            if decay is not None:
                lr = lr * (1./(1. + decay * epoch))
            for i in range(num_batches):
                np.random.shuffle(train_index)
                batch_x_val = x_val[train_index,:]
                batch_y_val = y_val[train_index,:]

                grads,outputs = self.get_cost_grads(batch_x_val, batch_y_val)
                #print 'Log Likelihood = ', outputs['log_likelihood']
                # update parameters
                for n in params.keys():
                    g,p = grads[n], params[n]
                    p.set_value(p.get_value() + lr * g)
                #print '\n\n'
                #pdb.set_trace()
                #tmp = 'update parameters'
            if epoch % 100 == 0:    
                print 'On Epoch %d, Log Likelihood = '%epoch, outputs['log_likelihood']
        print 'END Training, Log Likelihood = ', outputs['log_likelihood']





