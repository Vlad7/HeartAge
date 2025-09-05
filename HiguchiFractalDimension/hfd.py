#!/usr/bin/python3

"""
Higuchi Fractal Dimension according to:
T. Higuchi, Approach to an Irregular Time Series on the
Basis of the Fractal Theory, Physica D, 1988; 31: 277-283.
"""
import decimal
import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from decimal import *
import math


def curve_length(X,opt=True,num_k=50,k_max=None):
    """
    Calculate curve length <Lk> for Higuchi Fractal Dimension (HFD)
    
    Input:
    
    X - input (time) series (must be 1D, to be converted into a NumPy array)
    opt (=True) - optimized? (if libhfd.so was compiled uses the faster code).
    num_k - number of k values to generate.
    k_max - the maximum k (the k array is generated uniformly in log space 
            from 2 to k_max)
    Output:

    k - interval "times", window sizes
    Lk - curve length
    """
    ### Make sure X is a NumPy array with the correct dimension
    X = np.array(X)
    if X.ndim != 1:
        raise ValueError("Input array must be 1D (time series).")
    N = X.size

    ### Get interval "time"
    k_arr = interval_t(N,num_val=num_k,kmax=k_max)

    ### The average length
    Lk = np.empty(k_arr.size,dtype=float)

    ### C library
    if opt:
        X = np.require(X, float, ('C', 'A'))                                                # READ MORE ABOUT
        k_arr = np.require(k_arr, ctypes.c_size_t, ('C', 'A'))                              # READ MORE ABOUT
        Lk = np.require(Lk, float, ('C', 'A'))                                              # READ MORE ABOUT
        ## Load library here
        libhfd = init_lib()
        ## Run the C code here
        libhfd.curve_length(k_arr,k_arr.size,X,N,Lk)
    
    else:
        ### Native Python run
        for i in range(k_arr.size):# over array of k's
            Lmk = 0.0
            for j in range(k_arr[i]):# over m's
                ## Construct X_k^m, i.e. X_(k_arr[i])^j, as X[j::k_arr[i]]
                ## Calculate L_m(k)

                Lmk += (
                    np.sum(
                        np.abs(
                            np.diff( X[j::k_arr[i]] )
                        )
                    )

                #    np.sum(
                #    np.sqrt(np.power(np.abs(
                #    np.diff( X[j::k_arr[i]])
                #), 2) + np.power(k_arr[i],2))
                #)
                     #np.sum(
                        
                    #np.sqrt(np.power(np.abs(
                    #np.diff( X[j::k_arr[i]])
                #), 2) + np.power(k_arr[i], 2))
                #)
                    * (N - 1) /
                    (
                        ( (N-j-1)//k_arr[i] )
                        *
                        k_arr[i]
                    )
                ) / k_arr[i]

            ### Calculate the average Lmk
            Lk[i] = Lmk / k_arr[i]

    return (k_arr, Lk);

def lin_fit_hfd(k,L,log=True):
    """
    Calculate Higuchi Fractal Dimension (HFD) by fitting a line to already computed
    interval times k and curve lengths L

    Input:

    k - interval "times", window sizes
    L - curve length
    log (=True) - k and L values will be transformed to np.log2(k)    and np.log2(L),
                  respectively

    Output:

    HFD
    """

    if log:
        return (-np.polyfit(np.log2(k),np.log2(L),deg=1)[0]);
    else:
        return (-np.polyfit(k,L,deg=1)[0]);

def hfd(X,**kwargs):
    """
    Calculate Higuchi Fractal Dimension (HFD) for 1D data/series

    Input:

    X - input (time) series (must be 1D, to be converted into a NumPy array)

    Output:
    
    HFD
    """
    k, L = curve_length(X,**kwargs)

    return lin_fit_hfd(k, L);

def interval_t(size,num_val=50,kmax=None):
    ### Generate sequence of interval times, k
    #print("Size: "+str(size))

    if kmax is None:
        k_stop = size//2
    else:
        k_stop = kmax
    if k_stop > size//2:## prohibit going larger than N/2
        k_stop = size//2
        print("Warning: k cannot be longer than N/2")
    #print(size)
    #print(size // 2)
    #print(np.log2(k_stop))
    np.set_printoptions(precision=30)

    #print(np.longdouble(math.log2(np.longdouble(k_stop))))
    k = np.logspace(start=np.log2(2),stop=np.log2(k_stop), endpoint=True, base=2,num=num_val,dtype=int)


    ####################################################################
    # WARNING !!! Внести изменения в исходный код.
    ####################################################################
    k[len(k) - 1] = k_stop


    #k1 = np.around(k, decimals=14)
    #print(k1)
    #k = k.astype(dtype=np.int, copy=False)
    #y = np.linspace(np.log2(2), np.log2(k_stop), num=num_val, endpoint=True)
    #y = np.power(2,y)
    #print(y)
    #y[1] = 100.
    #getcontext().prec = 28
    #print(Decimal(math.log2(Decimal(k_stop))))

    #y = np.linspace(Decimal(np.log2(2)), Decimal(math.log2(Decimal(k_stop))), endpoint=True, num=num_val, axis=0)




    #print(y)
    #if dtype is None:
    #    return _nx.power(base, y)
    #return _nx.power(base, y).astype(dtype, copy=False)






    #print(k)





    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OWN CODE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Because bug in np.logspace with dtype = np.float?
    # 218 - 108 integer; 220 - 110 integer;
    #print(y)
    #y = y.astype(dtype=np.int, copy=False)
    #print(y)


    return np.unique(k);

def init_lib():
    libdir = os.path.dirname(__file__)
    libfile = os.path.join(libdir, "libhfd.so")
    lib = ctypes.CDLL(libfile)

    rwptr = ndpointer(float, flags=('C','A','W'))
    rwptr_sizet = ndpointer(ctypes.c_size_t, flags=('C','A','W'))

    lib.curve_length.restype = ctypes.c_int
    lib.curve_length.argtypes = [rwptr_sizet, ctypes.c_size_t, rwptr, ctypes.c_size_t, rwptr]

    return lib;

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    N = 10
    x1 = np.logspace(0.1, 1, N, endpoint=True)
    x2 = np.logspace(0.1, 1, N, endpoint=False)
    y = np.zeros(N)
    plt.plot(x1, y, 'o')
    print(interval_t(100))

    plt.plot(x2, y + 0.5, 'o')

    plt.ylim([-0.5, 1])

    plt.show()

