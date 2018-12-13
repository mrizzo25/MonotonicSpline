#! /usr/bin/python

import numpy as np

def monotonic_cubic_spline(x_values, y_values):
    """
    Algorithm from https://uglyduckling.nl/library_files/PRE-PRINT-UD-2014-03.pdf
    """
    n=len(x_values)
    h=np.zeros(n)
    m=np.zeros(n)
    beta=np.zeros(n)
    consts=np.zeros((n,3))

    for i in np.arange(0,n-1,1):
        h[i]=x_values[i+1]-x_values[i]
        m[i]=(y_values[i+1]-y_values[i])/h[i]

    consts[0,0]=m[0]
    consts[n-1,0]=m[-1]

    for i in np.arange(1,n-1,1):
        if (m[i-1]*m[i]<=0.):
            consts[i,0]=0.
        elif (m[i-1]*m[i]>0.):
            beta[i]=(3.*m[i-1]*m[i])/(max(m[i-1],m[i])+2.*min(m[i-1],m[i]))
            if np.sign(m[i-1])==1 and np.sign(m[i])==1:
                consts[i,0]= min(max(0.,beta[i]),3.*min(m[i-1],m[i]))
            elif np.sign(m[i-1])==-1 and np.sign(m[i])==-1:
                consts[i,0]=max(min(0.,beta[i]),3.*max(m[i-1],m[i]))
            else:
                print "We have a problem"

    for i in np.arange(0,n-1,1):
        consts[i,1]=(3*m[i]-consts[i+1,0]-2.*consts[i,0])/h[i]
        consts[i,2]=(consts[i+1,0]+consts[i,0]-2.*m[i])/h[i]**2

    return consts
    

def lin_extrapolate(x_values, y_values):
    """
    Extrapolate data linearly at endpoints
    """
    n=len(x_values)

    m1=(y_values[1]-y_values[0])/(x_values[1]-x_values[0])
    b1=y_values[0]-m1*x_values[0]

    m2=(y_values[n-1]-y_values[n-2])/(x_values[n-1]-x_values[n-2])
    b2=y_values[n-1]-m2*x_values[n-1]

    line_consts=np.array([[m1,b1],[m2,b2]])

    return line_consts


def interp_func(x, x_table, y_table, consts, extrapolate=False, extrap_consts=None, verbose=False):
    """
    Predict y value using interpolating functions
    """

    for i in np.arange(0, len(x_table)-1, 1):
      
        if x_table[i] < x and x_table[i+1] > x:
            ans = y_table[i] + consts[i,0] * (x - x_table[i]) + consts[i,1] * (x - x_table[i])**2 + consts[i,2] * (x - x_table[i])**3

            if verbose == True:
                print "Input", x
                print "Output", ans
            return ans

        elif x == x_table[i]:
            ans = y_table[i]
            if verbose == True:
                print "Input", x
                print "Output", ans
            return ans

        elif x == x_table[i+1]:
            ans = y_table[i+1]
            if verbose == True:
                print "Input", x
                print "Output", ans
            return ans

    if extrapolate == False:

        print "X value outside of range"
        return None
    
    else:

        if x < x_table[0]:
            return extrap_consts[0,0] * x + extrap_consts[0,1]
        elif x > x_table[-1]:
            return extrap_consts[1,0] * x + extrap_consts[1,1]
    

class Cubic:
    """
    Construct (monotonicity enforcing) cubic spline of data
    """

    def __init__(self, x, y, extrapolate=False):
        """
        Fit x and y data using monotonic cubic spline   
        """

        #make sure the x and y arrays are the same size
        if len(x) != len(y):
            raise ValueError("X and Y arrays must be the same size")

		#sort x values (important for randomly sampled data)
		sort = np.argsort(x)
	
		x = x[sort]
        y = y[sort]

        self.coefficients = monotonic_cubic_spline(x, y)

        if extrapolate == True:
            self.ext_coefficients = lin_extrapolate(x, y)
        else:
            self.ext_coefficients = None

    	self.extrapolate = extrapolate
        self.x_table = x
        self.y_table = y
   
    def predict(self, x, verbose=False):
        """
        Use interpolating splines to predict y value(s) given x value(s)
        """
        if hasattr(x, '__iter__'):
            y_predict = np.zeros(len(x))
            	
            for i in range(len(x)):
            	y_predict[i] = interp_func(x[i], self.x_table, self.y_table, self.coefficients, extrapolate = self.extrapolate, extrap_consts=self.ext_coefficients, verbose=verbose)

        else:
            y_predict = interp_func(x, self.x_table, self.y_table, self.coefficients, extrapolate = self.extrapolate, extrap_consts=self.ext_coefficients, verbose=verbose)

        return y_predict

