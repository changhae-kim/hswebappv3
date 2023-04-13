import numpy

from scipy.integrate import solve_ivp
from scipy.optimize import minimize


class BatchReactor():

    def __init__( self, nmax=5, mesh=0, grid='discrete',
            rho=None, alpha1m=None, H0=None, H1=None,
            concs=[0.0, 0.0, 0.0, 0.0, 1.0],
            temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0 ):

        if grid == 'discrete':
            self.n = numpy.arange(1, nmax+1, 1)
            self.get_melt = self.get_discrete_melt
            self.get_rhs = self.get_discrete_rhs
        elif grid == 'continuum':
            self.n = numpy.linspace(1.0, nmax, mesh)
            self.get_melt = self.get_continuum_melt
            self.get_rhs = self.get_continuum_rhs
        elif grid == 'log_n':
            self.n = numpy.logspace(0.0, numpy.log10(nmax), mesh)
            self.get_melt = self.get_log_n_melt
            self.get_rhs = self.get_log_n_rhs

        if rho is None:
            n = self.n
            if grid == 'discrete':
                rho = numpy.array(concs) / numpy.inner(n, concs)
            elif grid == 'continuum':
                dn = n[1] - n[0]
                w = numpy.ones_like(n)
                w[0] = w[-1] = 0.5
                rho = numpy.array(concs) / ( numpy.einsum('i,i,i->', n, concs, w) * dn )
            elif grid == 'log_n':
                r = numpy.log(n)
                dr = r[1] - r[0]
                w = numpy.ones_like(n)
                w[0] = w[-1] = 0.5
                rho = numpy.array(concs) / ( numpy.einsum('i,i,i,i->', n, concs, n, w) * dr )
        self.rho = rho

        self.rho_melt = ( dens * volume ) / ( mass )

        if alpha1m is None:
            if H0 is None:
                H0 = ( monomer * volume ) / ( mass * 0.082057366080960 * temp ) * numpy.exp( 8.124149532 + 472.8315525 / temp )
            if H1 is None:
                H1 = numpy.exp( 0.327292343 - 536.5152612 / temp )
        self.H0 = H0
        self.H1 = H1
        self.alpha1m = alpha1m

        if alpha1m is None:
            n = self.n
            alpha1m = 1.0 / ( 1.0 + n * H0 * H1**n )
        self.alpha1m0 = alpha1m

        return

    def get_part( self, n=None, rho=None, rho_melt=None, H0=None, H1=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if rho_melt is None:
            rho_melt = self.rho_melt
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        W = self.get_melt( n, rho, rho_melt, H0, H1, gtol=1e-6 )

        alpha1m = 1.0 / ( 1.0 + n * ( 1.0/W - 1.0/rho_melt ) * H0 * H1**n )

        return alpha1m

    def get_discrete_melt( self, n=None, rho=None, rho_melt=None, H0=None, H1=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if rho_melt is None:
            rho_melt = self.rho_melt
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        def fun(x):
            W = numpy.exp(x)
            dW = numpy.sum( ( n * rho ) / ( 1.0 + n * ( 1.0/W - 1.0/rho_melt ) * H0 * H1**n ) ) - W
            dWdx = numpy.sum( ( n * rho * n * (1.0/W) * H0 * H1**n ) / ( 1.0 + n * ( 1.0/W - 1.0/rho_melt ) * H0 * H1**n )**2 ) - W
            f = dW**2
            dfdx = 2.0 * dW * dWdx
            return f, dfdx

        x = numpy.log( numpy.sum( ( n * rho ) / ( 1.0 + n * H0 * H1**n ) ) )
        solver = minimize(fun, x, method='BFGS', jac=True, options={'gtol': gtol})
        W = numpy.exp(solver.x)

        return W

    def get_continuum_melt( self, n=None, rho=None, rho_melt=None, H0=None, H1=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if rho_melt is None:
            rho_melt = self.rho_melt
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        dn = n[1] - n[0]
        w = numpy.ones_like(n)
        w[0] = w[-1] = 0.5

        def fun(x):
            W = numpy.exp(x)
            dW = numpy.sum( ( n * rho ) / ( 1.0 + n * ( 1.0/W - 1.0/rho_melt ) * H0 * H1**n ) * w ) * dn - W
            dWdx = numpy.sum( ( n * rho * n * (1.0/W) * H0 * H1**n ) / ( 1.0 + n * ( 1.0/W - 1.0/rho_melt ) * H0 * H1**n )**2 * w ) * dn - W
            f = dW**2
            dfdx = 2.0 * dW * dWdx
            return f, dfdx

        x = numpy.log( numpy.sum( ( n * rho ) / ( 1.0 + n * H0 * H1**n ) * w ) * dn )
        solver = minimize(fun, x, method='BFGS', jac=True, options={'gtol': gtol})
        W = numpy.exp(solver.x)

        return W

    def get_log_n_melt( self, n=None, rho=None, rho_melt=None, H0=None, H1=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if rho_melt is None:
            rho_melt = self.rho_melt
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        r = numpy.log(n)
        dr = r[1] - r[0]
        w = numpy.ones_like(n)
        w[0] = w[-1] = 0.5

        def fun(x):
            W = numpy.exp(x)
            dW = numpy.sum( ( n * rho ) / ( 1.0 + n * ( 1.0/W - 1.0/rho_melt ) * H0 * H1**n ) * n * w ) * dr - W
            dWdx = numpy.sum( ( n * rho * n * (1.0/W) * H0 * H1**n ) / ( 1.0 + n * ( 1.0/W - 1.0/rho_melt ) * H0 * H1**n )**2 * w * n ) * dr - W
            f = dW**2
            dfdx = 2.0 * dW * dWdx
            return f, dfdx

        x = numpy.log( numpy.sum( ( n * rho ) / ( 1.0 + n * H0 * H1**n ) * n * w ) * dr )
        solver = minimize(fun, x, method='BFGS', jac=True, options={'gtol': gtol})
        W = numpy.exp(solver.x)

        return W

    def get_rate( self, n=None, rho=None, rho_melt=None, alpha1m=None, H0=None, H1=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if rho_melt is None:
            rho_melt = self.rho_melt
        if alpha1m is None:
            alpha1m = self.alpha1m
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        if alpha1m is None:
            alpha1m = self.get_part(n, rho, rho_melt, H0, H1, gtol)

        rate = self.get_rhs(n, rho, alpha1m)

        return rate

    def get_discrete_rhs( self, n=None, rho=None, alpha1m=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m

        y = alpha1m * rho

        dy = numpy.empty_like(y)
        dy[1:-1] = y[2:] - y[1:-1]
        dy[0   ] = y[1 ] - y[0   ]
        dy[  -1] =       - y[  -1]

        rate = dy

        return rate

    def get_continuum_rhs( self, n=None, rho=None, alpha1m=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m

        y = alpha1m * rho
        dn = n[1] - n[0]

        dydn = numpy.empty_like(y)
        dydn[1:-1] = ( y[2:  ] - y[ :-2] ) / ( 2.0 * dn )
        #dydn[0   ] = ( y[1   ] - y[0   ] ) / ( dn )
        #dydn[  -1] = ( y[  -1] - y[  -2] ) / ( dn )
        dydn[0   ] = dydn[1   ]
        dydn[  -1] = dydn[  -2]
        #dydn[ 0] = 2.0 * dydn[ 1] - dydn[ 2]
        #dydn[-1] = 2.0 * dydn[-2] - dydn[-3]

        d2ydn2 = numpy.empty_like(y)
        d2ydn2[1:-1] = ( y[2:  ] - 2.0 * y[1:-1] + y[ :-2] ) / ( dn**2.0 )
        #d2ydn2[0   ] = 0.0
        #d2ydn2[  -1] = 0.0
        d2ydn2[0   ] = d2ydn2[1   ]
        d2ydn2[  -1] = d2ydn2[  -2]
        #d2ydn2[ 0] = 2.0 * d2ydn2[ 1] - d2ydn2[ 2]
        #d2ydn2[-1] = 2.0 * d2ydn2[-2] - d2ydn2[-3]

        rate = dydn + 0.5 * d2ydn2

        return rate

    def get_log_n_rhs( self, n=None, rho=None, alpha1m=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m

        y = alpha1m * rho

        x = numpy.log(n)
        dx = x[1] - x[0]

        dydx = numpy.empty_like(y)
        dydx[1:-1] = ( y[2:  ] - y[ :-2] ) / ( 2.0 * dx )
        dydx[0   ] = ( y[1   ] - y[0   ] ) / ( dx )
        dydx[  -1] = ( y[  -1] - y[  -2] ) / ( dx )
        #dydx[0   ] = dydx[1   ]
        #dydx[  -1] = dydx[  -2]
        #dydx[ 0] = 2.0 * dydx[ 1] - dydx[ 2]
        #dydx[-1] = 2.0 * dydx[-2] - dydx[-3]

        d2ydx2 = numpy.empty_like(y)
        d2ydx2[1:-1] = ( y[2:  ] - 2.0 * y[1:-1] + y[ :-2] ) / ( dx**2.0 )
        d2ydx2[0   ] = 0.0
        d2ydx2[  -1] = 0.0
        #d2ydx2[0   ] = d2ydx2[1   ]
        #d2ydx2[  -1] = d2ydx2[  -2]
        #d2ydx2[ 0] = 2.0 * d2ydx2[ 1] - d2ydx2[ 2]
        #d2ydx2[-1] = 2.0 * d2ydx2[-2] - d2ydx2[-3]

        rate = ( numpy.exp(-x) - 0.5 * numpy.exp(-2.0*x) ) * dydx + 0.5 * numpy.exp(-2.0*x) * d2ydx2

        return rate

    def solve( self, t, n=None, rho=None, rho_melt=None, alpha1m=None, H0=None, H1=None, gtol=1e-6, rtol=1e-6, atol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if rho_melt is None:
            rho_melt = self.rho_melt
        if alpha1m is None:
            alpha1m = self.alpha1m
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        def fun( t, y ):
            return self.get_rate(n, y, rho_melt, alpha1m, H0, H1, gtol)

        solver = solve_ivp(fun, [0.0, t], rho, method='BDF', rtol=rtol, atol=atol)

        return solver.t, solver.y

'''
class CSTReactor(BatchReactor):

    def __init__( self, nmax=5, mesh=0, grid='discrete',
            rho=None, alpha=None, alpha1m=None, H0=None, H1=None, fin=None, fout=None,
            concs=[0.0, 0.0, 0.0, 0.0, 1.0], influx=[0.0, 0.0, 0.0, 0.0, 0.0], outflux=[0.0, 0.0],
            temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0 ):

        super().__init__(nmax, mesh, grid, rho, alpha1m, H0, H1, concs, temp, volume, mass, monomer, dens)

        self.get_batch_rate = self.get_rate
        self.get_rate = self.get_flux_rate

        if alpha is None:
            n = self.n
            H0 = self.H0
            H1 = self.H1
            alpha = 1.0 / ( 1.0 + 1.0 / ( n * H0 * H1**n ) )
        self.alpha = alpha

        if fin is None:
            n = self.n
            if grid == 'discrete':
                fin = numpy.array(influx) / numpy.inner(n, concs)
            elif grid == 'continuum':
                dn = n[1] - n[0]
                w = numpy.ones_like(n)
                w[0] = w[-1] = 0.5
                fin = numpy.array(influx) / ( numpy.einsum('i,i,i->', w, n, concs) * dn )
            elif grid == 'log_n':
                r = numpy.log(n)
                dr = r[1] - r[0]
                w = numpy.ones_like(n)
                w[0] = w[-1] = 0.5
                fin = numpy.array(influx) / ( numpy.einsum('i,i,i->', w, concs, numpy.exp(2.0*r)) * dr )
        self.fin = fin

        self.fout = numpy.array(outflux)

        return

    def get_flux_rate( self, n=None, rho=None, alpha=None, alpha1m=None, fin=None, fout=None ):
        rate = self.get_batch_rate(n, rho, alpha1m) + fin - fout[0] * alpha * rho - fout[1] * alpha1m * rho
        return rate
'''
