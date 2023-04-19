import numpy

from scipy.integrate import solve_ivp
from scipy.optimize import minimize


class BatchReactor():

    def __init__( self, nmax=5, mesh=0, grid='discrete',
            rho=None, alpha1m=None, rho_melt=None, H0=None, H1=None,
            concs=[0.0, 0.0, 0.0, 0.0, 1.0],
            temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0, rand=0.0 ):

        if grid == 'discrete':
            self.n = numpy.arange(1, nmax+1, 1)
            self.get_melt = self.get_discrete_melt
            self.get_rate = self.get_discrete_rate
        elif grid == 'continuum':
            self.n = numpy.linspace(1.0, nmax, mesh)
            self.get_melt = self.get_continuum_melt
            self.get_rate = self.get_continuum_rate
        elif grid == 'log_n':
            self.n = numpy.logspace(0.0, numpy.log10(nmax), mesh)
            self.get_melt = self.get_log_n_melt
            self.get_rate = self.get_log_n_rate

        if rho is None:
            n = self.n
            if grid == 'discrete':
                rho = numpy.array(concs) / numpy.inner(n, concs)
            elif grid == 'continuum':
                dn = n[1] - n[0]
                g = n * concs
                rho = numpy.array(concs) / ( 0.5 * numpy.einsum('i->', g[1:] + g[:-1]) * dn )
            elif grid == 'log_n':
                r = numpy.log(n)
                dr = r[1] - r[0]
                g = n**2 * concs
                rho = numpy.array(concs) / ( 0.5 * numpy.einsum('i->', g[1:] + g[:-1]) * dr )
        self.rho = rho

        if rho_melt is None:
            rho_melt = ( dens * volume ) / ( mass )
        self.rho_melt = rho_melt

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

        self.rand = rand

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

        W = self.get_melt(n, rho, rho_melt, H0, H1, gtol)

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

        def fun(x):
            W = numpy.exp(x)
            g = ( n * rho ) / ( 1.0 + n * ( 1.0/W - 1.0/rho_melt ) * H0 * H1**n )
            dW = 0.5 * numpy.einsum('i->', g[1:] + g[:-1]) * dn - W
            g = ( n * rho ) * ( n * (1.0/W) * H0 * H1**n ) / ( 1.0 + n * ( 1.0/W - 1.0/rho_melt ) * H0 * H1**n )**2
            dWdx = 0.5 * numpy.einsum('i->', g[1:] + g[:-1]) * dn - W
            f = dW**2
            dfdx = 2.0 * dW * dWdx
            return f, dfdx

        g = ( n * rho ) / ( 1.0 + n * H0 * H1**n )
        x = numpy.log( 0.5 * numpy.einsum('i->', g[1:] + g[:-1]) * dn )
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

        def fun(x):
            W = numpy.exp(x)
            g = ( n**2 * rho ) / ( 1.0 + n * ( 1.0/W - 1.0/rho_melt ) * H0 * H1**n )
            dW = 0.5 * numpy.einsum('i->', g[1:] + g[:-1]) * dr - W
            g = ( n**2 * rho ) * ( n * (1.0/W) * H0 * H1**n ) / ( 1.0 + n * ( 1.0/W - 1.0/rho_melt ) * H0 * H1**n )**2
            dWdx = 0.5 * numpy.einsum('i->', g[1:] + g[:-1]) * dr - W
            f = dW**2
            dfdx = 2.0 * dW * dWdx
            return f, dfdx

        g = ( n**2 * rho ) / ( 1.0 + n * H0 * H1**n )
        x = numpy.log( 0.5 * numpy.einsum('i->', g[1:] + g[:-1]) * dr )
        solver = minimize(fun, x, method='BFGS', jac=True, options={'gtol': gtol})
        W = numpy.exp(solver.x)

        return W

    def get_func( self, n=None, rho=None, alpha1m=None, rho_melt=None, H0=None, H1=None, rand=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_melt is None:
            rho_melt = self.rho_melt
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if rand is None:
            rand = self.rand

        if alpha1m is None:
            alpha1m = self.get_part(n, rho, rho_melt, H0, H1, gtol)

        rate = self.get_rate(n, rho, alpha1m, rand)

        func = rate

        return func

    def get_discrete_rate( self, n=None, rho=None, alpha1m=None, rand=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rand is None:
            rand = self.rand

        y = alpha1m * rho

        dy = numpy.zeros_like(y)
        if rand != 1.0:
            dy[1:-1] = y[2:] - y[1:-1]
            dy[0   ] = y[1 ] - y[0   ]
            dy[  -1] =       - y[  -1]

        sy = numpy.zeros_like(y)
        if rand != 0.0:
            sy[-2::-1] = numpy.cumsum(y[:0:-1])
            sy[-1    ] = 0.0

        rate = ( 1.0 - rand ) * ( dy ) + ( rand ) * ( 2.0 * sy - ( n - 1.0 ) * y )

        return rate

    def get_continuum_rate( self, n=None, rho=None, alpha1m=None, rand=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rand is None:
            rand = self.rand

        y = alpha1m * rho
        dn = n[1] - n[0]

        dydn = numpy.zeros_like(y)
        if rand != 1.0:
            dydn[1:-1] = ( y[2:  ] - y[ :-2] ) / ( 2.0 * dn )
            #dydn[0   ] = ( y[1   ] - y[0   ] ) / ( dn )
            #dydn[  -1] = ( y[  -1] - y[  -2] ) / ( dn )
            dydn[0   ] = dydn[1   ]
            dydn[  -1] = dydn[  -2]
            #dydn[ 0] = 2.0 * dydn[ 1] - dydn[ 2]
            #dydn[-1] = 2.0 * dydn[-2] - dydn[-3]

        d2ydn2 = numpy.zeros_like(y)
        if rand != 1.0:
            d2ydn2[1:-1] = ( y[2:  ] - 2.0 * y[1:-1] + y[ :-2] ) / ( dn**2.0 )
            #d2ydn2[0   ] = 0.0
            #d2ydn2[  -1] = 0.0
            d2ydn2[0   ] = d2ydn2[1   ]
            d2ydn2[  -1] = d2ydn2[  -2]
            #d2ydn2[ 0] = 2.0 * d2ydn2[ 1] - d2ydn2[ 2]
            #d2ydn2[-1] = 2.0 * d2ydn2[-2] - d2ydn2[-3]

        iy = numpy.zeros_like(y)
        if rand != 0.0:
            g = y
            iy[-2::-1] = 0.5 * numpy.cumsum(g[:0:-1] + g[-2::-1]) * dn
            iy[-1    ] = 0.0
            #iy[-1    ] = iy[-2]
            #iy[-1    ] = 2.0 * iy[-2] - iy[-3]

        rate = ( 1.0 - rand ) * ( dydn + 0.5 * d2ydn2 ) + ( rand ) * ( 2.0 * iy - n * y )

        return rate

    def get_log_n_rate( self, n=None, rho=None, alpha1m=None, rand=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rand is None:
            rand = self.rand

        y = alpha1m * rho

        r = numpy.log(n)
        dr = r[1] - r[0]

        dydr = numpy.zeros_like(y)
        if rand != 1.0:
            dydr[1:-1] = ( y[2:  ] - y[ :-2] ) / ( 2.0 * dr )
            dydr[0   ] = ( y[1   ] - y[0   ] ) / ( dr )
            dydr[  -1] = ( y[  -1] - y[  -2] ) / ( dr )
            #dydr[0   ] = dydr[1   ]
            #dydr[  -1] = dydr[  -2]
            #dydr[ 0] = 2.0 * dydr[ 1] - dydr[ 2]
            #dydr[-1] = 2.0 * dydr[-2] - dydr[-3]

        d2ydr2 = numpy.zeros_like(y)
        if rand != 1.0:
            d2ydr2[1:-1] = ( y[2:  ] - 2.0 * y[1:-1] + y[ :-2] ) / ( dr**2.0 )
            d2ydr2[0   ] = 0.0
            d2ydr2[  -1] = 0.0
            #d2ydr2[0   ] = d2ydr2[1   ]
            #d2ydr2[  -1] = d2ydr2[  -2]
            #d2ydr2[ 0] = 2.0 * d2ydr2[ 1] - d2ydr2[ 2]
            #d2ydr2[-1] = 2.0 * d2ydr2[-2] - d2ydr2[-3]

        iy = numpy.zeros_like(y)
        if rand != 0.0:
            g = n * y
            iy[-2::-1] = 0.5 * numpy.cumsum(g[:0:-1] + g[-2::-1]) * dr
            iy[-1    ] = 0.0
            #iy[-1    ] = iy[-2]
            #iy[-1    ] = 2.0 * iy[-2] - iy[-3]

        rate = ( 1.0 - rand ) * ( ( 1.0/n - 0.5/n**2 ) * dydr + (0.5/n**2) * d2ydr2 ) + ( rand ) * ( 2.0 * iy - n * y )

        return rate

    def solve( self, t, n=None, rho=None, alpha1m=None, rho_melt=None, H0=None, H1=None, rand=None, gtol=1e-6, rtol=1e-6, atol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_melt is None:
            rho_melt = self.rho_melt
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if rand is None:
            rand = self.rand

        def fun( t, y ):
            return self.get_func(n, y, alpha1m, rho_melt, H0, H1, rand, gtol)

        solver = solve_ivp(fun, [0.0, t], rho, method='BDF', rtol=rtol, atol=atol)

        self.solver = solver

        return solver.t, solver.y


class CSTReactor(BatchReactor):

    def __init__( self, nmax=5, mesh=0, grid='discrete',
            rho=None, alpha=None, alpha1m=None, rho_melt=None, H0=None, H1=None, fin=None, fout=None,
            concs=[0.0, 0.0, 0.0, 0.0, 1.0], influx=[0.0, 0.0, 0.0, 0.0, 0.0], outflux=[0.0, 0.0],
            temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0, rand=0.0 ):

        super().__init__(nmax, mesh, grid, rho, alpha1m, rho_melt, H0, H1, concs, temp, volume, mass, monomer, dens, rand)

        self.alpha = alpha

        if alpha is None:
            n = self.n
            H0 = self.H0
            H1 = self.H1
            alpha = 1.0 / ( 1.0 + 1.0 / ( n * H0 * H1**n ) )
        self.alpha0 = alpha

        if fin is None:
            n = self.n
            if grid == 'discrete':
                fin = numpy.array(influx) / numpy.inner(n, concs)
            elif grid == 'continuum':
                dn = n[1] - n[0]
                g = n * concs
                fin = numpy.array(influx) / ( 0.5 * numpy.einsum('i->', g[1:] + g[:-1]) * dn )
            elif grid == 'log_n':
                r = numpy.log(n)
                dr = r[1] - r[0]
                g = n**2 * concs
                fin = numpy.array(influx) / ( 0.5 * numpy.einsum('i->', g[1:] + g[:-1]) * dr )
        self.fin = fin

        self.fout = numpy.array(outflux)

        return

    def get_part( self, n=None, rho=None, rho_melt=None, H0=None, H1=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho_melt is None:
            rho_melt = self.rho_melt
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        W = self.get_melt(n, rho, rho_melt, H0, H1, gtol)

        A = n * ( 1.0/W - 1.0/rho_melt ) * H0 * H1**n
        alpha = 1.0 / ( 1.0 + 1.0 / A )
        alpha1m = 1.0 / ( 1.0 + A )

        return alpha, alpha1m

    def get_func( self, n=None, rho=None, alpha=None, alpha1m=None, rho_melt=None, H0=None, H1=None, fin=None, fout=None, rand=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha is None:
            alpha = self.alpha
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_melt is None:
            rho_melt = self.rho_melt
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if fin is None:
            fin = self.fin
        if fout is None:
            fout = self.fout
        if rand is None:
            rand = self.rand

        if alpha is None:
            if alpha1m is None:
                alpha, alpha1m = self.get_part(n, rho, rho_melt, H0, H1, gtol)
            else:
                alpha, _ = self.get_part(n, rho, rho_melt, H0, H1, gtol)
        elif alpha1m is None:
            _, alpha1m = self.get_part(n, rho, rho_melt, H0, H1, gtol)

        rate = self.get_rate(n, rho, alpha1m, rand)

        func = rate + fin - fout[0] * alpha * rho - fout[1] * alpha1m * rho

        return func

    def solve( self, t, n=None, rho=None, alpha=None, alpha1m=None, rho_melt=None, H0=None, H1=None, fin=None, fout=None, rand=None, gtol=1e-6, rtol=1e-6, atol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha is None:
            alpha = self.alpha
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_melt is None:
            rho_melt = self.rho_melt
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if fin is None:
            fin = self.fin
        if fout is None:
            fout = self.fout
        if rand is None:
            rand = self.rand

        def fun( t, y ):
            return self.get_func(n, y, alpha, alpha1m, rho_melt, H0, H1, fin, fout, rand, gtol)

        solver = solve_ivp(fun, [0.0, t], rho, method='BDF', rtol=rtol, atol=atol)

        self.solver = solver

        return solver.t, solver.y

    def cointegrate( self, t=None, y=None, n=None, rho=None, alpha=None, alpha1m=None, rho_melt=None, H0=None, H1=None, fin=None, fout=None, rand=None, gtol=1e-6 ):

        if t is None:
            t = self.solver.t
        if y is None:
            y = self.solver.y

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha is None:
            alpha = self.alpha
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_melt is None:
            rho_melt = self.rho_melt
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if fin is None:
            fin = self.fin
        if fout is None:
            fout = self.fout
        if rand is None:
            rand = self.rand

        dt = t[1:] - t[:-1]

        g = numpy.zeros_like(y)
        gin = numpy.zeros_like(y)
        gout = numpy.zeros_like(y)

        for i, _ in enumerate(t):

            alp = alpha
            a1m = alpha1m

            if alp is None:
                if a1m is None:
                    alp, a1m = self.get_part(n, y[:, i], rho_melt, H0, H1, gtol)
                else:
                    alp, _ = self.get_part(n, y[:, i], rho_melt, H0, H1, gtol)
            elif a1m is None:
                _, a1m = self.get_part(n, y[:, i], rho_melt, H0, H1, gtol)

            rate = self.get_rate(n, y[:, i], a1m)

            g[:, i] = rate
            gin[:, i] = fin
            gout[:, i] = fout[0] * alp * y[:, i] + fout[1] * a1m * y[:, i]

        G = numpy.zeros_like(y)
        Gin = numpy.zeros_like(y)
        Gout = numpy.zeros_like(y)

        G[:, 1:] = 0.5 * numpy.cumsum(g[:, 1:] + g[:, :-1], axis=1) * dt
        Gin[:, 1:] = 0.5 * numpy.cumsum(gin[:, 1:] + gin[:, :-1], axis=1) * dt
        Gout[:, 1:] = 0.5 * numpy.cumsum(gout[:, 1:] + gout[:, :-1], axis=1) * dt

        G[:, 0] = 0.0
        Gin[:, 0] = 0.0
        Gout[:, 0] = 0.0

        return G, Gin, Gout

