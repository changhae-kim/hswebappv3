import numpy

from scipy.integrate import solve_ivp


class BatchReactor():

    def __init__( self, nmax=5, grid=0, rho=None, alpha1m=None, H0=None, H1=None, concs=[0, 0, 0, 0, 1], temp=573.15, volume=1.0, mass=10.0, monomer=14.027 ):

        if grid == 0:
            self.n = numpy.arange(1, nmax+1, 1)
            self.get_rate = self.get_exact_rate
        else:
            self.n = numpy.linspace(1.0, nmax, grid)
            self.get_rate = self.get_continuum_rate

        n = self.n
        dn = n[1] - n[0]

        if rho is None:
            if concs is None:
                concs = numpy.ones_like(n)
            if grid == 0:
                rho = numpy.array(concs) / numpy.inner(n, concs)
            else:
                w = numpy.ones_like(n)
                w[0] = w[-1] = 0.5
                rho = numpy.array(concs) / numpy.einsum('i,i,i->', w, n, concs) * dn
        self.rho = rho

        if alpha1m is None:
            if H0 is None:
                H0 = ( monomer * volume ) / ( mass * 0.082057366080960 * temp ) * numpy.exp( 8.09559982139232E+00 + 4.59679345240217E+02 / temp )
            if H1 is None:
                H1 = numpy.exp( 3.31727830327662E-01 - 5.34127400185307E+02 / temp )
            alpha1m = 1.0 / ( 1.0 + n * H0 * H1**n )
        self.alpha1m = alpha1m

        return

    def get_exact_rate( self, n=None, rho=None, alpha1m=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m

        y = alpha1m * rho

        dy = numpy.empty_like(y)
        dy[1:-1] = y[2:] - y[1:-1]
        dy[0   ] = y[1 ]
        dy[  -1] =       - y[  -1]

        rate = dy

        return rate

    def get_continuum_rate( self, n=None, rho=None, alpha1m=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m

        y = alpha1m * rho
        dn = n[1] - n[0]

        dydn = numpy.empty_like(y)
        dydn[1:-1] = ( y[2:] - y[:-2] ) / ( 2.0 * dn )
        dydn[0   ] = ( y[1 ]          ) / dn # ( 2.0 * dn )
        dydn[  -1] = (       - y[ -2] ) / dn # ( 2.0 * dn )

        d2ydn2 = numpy.empty_like(y)
        d2ydn2[1:-1] = ( y[2:] - 2.0 * y[1:-1] + y[:-2] ) / ( dn**2.0 )
        d2ydn2[0   ] = 0.0 # ( y[1 ] - 2.0 * y[0   ]          ) / ( dn**2.0 )
        d2ydn2[  -1] = 0.0 # (       - 2.0 * y[  -1] + y[ -2] ) / ( dn**2.0 )

        rate = dydn + 0.5 * d2ydn2

        return rate

    def solve( self, t, n=None, rho=None, alpha1m=None, rtol=1e-3, atol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m

        def fun( t, y ):
            return self.get_rate(n, y, alpha1m)

        solver = solve_ivp(fun, [0.0, t], rho, rtol=rtol, atol=atol)

        return solver.t, solver.y
