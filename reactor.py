import numpy

from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def convert_y_to_rho( mode, x, y, mass=10.0, monomer=14.027 ):

    assert mode in ['dwdn', 'dwdlogn', 'dWdn', 'dWdlogn', 'dWdM', 'dWdlogM', 'dndn', 'dndlogn']

    if mode in ['dwdn', 'dwdlogn', 'dWdn', 'dWdlogn', 'dndn', 'dndlogn']:
        n = x
    elif mode in ['dWdM', 'dWdlogM']:
        n = x / monomer

    if mode == 'dwdn':
        rho = y.T / n
    elif mode == 'dwdlogn':
        rho = y.T / n**2
    elif mode == 'dWdn':
        rho = y.T / n / mass
    elif mode == 'dWdlogn':
        rho = y.T / n**2 / mass
    elif mode == 'dWdM':
        rho = y.T / n * monomer / mass
    elif mode == 'dWdlogM':
        rho = y.T / n**2 / mass
    elif mode == 'dndn':
        rho = y.T
    elif mode == 'dndlogn':
        rho = y.T / n

    return n, rho.T

def convert_rho_to_y( mode, n, rho, mass=10.0, monomer=14.027 ):

    assert mode in ['dwdn', 'dwdlogn', 'dWdn', 'dWdlogn', 'dWdM', 'dWdlogM', 'dndn', 'dndlogn']

    if mode in ['dwdn', 'dwdlogn', 'dWdn', 'dWdlogn', 'dndn', 'dndlogn']:
        x = n
    elif mode in ['dWdM', 'dWdlogM']:
        x = monomer * n

    if mode == 'dwdn':
        y = n * rho.T
    elif mode == 'dwdlogn':
        y = n**2 * rho.T
    elif mode == 'dWdn':
        y = mass * n * rho.T
    elif mode == 'dWdlogn':
        y = mass * n**2 * rho.T
    elif mode == 'dWdM':
        y = mass / monomer * n * rho.T
    elif mode == 'dWdlogM':
        y = mass * n**2 * rho.T
    elif mode == 'dndn':
        y = rho.T
    elif mode == 'dndlogn':
        y = n * rho.T

    return x, y.T

def get_dispersity( mode, t, n, rho, monomer=14.027 ):

    assert mode in ['D_n', 'D_logn', 'D_M', 'D_logM']

    if mode in ['D_n', 'D_M']:
        dx = n[1:] - n[:-1]
        g0 = n**0 * rho.T
        g1 = n**1 * rho.T
        g2 = n**2 * rho.T

    elif mode in ['D_logn', 'D_logM']:
        logn = numpy.log(n)
        dx = logn[1:] - logn[:-1]
        g0 = n**1 * rho.T
        g1 = n**2 * rho.T
        g2 = n**3 * rho.T

    G0 = 0.5 * numpy.sum( ( g0.T[1:] + g0.T[:-1] ).T * dx, axis=1 )
    G1 = 0.5 * numpy.sum( ( g1.T[1:] + g1.T[:-1] ).T * dx, axis=1 )
    G2 = 0.5 * numpy.sum( ( g2.T[1:] + g2.T[:-1] ).T * dx, axis=1 )

    nn = G1 / G0
    nw = G2 / G1
    Dn = nw / nn

    if mode in ['D_n', 'D_logn']:
        return nn, nw, Dn
    elif mode in ['D_M', 'D_logM']:
        Mn = monomer * nn
        Mw = monomer * nw
        return Mn, Mw, Dn

def get_pressure( mode, t, n, rho, V, alpha, temp=573.15, volume=1.0, mass=10.0, monomer=14.027 ):

    assert mode in ['dpdn', 'dpdlogn', 'dPdn', 'dPdlogn', 'dPdM', 'dPdlogM']

    dpdn = alpha.T * ( rho / V ).T

    if mode in ['dpdn', 'dPdn', 'dPdM']:
        dx = n[1:] - n[:-1]
        dpdx = dpdn
    elif mode in ['dpdlogn', 'dPdlogn', 'dPdlogM']:
        logn = numpy.log(n)
        dx = logn[1:] - logn[:-1]
        dpdx = n * dpdn

    p = 0.5 * numpy.sum( ( dpdx.T[1:] + dpdx.T[:-1] ).T * dx, axis=1 )

    if mode in ['dpdn', 'dpdlogn']:
        return p, dpdx.T
    elif mode in ['dPdn', 'dPdlogn', 'dPdM', 'dPdlogM']:
        NRT_V = ( mass / monomer ) * ( 8.31446261815324 * temp / volume )
        P = NRT_V * p
        if mode in ['dPdM']:
            dPdx = NRT_V * dpdx / monomer
        elif mode in ['dPdn', 'dPdlogn', 'dPdlogM']:
            dPdx = NRT_V * dpdx
        return P, dPdx.T

def get_states( mode, t, n, rho, state_cutoffs=[4.5, 16.5], mass=10.0, monomer=14.027, renorm=False ):

    assert mode in ['rho_n', 'rho_logn', 'concs_n', 'concs_logn', 'w_n', 'w_logn', 'W_n', 'W_logn']

    ngl, nls = state_cutoffs
    igl = numpy.count_nonzero(n < ngl)
    ils = numpy.count_nonzero(n < nls)

    if mode in ['rho_n', 'concs_n', 'w_n', 'W_n']:
        dx = n[1:] - n[:-1]
        dxgl = ngl - n[igl]
        dxls = nls - n[ils]
    elif mode in ['rho_logn', 'concs_logn', 'w_logn', 'W_logn']:
        logn = numpy.log(n)
        dx = logn[1:] - logn[:-1]
        dxgl = numpy.log(ngl) - logn[igl]
        dxls = numpy.log(nls) - logn[ils]

    if mode in ['rho_n', 'concs_n']:
        g = rho.T
    elif mode in ['rho_logn', 'concs_logn']:
        g = n * rho.T
    elif mode in ['w_n', 'W_n']:
        g = n * rho.T
    elif mode in ['w_logn', 'W_logn']:
        g = n**2 * rho.T

    Ggg = 0.5 * numpy.sum( ( g.T[    1:igl+1] + g.T[     :igl  ] ).T * dx[     :igl  ], axis=1 )
    Ggl = 0.5 * numpy.sum( ( g.T[igl+1:igl+2] + g.T[igl  :igl+1] ).T * dx[igl  :igl+1], axis=1 )
    Gll = 0.5 * numpy.sum( ( g.T[igl+2:ils+1] + g.T[igl+1:ils  ] ).T * dx[igl+1:ils  ], axis=1 )
    Gls = 0.5 * numpy.sum( ( g.T[ils+1:ils+2] + g.T[ils  :ils+1] ).T * dx[ils  :ils+1], axis=1 )
    Gss = 0.5 * numpy.sum( ( g.T[ils+2:     ] + g.T[ils+1:   -1] ).T * dx[ils+1:     ], axis=1 )

    yg = Ggg                                  + Ggl * ( dxgl / dx[igl] )
    yl = Gll + Ggl * ( 1.0 - dxgl / dx[igl] ) + Gls * ( dxls / dx[ils] )
    ys = Gss + Ggl * ( 1.0 - dxls / dx[ils] )

    if renorm:
        yg, yl, ys = numpy.array([ yg, yl, ys ]) / numpy.sum([ yg, yl, ys ], axis=0)
    else:
        yg, yl, ys = numpy.array([ yg, yl, ys ])

    if mode in ['rho_n', 'rho_logn', 'w_n', 'w_logn']:
        return yg, yl, ys
    elif mode in ['concs_n', 'concs_logn']:
        N = mass / monomer
        return N * yg, N * yl, N * ys
    elif mode in ['W_n', 'W_logn']:
        return mass * yg, mass * yl, mass * ys


class BatchReactor():

    def __init__( self, nmin=1, nmax=5, mesh=0, grid='discrete',
            rho=None, W=None, V=None, alpha=None, alpha1m=None, rho_M=None, H0=None, H1=None,
            concs=[0.0, 0.0, 0.0, 0.0, 1.0],
            temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0, end=1.0, rand=0.0 ):

        if grid == 'discrete':
            self.n = numpy.arange(nmin, nmax+1, 1)
            self.get_melt = self.get_discrete_melt
            self.get_rate = self.get_discrete_rate
            self.get_deriv = self.get_discrete_deriv
        elif grid == 'continuum':
            self.n = numpy.linspace(nmin, nmax, mesh)
            self.get_melt = self.get_continuum_melt
            self.get_rate = self.get_continuum_rate
            self.get_deriv = self.get_continuum_deriv
        elif grid == 'logn':
            self.n = numpy.logspace(numpy.log10(nmin), numpy.log10(nmax), mesh)
            self.get_melt = self.get_logn_melt
            self.get_rate = self.get_logn_rate
            self.get_deriv = self.get_logn_deriv

        if rho is None:
            n = self.n
            if grid == 'discrete':
                rho = numpy.array(concs) / numpy.inner(n, concs)
            elif grid == 'continuum':
                dn = n[1] - n[0]
                g = n * concs
                rho = numpy.array(concs) / ( 0.5 * numpy.sum( g[1:] + g[:-1] ) * dn )
            elif grid == 'logn':
                logn = numpy.log(n)
                dlogn = logn[1] - logn[0]
                g = n**2 * concs
                rho = numpy.array(concs) / ( 0.5 * numpy.sum( g[1:] + g[:-1] ) * dlogn )
        self.rho = rho

        if rho_M is None:
            rho_M = ( dens * volume ) / ( mass )
        self.rho_M = rho_M

        if W is None or V is None or alpha is None or alpha1m is None:
            n = self.n
            if H0 is None:
                H0 = ( monomer * volume ) / ( mass * 0.082057366080960 * temp ) * numpy.exp( 8.124149532 + 472.8315525 / temp )
            if H1 is None:
                H1 = numpy.exp( 0.327292343 - 536.5152612 / temp )
            if W is None:
                W = 1.0
            if V is None:
                V = 1.0
            if alpha is None:
                alpha = ( n * H0 * H1**n * V / W ) / ( 1.0 + n * H0 * H1**n * V / W )
            if alpha1m is None:
                alpha1m = 1.0 / ( 1.0 + n * H0 * H1**n * V / W )
        self.H0 = H0
        self.H1 = H1
        self.W = W
        self.V = V
        self.alpha = alpha
        self.alpha1m = alpha1m

        if end + rand == 1.0:
            self.end = end
            self.rand = rand
        elif end != 1.0:
            self.end = end
            self.rand = 1.0 - end
        elif rand != 0.0:
            self.end = 1.0 - rand
            self.rand = rand

        self.solver = None

        return

    def get_part( self, n=None, rho=None, rho_M=None, H0=None, H1=None, gtol=1e-6, alpha_only=True ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if rho_M is None:
            rho_M = self.rho_M
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        W = self.get_melt(n, rho, rho_M, H0, H1, gtol)
        V = 1.0 - W / rho_M

        An = n * H0 * H1**n
        alpha = ( An * V ) / ( W + An * V )
        alpha1m = ( W ) / ( W + An * V )

        if alpha_only:
            return alpha, alpha1m
        else:
            return W, V, alpha, alpha1m

    def get_discrete_melt( self, n=None, rho=None, rho_M=None, H0=None, H1=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if rho_M is None:
            rho_M = self.rho_M
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        An = n * H0 * H1**n

        def fun(x):
            W = numpy.exp(x)
            dW = numpy.sum( ( n * rho ) * ( W ) / ( W + An * ( 1.0 - W / rho_M ) ) ) - W
            dWdx = numpy.sum( ( n * rho ) * ( An * W ) / ( W + An * ( 1.0 - W / rho_M ) )**2 ) - W
            f = dW**2
            dfdx = 2.0 * dW * dWdx
            return f, dfdx

        W = numpy.sum( ( n * rho ) / ( 1.0 + An ) )
        if W > 0.0:
            solver = minimize(fun, numpy.log(W), method='BFGS', jac=True, options={'gtol': gtol})
            W = numpy.exp(solver.x)

        return W

    def get_continuum_melt( self, n=None, rho=None, rho_M=None, H0=None, H1=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if rho_M is None:
            rho_M = self.rho_M
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        An = n * H0 * H1**n
        dn = n[1] - n[0]

        def fun(x):
            W = numpy.exp(x)
            g = ( n * rho ) * ( W ) / ( W + An * ( 1.0 - W / rho_M ) )
            dW = 0.5 * numpy.sum( g[1:] + g[:-1] ) * dn - W
            g = ( n * rho ) * ( An * W ) / ( W + An * ( 1.0 - W / rho_M ) )**2
            dWdx = 0.5 * numpy.sum( g[1:] + g[:-1] ) * dn - W
            f = dW**2
            dfdx = 2.0 * dW * dWdx
            return f, dfdx

        g = ( n * rho ) / ( 1.0 + An )
        W = 0.5 * numpy.sum( g[1:] + g[:-1]) * dn
        if W > 0.0:
            solver = minimize(fun, numpy.log(W), method='BFGS', jac=True, options={'gtol': gtol})
            W = numpy.exp(solver.x)

        return W

    def get_logn_melt( self, n=None, rho=None, rho_M=None, H0=None, H1=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if rho_M is None:
            rho_M = self.rho_M
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        An = n * H0 * H1**n
        logn = numpy.log(n)
        dlogn = logn[1] - logn[0]

        def fun(x):
            W = numpy.exp(x)
            g = ( n**2 * rho ) * ( W ) / ( W + An * ( 1.0 - W / rho_M ) )
            dW = 0.5 * numpy.sum( g[1:] + g[:-1] ) * dlogn - W
            g = ( n**2 * rho ) * ( An * W ) / ( W + An * ( 1.0 - W / rho_M ) )**2
            dWdx = 0.5 * numpy.sum( g[1:] + g[:-1] ) * dlogn - W
            f = dW**2
            dfdx = 2.0 * dW * dWdx
            return f, dfdx

        g = ( n**2 * rho ) / ( 1.0 + An )
        W = 0.5 * numpy.sum( g[1:] + g[:-1] ) * dlogn
        if W > 0.0:
            solver = minimize(fun, numpy.log(W), method='BFGS', jac=True, options={'gtol': gtol})
            W = numpy.exp(solver.x)

        return W

    def get_func( self, n=None, rho=None, alpha1m=None, rho_M=None, H0=None, H1=None, end=None, rand=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_M is None:
            rho_M = self.rho_M
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if end is None:
            end = self.end
        if rand is None:
            rand = self.rand

        if alpha1m is None:
            _, alpha1m = self.get_part(n, rho, rho_M, H0, H1, gtol)

        rate = self.get_rate(n, rho, alpha1m, end, rand)

        func = rate

        return func

    def get_discrete_rate( self, n=None, rho=None, alpha1m=None, end=None, rand=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m
        if end is None:
            end = self.end
        if rand is None:
            rand = self.rand

        f = alpha1m * rho

        df = numpy.zeros_like(f)
        if end != 0.0:
            df[1:-1] = f[2:] - f[1:-1]
            df[0   ] = f[1 ] - f[0   ]
            df[  -1] =       - f[  -1]

        sf = numpy.zeros_like(f)
        if rand != 0.0:
            sf[-2::-1] = numpy.cumsum(f[:0:-1])
            sf[-1    ] = 0.0

        rate = ( end ) * ( 1.0 * df ) + ( rand ) * ( 2.0 * sf - ( n - 1.0 ) * f )

        return rate

    def get_continuum_rate( self, n=None, rho=None, alpha1m=None, end=None, rand=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m
        if end is None:
            end = self.end
        if rand is None:
            rand = self.rand

        f = alpha1m * rho
        dn = n[1] - n[0]

        dfdn = numpy.zeros_like(f)
        if end != 0.0:
            dfdn[1:-1] = ( f[2:  ] - f[ :-2] ) / ( 2.0 * dn )
            #dfdn[0   ] = ( f[1   ] - f[0   ] ) / ( dn )
            #dfdn[  -1] = ( f[  -1] - f[  -2] ) / ( dn )
            dfdn[0   ] = dfdn[1   ]
            dfdn[  -1] = dfdn[  -2]
            #dfdn[ 0] = 2.0 * dfdn[ 1] - dfdn[ 2]
            #dfdn[-1] = 2.0 * dfdn[-2] - dfdn[-3]

        d2fdn2 = numpy.zeros_like(f)
        if end != 0.0:
            d2fdn2[1:-1] = ( f[2:  ] - 2.0 * f[1:-1] + f[ :-2] ) / ( dn**2.0 )
            #d2fdn2[0   ] = 0.0
            #d2fdn2[  -1] = 0.0
            d2fdn2[0   ] = d2fdn2[1   ]
            d2fdn2[  -1] = d2fdn2[  -2]
            #d2fdn2[ 0] = 2.0 * d2fdn2[ 1] - d2fdn2[ 2]
            #d2fdn2[-1] = 2.0 * d2fdn2[-2] - d2fdn2[-3]

        sf = numpy.zeros_like(f)
        if rand != 0.0:
            g = f
            sf[-2::-1] = 0.5 * numpy.cumsum( g[:0:-1] + g[-2::-1] ) * dn
            sf[-1    ] = 0.0
            #sf[-1    ] = sf[-2]
            #sf[-1    ] = 2.0 * sf[-2] - sf[-3]

        rate = ( end ) * ( 1.0 * dfdn + 0.5 * d2fdn2 ) + ( rand ) * ( 2.0 * sf - n * f )

        return rate

    def get_logn_rate( self, n=None, rho=None, alpha1m=None, end=None, rand=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m
        if end is None:
            end = self.end
        if rand is None:
            rand = self.rand

        f = alpha1m * rho
        logn = numpy.log(n)
        dlogn = logn[1] - logn[0]

        dfdr = numpy.zeros_like(f)
        if end != 0.0:
            dfdr[1:-1] = ( f[2:  ] - f[ :-2] ) / ( 2.0 * dlogn )
            dfdr[0   ] = ( f[1   ] - f[0   ] ) / ( dlogn )
            dfdr[  -1] = ( f[  -1] - f[  -2] ) / ( dlogn )
            #dfdr[0   ] = dfdr[1   ]
            #dfdr[  -1] = dfdr[  -2]
            #dfdr[ 0] = 2.0 * dfdr[ 1] - dfdr[ 2]
            #dfdr[-1] = 2.0 * dfdr[-2] - dfdr[-3]

        d2fdr2 = numpy.zeros_like(f)
        if end != 0.0:
            d2fdr2[1:-1] = ( f[2:  ] - 2.0 * f[1:-1] + f[ :-2] ) / ( dlogn**2.0 )
            d2fdr2[0   ] = 0.0
            d2fdr2[  -1] = 0.0
            #d2fdr2[0   ] = d2fdr2[1   ]
            #d2fdr2[  -1] = d2fdr2[  -2]
            #d2fdr2[ 0] = 2.0 * d2fdr2[ 1] - d2fdr2[ 2]
            #d2fdr2[-1] = 2.0 * d2fdr2[-2] - d2fdr2[-3]

        sf = numpy.zeros_like(f)
        if rand != 0.0:
            g = n * f
            sf[-2::-1] = 0.5 * numpy.cumsum( g[:0:-1] + g[-2::-1] ) * dlogn
            sf[-1    ] = 0.0
            #sf[-1    ] = sf[-2]
            #sf[-1    ] = 2.0 * sf[-2] - sf[-3]

        rate = ( end ) * ( ( 1.0/n - 0.5/n**2 ) * dfdr + ( 0.5/n**2 ) * d2fdr2 ) + ( rand ) * ( 2.0 * sf - n * f )

        return rate

    def get_jac( self, n=None, rho=None, W=None, alpha1m=None, rho_M=None, H0=None, H1=None, rand=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if W is None:
            W = self.W
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_M is None:
            rho_M = self.rho_M
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if rand is None:
            rand = self.rand

        if W is None or alpha1m is None:
            W, _, _, alpha1m = self.get_part(n, rho, rho_M, H0, H1, gtol, alpha_only=False)

        deriv = self.get_deriv(n, rho, W, alpha1m, H0, H1, rand)

        jac = deriv

        return jac

    def get_discrete_aux( self, n=None, rho=None, W=None, alpha1m=None, H0=None, H1=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if W is None:
            W = self.W
        if alpha1m is None:
            alpha1m = self.alpha1m
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        A = n * H0 * H1**n * ( alpha1m / W )**2
        B = n * alpha1m
        C = n * rho

        b = numpy.outer(A, B)
        a = numpy.identity(len(n)) - numpy.outer(A, C)
        x = numpy.linalg.solve(a, b)

        return x

    def get_discrete_deriv( self, n=None, rho=None, W=None, alpha1m=None, H0=None, H1=None, rand=None, rate_only=True ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if W is None:
            W = self.W
        if alpha1m is None:
            alpha1m = self.alpha1m
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if rand is None:
            rand = self.rand

        # f = alpha1m * rho
        aux = self.get_discrete_aux(n, rho, W, alpha1m, H0, H1)
        f = numpy.diag(alpha1m) + ( aux.T * rho ).T

        df = numpy.zeros_like(f)
        if rand != 1.0:
            df[1:-1] = f[2:] - f[1:-1]
            df[0   ] = f[1 ] - f[0   ]
            df[  -1] =       - f[  -1]

        sf = numpy.zeros_like(f)
        if rand != 0.0:
            sf[-2::-1] = numpy.cumsum(f[:0:-1], axis=0)
            sf[-1    ] = 0.0

        deriv = ( 1.0 - rand ) * ( 1.0 * df.T ).T + ( rand ) * ( 2.0 * sf.T - ( n - 1.0 ) * f.T ).T

        if rate_only:
            return deriv
        else:
            return aux, deriv

    def get_continuum_aux( self, n=None, rho=None, W=None, alpha1m=None, H0=None, H1=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if W is None:
            W = self.W
        if alpha1m is None:
            alpha1m = self.alpha1m
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        dn = n[1] - n[0]
        w = numpy.ones_like(n)
        w[0] = w[-1] = 0.5

        A = n * H0 * H1**n * ( alpha1m / W )**2
        B = w * n * alpha1m * dn
        C = w * n * rho * dn

        b = numpy.outer(A, B)
        a = numpy.identity(len(n)) - numpy.outer(A, C)
        x = numpy.linalg.solve(a, b)

        return x

    def get_continuum_deriv( self, n=None, rho=None, W=None, alpha1m=None, H0=None, H1=None, rand=None, rate_only=True ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if W is None:
            W = self.W
        if alpha1m is None:
            alpha1m = self.alpha1m
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if rand is None:
            rand = self.rand

        # f = alpha1m * rho
        dn = n[1] - n[0]

        aux = self.get_continuum_aux(n, rho, W, alpha1m, H0, H1)
        f = numpy.diag(alpha1m) + ( aux.T * rho ).T

        dfdn = numpy.zeros_like(f)
        if rand != 1.0:
            dfdn[1:-1] = ( f[2:  ] - f[ :-2] ) / ( 2.0 * dn )
            #dfdn[0   ] = ( f[1   ] - f[0   ] ) / ( dn )
            #dfdn[  -1] = ( f[  -1] - f[  -2] ) / ( dn )
            dfdn[0   ] = dfdn[1   ]
            dfdn[  -1] = dfdn[  -2]
            #dfdn[ 0] = 2.0 * dfdn[ 1] - dfdn[ 2]
            #dfdn[-1] = 2.0 * dfdn[-2] - dfdn[-3]

        d2fdn2 = numpy.zeros_like(f)
        if rand != 1.0:
            d2fdn2[1:-1] = ( f[2:  ] - 2.0 * f[1:-1] + f[ :-2] ) / ( dn**2.0 )
            #d2fdn2[0   ] = 0.0
            #d2fdn2[  -1] = 0.0
            d2fdn2[0   ] = d2fdn2[1   ]
            d2fdn2[  -1] = d2fdn2[  -2]
            #d2fdn2[ 0] = 2.0 * d2fdn2[ 1] - d2fdn2[ 2]
            #d2fdn2[-1] = 2.0 * d2fdn2[-2] - d2fdn2[-3]

        sf = numpy.zeros_like(f)
        if rand != 0.0:
            g = f
            sf[-2::-1] = 0.5 * numpy.cumsum( g[:0:-1] + g[-2::-1], axis=0 ) * dn
            sf[-1    ] = 0.0
            #sf[-1    ] = sf[-2]
            #sf[-1    ] = 2.0 * sf[-2] - sf[-3]

        deriv = ( 1.0 - rand ) * ( 1.0 * dfdn.T + 0.5 * d2fdn2.T ).T + ( rand ) * ( 2.0 * sf.T - n * f.T ).T

        if rate_only:
            return deriv
        else:
            return aux, deriv

    def get_logn_aux( self, n=None, rho=None, W=None, alpha1m=None, H0=None, H1=None ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if W is None:
            W = self.W
        if alpha1m is None:
            alpha1m = self.alpha1m
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        logn = numpy.log(n)
        dlogn = logn[1] - logn[0]
        w = numpy.ones_like(n)
        w[0] = w[-1] = 0.5

        A = n * H0 * H1**n * ( alpha1m / W )**2
        B = w * n**2 * alpha1m * dlogn
        C = w * n**2 * rho * dlogn

        b = numpy.outer(A, B)
        a = numpy.identity(len(n)) - numpy.outer(A, C)
        x = numpy.linalg.solve(a, b)

        return x

    def get_logn_deriv( self, n=None, rho=None, W=None, alpha1m=None, H0=None, H1=None, rand=None, rate_only=True ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if W is None:
            W = self.W
        if alpha1m is None:
            alpha1m = self.alpha1m
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if rand is None:
            rand = self.rand

        # f = alpha1m * rho
        logn = numpy.log(n)
        dlogn = logn[1] - logn[0]

        aux = self.get_logn_aux(n, rho, W, alpha1m, H0, H1)
        f = numpy.diag(alpha1m) + ( aux.T * rho ).T

        dfdr = numpy.zeros_like(f)
        if rand != 1.0:
            dfdr[1:-1] = ( f[2:  ] - f[ :-2] ) / ( 2.0 * dlogn )
            dfdr[0   ] = ( f[1   ] - f[0   ] ) / ( dlogn )
            dfdr[  -1] = ( f[  -1] - f[  -2] ) / ( dlogn )
            #dfdr[0   ] = dfdr[1   ]
            #dfdr[  -1] = dfdr[  -2]
            #dfdr[ 0] = 2.0 * dfdr[ 1] - dfdr[ 2]
            #dfdr[-1] = 2.0 * dfdr[-2] - dfdr[-3]

        d2fdr2 = numpy.zeros_like(f)
        if rand != 1.0:
            d2fdr2[1:-1] = ( f[2:  ] - 2.0 * f[1:-1] + f[ :-2] ) / ( dlogn**2.0 )
            d2fdr2[0   ] = 0.0
            d2fdr2[  -1] = 0.0
            #d2fdr2[0   ] = d2fdr2[1   ]
            #d2fdr2[  -1] = d2fdr2[  -2]
            #d2fdr2[ 0] = 2.0 * d2fdr2[ 1] - d2fdr2[ 2]
            #d2fdr2[-1] = 2.0 * d2fdr2[-2] - d2fdr2[-3]

        sf = numpy.zeros_like(f)
        if rand != 0.0:
            g = ( n * f.T ).T
            sf[-2::-1] = 0.5 * numpy.cumsum( g[:0:-1] + g[-2::-1], axis=0 ) * dlogn
            sf[-1    ] = 0.0
            #sf[-1    ] = sf[-2]
            #sf[-1    ] = 2.0 * sf[-2] - sf[-3]

        deriv = ( 1.0 - rand ) * ( ( 1.0/n - 0.5/n**2 ) * dfdr.T + ( 0.5/n**2 ) * d2fdr2.T ).T + ( rand ) * ( 2.0 * sf.T - n * f.T ).T

        if rate_only:
            return deriv
        else:
            return aux, deriv

    def solve( self, t, n=None, rho=None, alpha1m=None, rho_M=None, H0=None, H1=None, end=None, rand=None, gtol=1e-6, rtol=1e-6, atol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_M is None:
            rho_M = self.rho_M
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if end is None:
            end = self.end
        if rand is None:
            rand = self.rand

        def fun( t, y ):
            return self.get_func(n, y, alpha1m, rho_M, H0, H1, end, rand, gtol)

        solver = solve_ivp(fun, [0.0, t], rho, method='BDF', rtol=rtol, atol=atol)

        self.solver = solver

        return solver.t, solver.y

    def postprocess( self, mode, t=None, n=None, rho=None, V=None, alpha=None, rho_M=None, H0=None, H1=None, state_cutoffs=[4.5, 16.5], temp=573.15, volume=1.0, mass=10.0, monomer=14.027, gtol=1e-6, renorm=False ):

        if t is None:
            if self.solver is None:
                t = None
            else:
                t = self.solver.t
        if n is None:
            n = self.n
        if rho is None:
            if self.solver is None:
                rho = self.rho
            else:
                rho = self.solver.y
        if V is None:
            V = self.V
        if alpha is None:
            alpha = self.alpha
        if rho_M is None:
            rho_M = self.rho_M
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1

        if V is None or alpha is None:
            if t is None:
                _, V, alpha, _ = self.get_part(n, rho, rho_M, H0, H1, gtol, alpha_only=False)
            else:
                V = numpy.zeros_like(t)
                alpha = numpy.zeros_like(rho)
                for i, _ in enumerate(t):
                    _, V[i], alpha[:, i], _ = self.get_part(n, rho[:, i], rho_M, H0, H1, gtol, alpha_only=False)

        if mode in ['dwdn', 'dwdlogn', 'dWdn', 'dWdlogn', 'dWdM', 'dWdlogM', 'dndn', 'dndlogn']:
            return convert_rho_to_y(mode, n, rho, mass, monomer)
        elif mode in ['D_n', 'D_logn', 'D_M', 'D_logM']:
            return get_dispersity(mode, t, n, rho, monomer)
        elif mode in ['dpdn', 'dpdlogn', 'dPdn', 'dPdlogn', 'dPdM', 'dPdlogM']:
            return get_pressure(mode, t, n, rho, V, alpha, temp, volume, mass, monomer)
        elif mode in ['rho_n', 'rho_logn', 'concs_n', 'concs_logn', 'w_n', 'w_logn', 'W_n', 'W_logn']:
            return get_states(mode, t, n, rho, state_cutoffs, mass, monomer, renorm)


class SemiBatchReactor(BatchReactor):

    def __init__( self, nmin=1, nmax=5, mesh=0, grid='discrete',
            rho=None, W=None, V=None, alpha=None, alpha1m=None, rho_M=None, H0=None, H1=None, fin=None, fout=None,
            concs=[0.0, 0.0, 0.0, 0.0, 1.0], influx=[0.0, 0.0, 0.0, 0.0, 0.0], outflux=[0.0, 0.0],
            temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0, end=1.0, rand=0.0 ):

        super().__init__(nmin, nmax, mesh, grid, rho, W, V, alpha, alpha1m, rho_M, H0, H1, concs, temp, volume, mass, monomer, dens, end, rand)

        if fin is None:
            n = self.n
            if grid == 'discrete':
                fin = numpy.array(influx) / numpy.inner(n, concs)
            elif grid == 'continuum':
                dn = n[1] - n[0]
                g = n * concs
                fin = numpy.array(influx) / ( 0.5 * numpy.sum( g[1:] + g[:-1] ) * dn )
            elif grid == 'logn':
                logn = numpy.log(n)
                dlogn = logn[1] - logn[0]
                g = n**2 * concs
                fin = numpy.array(influx) / ( 0.5 * numpy.sum( g[1:] + g[:-1] ) * dlogn )
        self.fin = fin

        if fout is None:
            fout = numpy.array(outflux)
        self.fout = fout

        return

    def get_func( self, n=None, rho=None, W=None, V=None, alpha=None, alpha1m=None, rho_M=None, H0=None, H1=None, fin=None, fout=None, end=None, rand=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if W is None:
            W = self.W
        if V is None:
            V = self.V
        if alpha is None:
            alpha = self.alpha
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_M is None:
            rho_M = self.rho_M
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if fin is None:
            fin = self.fin
        if fout is None:
            fout = self.fout
        if end is None:
            end = self.end
        if rand is None:
            rand = self.rand

        if W is None or V is None or alpha is None or alpha1m is None:
            W, V, alpha, alpha1m = self.get_part(n, rho, rho_M, H0, H1, gtol, alpha_only=False)

        rate = self.get_rate(n, rho, alpha1m, end, rand)

        func = rate + fin - fout[0] * alpha * rho / V - fout[1] * alpha1m * rho * rho_M / (W + (W == 0.0))

        return func

    def get_jac( self, n=None, rho=None, W=None, V=None, alpha=None, alpha1m=None, rho_M=None, H0=None, H1=None, fin=None, fout=None, rand=None, gtol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if W is None:
            W = self.W
        if V is None:
            V = self.V
        if alpha is None:
            alpha = self.alpha
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_M is None:
            rho_M = self.rho_M
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

        if W is None or V is None or alpha is None or alpha1m is None:
            W, V, alpha, alpha1m = self.get_part(n, rho, rho_M, H0, H1, gtol, alpha_only=False)

        aux, deriv = self.get_deriv(n, rho, W, alpha1m, H0, H1, rand, rate_only=False)

        jac = deriv - fout[0] * ( numpy.diag(alpha) - ( aux.T * rho ).T ) / V - fout[1] * ( numpy.diag(alpha1m) + ( aux.T * rho ).T ) / (W + (W == 0.0))

        return jac

    def solve( self, t, n=None, rho=None, W=None, V=None, alpha=None, alpha1m=None, rho_M=None, H0=None, H1=None, fin=None, fout=None, end=None, rand=None, gtol=1e-6, rtol=1e-6, atol=1e-6 ):

        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if W is None:
            W = self.W
        if V is None:
            V = self.V
        if alpha is None:
            alpha = self.alpha
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_M is None:
            rho_M = self.rho_M
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if fin is None:
            fin = self.fin
        if fout is None:
            fout = self.fout
        if end is None:
            end = self.end
        if rand is None:
            rand = self.rand

        def fun( t, y ):
            return self.get_func(n, y, W, V, alpha, alpha1m, rho_M, H0, H1, fin, fout, end, rand, gtol)

        solver = solve_ivp(fun, [0.0, t], rho, method='BDF', rtol=rtol, atol=atol)

        self.solver = solver

        return solver.t, solver.y

    def cointegrate( self, t=None, n=None, rho=None, W=None, V=None, alpha=None, alpha1m=None, rho_M=None, H0=None, H1=None, fin=None, fout=None, end=None, rand=None, gtol=1e-6, integrals_only=True ):

        if t is None:
            t = self.solver.t
        if n is None:
            n = self.n
        if rho is None:
            rho = self.solver.y
        if W is None:
            W = self.W
        if V is None:
            V = self.V
        if alpha is None:
            alpha = self.alpha
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_M is None:
            rho_M = self.rho_M
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if fin is None:
            fin = self.fin
        if fout is None:
            fout = self.fout
        if end is None:
            end = self.end
        if rand is None:
            rand = self.rand

        dt = t[1:] - t[:-1]

        g = numpy.zeros_like(rho)
        gin = numpy.zeros_like(rho)
        gout = numpy.zeros_like(rho)

        for i, _ in enumerate(t):

            w = W
            v = V
            alp = alpha
            a1m = alpha1m

            if w is None or v is None or alp is None or a1m is None:
                w, v, alp, a1m = self.get_part(n, rho[:, i], rho_M, H0, H1, gtol, alpha_only=False)

            rate = self.get_rate(n, rho[:, i], a1m)

            g[:, i] = rate
            gin[:, i] = fin
            gout[:, i] = fout[0] * alp * rho[:, i] / v + fout[1] * a1m * rho[:, i] * rho_M / (w + (w == 0.0))

        G = numpy.zeros_like(rho)
        Gin = numpy.zeros_like(rho)
        Gout = numpy.zeros_like(rho)

        G[:, 1:] = 0.5 * numpy.cumsum( ( g[:, 1:] + g[:, :-1] ) * dt, axis=1)
        Gin[:, 1:] = 0.5 * numpy.cumsum( ( gin[:, 1:] + gin[:, :-1] ) * dt, axis=1)
        Gout[:, 1:] = 0.5 * numpy.cumsum( ( gout[:, 1:] + gout[:, :-1] ) * dt, axis=1)

        if integrals_only:
            return G, Gin, Gout
        else:
            return g, gin, gout, G, Gin, Gout


class CSTReactor(SemiBatchReactor):

    def __init__( self, nmin=1, nmax=5, mesh=0, grid='discrete',
            rho=None, W=None, V=None, alpha=None, alpha1m=None, rho_M=None, H0=None, H1=None, fin=None, fout=None,
            concs=[0.0, 0.0, 0.0, 0.0, 1.0], influx=[0.0, 0.0, 0.0, 0.0, 0.0], outflux=[0.0, 0.0],
            temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0, end=1.0, rand=0.0 ):

        super().__init__(nmin, nmax, mesh, grid, rho, W, V, alpha, alpha1m, rho_M, H0, H1, fin, fout, concs, influx, outflux, temp, volume, mass, monomer, dens, end, rand)

        self.grid = grid

        ##n = self.n
        ##fin = self.fin
        ##if grid == 'discrete':
        ##    Da = numpy.inner(n, fin)
        ##elif grid == 'continuum':
        ##    dn = n[1] - n[0]
        ##    g = n * fin
        ##    Da = 0.5 * numpy.sum( g[1:] + g[:-1] ) * dn
        ##elif grid == 'logn':
        ##    logn = numpy.log(n)
        ##    dlogn = logn[1] - logn[0]
        ##    g = n**2 * fin
        ##    Da = 0.5 * numpy.sum( g[1:] + g[:-1] ) * dlogn
        ##self.Da = 1.0 / Da

        return

    def solve( self, grid=None, n=None, rho=None, W=None, V=None, alpha=None, alpha1m=None, rho_M=None, H0=None, H1=None, fin=None, fout=None, end=None, rand=None, Gmax=0.97725, dmax=0.1, gtol=1e-6, rtol=1e-6, atol=1e-6 ):

        if grid is None:
            grid = self.grid
        if n is None:
            n = self.n
        if rho is None:
            rho = self.rho
        if W is None:
            W = self.W
        if V is None:
            V = self.V
        if alpha is None:
            alpha = self.alpha
        if alpha1m is None:
            alpha1m = self.alpha1m
        if rho_M is None:
            rho_M = self.rho_M
        if H0 is None:
            H0 = self.H0
        if H1 is None:
            H1 = self.H1
        if fin is None:
            fin = self.fin
        if fout is None:
            fout = self.fout
        if end is None:
            end = self.end
        if rand is None:
            rand = self.rand

        if grid == 'discrete':
            g = n * rho
            G = numpy.cumsum(g)
        elif grid == 'continuum':
            dn = n[1] - n[0]
            g = n * rho
            G = numpy.zeros_like(rho)
            G[1:] = 0.5 * numpy.cumsum( g[1:] + g[:-1] ) * dn
        elif grid == 'logn':
            logn = numpy.log(n)
            dlogn = logn[1] - logn[0]
            g = n**2 * rho
            G = numpy.zeros_like(rho)
            G[1:] = 0.5 * numpy.cumsum( g[1:] + g[:-1] ) * dlogn

        #Gmax = 0.97725
        #dmax = 0.1
        imax = numpy.max(numpy.argwhere( G < Gmax ))
        nmax = n[imax] + ( n[imax+1] - n[imax] ) / ( G[imax+1] - G[imax] ) * ( Gmax - G[imax] )
        tmax = dmax / ( end / nmax + rand )

        y = numpy.transpose([ numpy.full_like(rho, numpy.inf), rho ])
        while numpy.any( numpy.abs( y[:, -1] - y[:, 0] ) > atol + rtol * numpy.abs( y[:, -1] ) ):
            t, y = super().solve(tmax, n, y[:, -1], W, V, alpha, alpha1m, rho_M, H0, H1, fin, fout, end, rand, gtol, rtol, atol)

        return y[:, -1]

