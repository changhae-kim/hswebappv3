import itertools
import numpy
import os, sys

from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 14})

from reactor import SemiBatchReactor
from utils import plot_curves, plot_two_axes, plot_populations, plot_colormap


volume  = 1.0
monomer = 14.027
dens    = 940.0

grid = 'logn'

N = 50
temps  = [ 573.15 ]
masses = numpy.logspace( 0, 2, N+1 )
mus    = [ 3.0 ]
sigmas = [ 0.1 ]
fluxes = numpy.logspace( 0, 2, N+1 )

Mn = []
Mw = []
DDn = []
PP = []
Rho_G = []
Rho_L = []
Rho_S = []
WG = []
WL = []
WS = []

for temp, mass, mu, sigma, flux in itertools.product(temps, masses, mus, sigmas, fluxes):

    basename = '{temp:g}K_{mass:g}gpL_mu{mu:g}_sigma{sigma:g}_flux{flux:g}'.format(temp=temp, mass=mass, mu=mu, sigma=sigma, flux=flux)

    nmin = 1e-2
    nmax = 10.0**(mu+4.0*sigma)
    mesh = int(100.0*(numpy.log10(nmax)-numpy.log10(nmin)))
    n = numpy.logspace(numpy.log10(nmin), numpy.log10(nmax), mesh)
    concs = (1.0/n**2)*numpy.exp(-0.5*((numpy.log10(n)-mu)/(sigma))**2)
    tmax = 1.0

    influx = numpy.zeros_like(concs)
    outflux = [flux, 0.0]

    reactor = SemiBatchReactor(nmin=nmin, nmax=nmax, mesh=mesh, grid=grid, influx=influx, outflux=outflux, concs=concs, temp=temp, volume=volume, mass=mass, monomer=monomer, dens=dens, rand=1.0)
    n = reactor.n
    alpha, alpha1m = reactor.get_part()
    reactor.W = None
    reactor.V = None
    reactor.alpha = None
    reactor.alpha1m = None

    if not os.path.exists('t_'+basename+'.npy') or not os.path.exists('rho_'+basename+'.npy'):
        t, rho = reactor.solve(tmax, gtol=1e-12, rtol=1e-12, atol=1e-12)
        numpy.save('t_'+basename+'.npy', t)
        numpy.save('rho_'+basename+'.npy', rho)
    else:
        t = numpy.load('t_'+basename+'.npy')
        rho = numpy.load('rho_'+basename+'.npy')

    G, Gin, Gout = reactor.cointegrate(t=t, rho=rho)
    rho1 = ( reactor.rho + ( G + Gin - Gout ).T ).T
    rho2 = Gout
    rho3 = ( reactor.rho + ( G + Gin ).T ).T

    P, dPdn = reactor.postprocess('dPdn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)
    P, dPdlogn = reactor.postprocess('dPdlogn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)
    nn, nw, Dn = reactor.postprocess('D_logn', t=t, rho=rho3)
    rho_g, rho_l, rho_s = reactor.postprocess('rho_logn', t=t, rho=rho3)
    wg, wl, ws = reactor.postprocess('w_logn', t=t, rho=rho3)

    Mn.append(nn[-1])
    Mw.append(nw[-1])
    DDn.append(Dn[-1])
    PP.append(P.max())
    Rho_G.append(rho_g[-1])
    Rho_L.append(rho_l[-1])
    Rho_S.append(rho_s[-1])
    WG.append(wg[-1])
    WL.append(wl[-1])
    WS.append(ws[-1])

data = [
        [Mn,    'Mn',    'linear', None, None, r'$\widetilde{M}_N$',                    ],
        [Mw,    'Mw',    'linear', None, None, r'$\widetilde{M}_W$',                    ],
        [DDn,   'DDn',   'linear', None, None, '$'+u'\u0110'+'$',                       ],
        [PP,    'PP',    'log',    None, None, 'Peak Hydrocarbon Pressure (atm)',       ],
        [Rho_G, 'Rho_G', 'linear', None, None, 'Mole Fraction of C$_{1}$$_{-}$$_{4}$',  ],
        [Rho_L, 'Rho_L', 'linear', None, None, 'Mole Fraction of C$_{5}$$_{-}$$_{16}$', ],
        [Rho_S, 'Rho_S', 'linear', None, None, 'Mole Fraction of C$_{17}$$_{+}$',       ],
        [WG,    'WG',    'linear', None, None, 'Mass Fraction of C$_{1}$$_{-}$$_{4}$',  ],
        [WL,    'WL',    'linear', None, None, 'Mass Fraction of C$_{5}$$_{-}$$_{16}$', ],
        [WS,    'WS',    'linear', None, None, 'Mass Fraction of C$_{17}$$_{+}$',       ],
        ]

for i, _ in enumerate(data):
    if not os.path.exists(data[i][1]+'.npy'):
        data[i][0] = numpy.array(data[i][0]).reshape(N+1, N+1)
        numpy.save(data[i][1]+'.npy', data[i][0])
    else:
        data[i][0] = numpy.load(data[i][1]+'.npy')

for i, _ in enumerate(data):
    plot_colormap(fluxes, 1000.0/masses, data[i][0], r'$\mathrm{Da}^{-1}$', '$V/W$ (cm$^3$ g$^{-1}$)', data[i][-1], data[i][1]+'.png', xscale='log', yscale='log', zscale=data[i][2], zmin=data[i][3], zmax=data[i][4])

