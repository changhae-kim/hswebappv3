import itertools
import numpy
import os, sys

from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 14})

from reactor import CSTReactor
from utils import plot_curves, plot_two_axes, plot_populations


volume  = 1.0
monomer = 14.027
dens    = 920.0

grid = 'logn'

temps  = [ 423.15, 473.15, 523.15, 573.15 ]
masses = [ 1.0, 10.0, 100.0, 1000.0 ]
mus    = [3.0]
sigmas = [0.1]

fluxes = [ 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0 ]

prune = 10

for temp, mass, mu, sigma, flux in itertools.product(temps, masses, mus, sigmas, fluxes):

    basename = '{temp:g}K_{mass:g}gpL_mu{mu:g}_sigma{sigma:g}_flux{flux:g}'.format(temp=temp, mass=mass, mu=mu, sigma=sigma, flux=flux)

    nmin = 1e-2
    nmax = 10.0**(mu+4.0*sigma)
    mesh = int(100.0*(numpy.log10(nmax)-numpy.log10(nmin)))
    n = numpy.logspace(numpy.log10(nmin), numpy.log10(nmax), mesh)
    concs = (1.0/n**2)*numpy.exp(-0.5*((numpy.log10(n)-mu)/(sigma))**2)
    tmax = 0.2+0.2*numpy.log10(mass)

    influx = numpy.zeros_like(concs)
    outflux = [flux, 0.0]

    reactor = CSTReactor(nmin=nmin, nmax=nmax, mesh=mesh, grid=grid, influx=influx, outflux=outflux, concs=concs, temp=temp, volume=volume, mass=mass, monomer=monomer, dens=dens, rand=1.0)
    n = reactor.n
    alpha = reactor.alpha
    alpha1m = reactor.alpha1m
    # alpha = numpy.zeros_like(reactor.alpha)
    # alpha1m = numpy.ones_like(reactor.alpha1m)

    if not os.path.exists('t_'+basename+'.npy') or not os.path.exists('rho_'+basename+'.npy'):
        t, rho = reactor.solve(tmax, alpha=alpha, alpha1m=alpha1m, gtol=1e-12, rtol=1e-12, atol=1e-12)
        numpy.save('t_'+basename+'.npy', t)
        numpy.save('rho_'+basename+'.npy', rho)
    else:
        t = numpy.load('t_'+basename+'.npy')
        rho = numpy.load('rho_'+basename+'.npy')

    n, dwdn = reactor.postprocess('dwdn', rho=rho)
    n, dwdlogn = reactor.postprocess('dwdlogn', rho=rho)

    # plot_populations(t, n, rho, alpha1m, 'Chain Length', 'Chain Concentration', 'rho_n_'+basename+'.png', prune=prune)
    # plot_populations(t, n, dwdn, alpha1m, 'Chain Length', r'$d\widetilde{W}/d{n}$', 'dwdn_'+basename+'.png', prune=prune)
    # plot_populations(t, n, dwdlogn, alpha1m, 'Chain Length', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png', prune=prune)

    # plot_populations(t, n, rho, alpha1m, 'Chain Length', 'Chain Concentration', 'rho_n_'+basename+'.png', prune=prune, xlim=[1.0, nmax])
    # plot_populations(t, n, dwdlogn, alpha1m, 'Chain Length', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png', prune=prune, xlim=[1.0, nmax])
    # plot_populations(t, n, dwdlogn, alpha1m, 'Chain Length', r'$d\widetilde{W}/d\ln{n}$', 'dwdlogn_'+basename+'.png', prune=prune, xlim=[1.0, nmax])
    # plot_populations(t, n, dwdlogn*numpy.log(10.0), alpha1m, 'Chain Length', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png', prune=prune, xlim=[1.0, nmax])

    # plot_populations(t, n, rho, alpha1m, 'Chain Length', 'Chain Concentration', 'rho_n_'+basename+'.png', prune=prune, xscale='linear', xlim=[1.0, 29.0])
    # plot_populations(t, n, dwdn, alpha1m, 'Chain Length', r'$d\widetilde{W}/d{n}$', 'dwdn_'+basename+'.png', prune=prune, xscale='linear', xlim=[1.0, 29.0])

    G, Gin, Gout = reactor.cointegrate(t=t, rho=rho, alpha=alpha, alpha1m=alpha1m)

    #n, dwdn = reactor.postprocess('dwdn', rho=G)
    #n, dwdlogn = reactor.postprocess('dwdlogn', rho=G)
    #M, dWdM = reactor.postprocess('dWdM', rho=G)
    #M, dWdlogM = reactor.postprocess('dWdlogM', rho=G)

    rho2 = ( rho[:, 0] + ( G + Gin - Gout ).T ).T
    n, dwdn2 = reactor.postprocess('dwdn', rho=rho2)
    n, dwdlogn2 = reactor.postprocess('dwdlogn', rho=rho2)

    # plot_populations(t, n, rho2, alpha1m, 'Chain Length', 'Chain Concentration', 'rho_n_'+basename+'.png', prune=prune, xlim=[1.0, nmax])
    # plot_populations(t, n, dwdn2, alpha1m, 'Chain Length', r'$d\widetilde{W}/d{n}$', 'dwdn_'+basename+'.png', prune=prune, xlim=[1.0, nmax])
    # plot_populations(t, n, dwdlogn2, alpha1m, 'Chain Length', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png', prune=prune, xlim=[1.0, nmax])

    n, dwdn_out = reactor.postprocess('dwdn', rho=Gout)
    n, dwdlogn_out = reactor.postprocess('dwdlogn', rho=Gout)

    # plot_populations(t, n, Gout, alpha1m, 'Chain Length', 'Chain Concentration', 'rho_n_'+basename+'.png', prune=prune, xlim=[1.0, nmax])
    # plot_populations(t, n, dwdlogn_out, alpha1m, 'Chain Length', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png', prune=prune, xlim=[1.0, nmax])
    # plot_populations(t, n, dwdlogn_out, alpha1m, 'Chain Length', r'$d\widetilde{W}/d\ln{n}$', 'dwdlogn_'+basename+'.png', prune=prune, xlim=[1.0, nmax])
    # plot_populations(t, n, dwdlogn_out*numpy.log(10.0), alpha1m, 'Chain Length', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png', prune=prune, xlim=[1.0, nmax])

    # plot_populations(t, n, Gout, alpha1m, 'Chain Length', 'Chain Concentration', 'rho_n_'+basename+'.png', prune=prune, xscale='linear', xlim=[1.0, 29.0])
    # plot_populations(t, n, dwdn_out, alpha1m, 'Chain Length', r'$d\widetilde{W}/d{n}$', 'dwdn_'+basename+'.png', prune=prune, xscale='linear', xlim=[1.0, 29.0])

    nn, nw, Dn = reactor.postprocess('D_logn', t=t, rho=Gout)

    # plot_two_axes(t, nn, Dn, r'$\overline{n}$', 'PDI', 'disp_'+basename+'.png', y1scale='linear')

    '''
    logn = numpy.log(n)
    dlogn = logn[1] - logn[0]

    n10 = numpy.count_nonzero(n < 10.0)

    g0 = n**1 * Gout.T
    g1 = n**2 * Gout.T

    G0 = 0.5 * numpy.sum(g0[-1, 1:] + g0[-1, :-1]) * dlogn
    G1 = 0.5 * numpy.sum(g1[-1, 1:] + g1[-1, :-1]) * dlogn

    F0 = 0.5 * numpy.sum(g0[-1, 1:n10+1] + g0[-1, :n10]) * dlogn
    F1 = 0.5 * numpy.sum(g1[-1, 1:n10+1] + g1[-1, :n10]) * dlogn

    H0 = 0.5 * numpy.sum(g0[-1, n10+1:] + g0[-1, n10:-1]) * dlogn
    H1 = 0.5 * numpy.sum(g1[-1, n10+1:] + g1[-1, n10:-1]) * dlogn

    print('{:40s} {:.3f} {:.3f} {:.3f} {:.3f}'.format( basename, F0/G0, F1/G1, H0/G0, H1/G1 ))
    '''

    p, dpdn = reactor.postprocess('dpdn', t=t, rho=rho)
    p, dpdlogn = reactor.postprocess('dpdlogn', t=t, rho=rho)
    P, dPdn = reactor.postprocess('dPdn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)
    P, dPdlogn = reactor.postprocess('dPdlogn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)

    # plot_populations(t, n, dpdlogn*numpy.log(10.0), alpha1m, 'Chain Length', r'$d\widetilde{P}/d\log{n}$', 'dpdlogn_'+basename+'.png', prune=prune, xlim=[1.0, nmax])
    # plot_populations(t, n, dpdn, alpha1m, 'Chain Length', r'$d\widetilde{P}/d{n}$', 'dpdn_'+basename+'.png', prune=prune, xscale='linear', xlim=[1.0, 29.0])
    # plot_curves(t, [P], 'Pressure (atm)', 'pressure_'+basename+'.png')

    rho_g, rho_l, rho_s = reactor.postprocess('rho_logn', t=t, rho=reactor.rho[:, numpy.newaxis]+G+Gout)
    wg, wl, ws = reactor.postprocess('w_logn', t=t, rho=reactor.rho[:, numpy.newaxis]+G+Gout)

    # plot_curves(t, [rho_s, rho_l, rho_g], 'Chain Concentration', 'rho_gls_'+basename+'.png', labels=['Solid (C$_{17'+u'\u2010'+'\infty}$)', 'Liquid (C$_{5'+u'\u2010'+'16}$)', 'Gas (C$_{1'+u'\u2010'+'4}$)'])
    # plot_curves(t, [ws, wl, wg], 'Mass Fraction', 'w_gls_'+basename+'.png', labels=['Solid (C$_{17'+u'\u2010'+'\infty}$)', 'Liquid (C$_{5'+u'\u2010'+'16}$)', 'Gas (C$_{1'+u'\u2010'+'4}$)'])

