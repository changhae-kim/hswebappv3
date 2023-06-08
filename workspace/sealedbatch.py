import itertools
import numpy
import os, sys

from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 14})

from reactor import BatchReactor
from utils import plot_curves, plot_two_axes, plot_populations


volume  = 1.0
monomer = 14.027
dens    = 920.0

grid = 'logn'

temps  = [ 423.15, 473.15, 523.15, 573.15 ]
masses = [ 1.0, 10.0, 100.0, 1000.0 ]
mus    = [ 2.0, 3.0, 4.0, 5.0 ]
sigmas = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ]

prune = 10

for temp, mass, mu, sigma in itertools.product(temps, masses, mus, sigmas):

    basename = '{temp:g}K_{mass:g}gpL_mu{mu:g}_sigma{sigma:g}'.format(temp=temp, mass=mass, mu=mu, sigma=sigma)

    nmin = 1e-2
    nmax = 10.0**(mu+4.0*sigma)
    mesh = int(100.0*(numpy.log10(nmax)-numpy.log10(nmin)))
    n = numpy.logspace(numpy.log10(nmin), numpy.log10(nmax), mesh)
    concs = (1.0/n**2)*numpy.exp(-0.5*((numpy.log10(n)-mu)/(sigma))**2)
    tmax = 0.2+0.2*numpy.log10(mass)

    reactor = BatchReactor(nmin=nmin, nmax=nmax, mesh=mesh, grid=grid, concs=concs, temp=temp, volume=volume, mass=mass, monomer=monomer, dens=dens, rand=1.0)
    n = reactor.n
    alpha1m = reactor.alpha1m
    # alpha1m = numpy.ones_like(reactor.alpha1m)

    if not os.path.exists('t_'+basename+'.npy') or not os.path.exists('rho_'+basename+'.npy'):
        t, rho = reactor.solve(tmax, alpha1m=alpha1m, gtol=1e-12, rtol=1e-12, atol=1e-12)
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

    nn, nw, Dn = reactor.postprocess('D_logn', t=t, rho=rho)

    # plot_two_axes(t, nn, Dn, r'$\overline{n}$', 'PDI', 'disp_'+basename+'.png')

    p, dpdn = reactor.postprocess('dpdn', t=t, rho=rho)
    p, dpdlogn = reactor.postprocess('dpdlogn', t=t, rho=rho)
    P, dPdn = reactor.postprocess('dPdn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)
    P, dPdlogn = reactor.postprocess('dPdlogn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)

    # plot_populations(t, n, dpdlogn*numpy.log(10.0), alpha1m, 'Chain Length', r'$d\widetilde{P}/d\log{n}$', 'dpdlogn_'+basename+'.png', prune=prune, xlim=[1.0, nmax])
    # plot_populations(t, n, dpdn, alpha1m, 'Chain Length', r'$d\widetilde{P}/d{n}$', 'dpdn_'+basename+'.png', prune=prune, xscale='linear', xlim=[1.0, 29.0])
    # plot_curves(t, [P], 'Pressure (atm)', 'pressure_'+basename+'.png')

    rho_g, rho_l, rho_s = reactor.postprocess('rho_logn', t=t, rho=rho)
    wg, wl, ws = reactor.postprocess('w_logn', t=t, rho=rho)

    # plot_curves(t, [rho_s, rho_l, rho_g], 'Chain Concentration', 'rho_gls_'+basename+'.png', labels=['Solid (C$_{17'+u'\u2010'+'\infty}$)', 'Liquid (C$_{5'+u'\u2010'+'16}$)', 'Gas (C$_{1'+u'\u2010'+'4}$)'])
    # plot_curves(t, [ws, wl, wg], 'Mass Fraction', 'w_gls_'+basename+'.png', labels=['Solid (C$_{17'+u'\u2010'+'\infty}$)', 'Liquid (C$_{5'+u'\u2010'+'16}$)', 'Gas (C$_{1'+u'\u2010'+'4}$)'])

