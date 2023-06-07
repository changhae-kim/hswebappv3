import itertools
import numpy
import os, sys

from matplotlib import pyplot

sys.path.append('..')
from reactor import CSTReactor
from utils import plot_populations


volume  = 1.0
monomer = 14.027
dens    = 920.0

grid = 'logn'

temps  = [573.15] # [ 423.15, 473.15, 523.15, 573.15 ]
masses = [10.0] # [ 1.0, 10.0, 100.0, 1000.0 ]
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
    '''
    logn = numpy.log(n)
    dlogn = logn[1] - logn[0]

    g0 = n**1 * Gout.T
    g1 = n**2 * Gout.T
    g2 = n**3 * Gout.T

    G0 = numpy.zeros_like(t)
    G1 = numpy.zeros_like(t)
    G2 = numpy.zeros_like(t)

    G0 = 0.5 * numpy.einsum('ij->i', g0[:, 1:] + g0[:, :-1]) * dlogn
    G1 = 0.5 * numpy.einsum('ij->i', g1[:, 1:] + g1[:, :-1]) * dlogn
    G2 = 0.5 * numpy.einsum('ij->i', g2[:, 1:] + g2[:, :-1]) * dlogn

    nn = G1/G0
    nw = G2/G1
    ds = nw/nn

    fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax1.plot(t, nn, color=color)
    # ax1.set_yscale('log')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(r'$\overline{n}$', color=color)
    ax1.tick_params(axis='y', which='both', labelcolor=color)
    color = 'tab:orange'
    ax2.plot(t, ds, color=color)
    ax2.set_ylabel('PDI', color=color)
    ax2.tick_params(axis='y', which='both', labelcolor=color)
    fig.tight_layout()
    fig.savefig('disp_'+basename+'.png')
    pyplot.close()
    
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
