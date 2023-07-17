import itertools
import numpy
import os, sys

from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 14})

from reactor import BatchReactor
from utils import plot_curves, plot_two_axes, plot_populations


volume  = 1.0
monomer = 14.027
dens    = 940.0

grid = 'logn'

temps  = [ 423.15, 573.15 ]
masses = [ 100.0, 10.0, 1.0 ]
mus    = [ 3.0 ]
sigmas = [ 0.1 ]

for temp, mass, mu, sigma in itertools.product(temps, masses, mus, sigmas):

    basename = '{temp:g}K_{mass:g}gpL_mu{mu:g}_sigma{sigma:g}'.format(temp=temp, mass=mass, mu=mu, sigma=sigma)

    nmin = 1e-2 # 1e-3
    nmax = 10.0**(mu+4.0*sigma)
    mesh = int(100.0*(numpy.log10(nmax)-numpy.log10(nmin)))
    n = numpy.logspace(numpy.log10(nmin), numpy.log10(nmax), mesh)
    concs = (1.0/n**2)*numpy.exp(-0.5*((numpy.log10(n)-mu)/(sigma))**2)
    tmax = 1.0

    reactor = BatchReactor(nmin=nmin, nmax=nmax, mesh=mesh, grid=grid, concs=concs, temp=temp, volume=volume, mass=mass, monomer=monomer, dens=dens, rand=1.0)
    n = reactor.n
    alpha, alpha1m = reactor.get_part()
    reactor.alpha = None
    reactor.alpha1m = None
    # reactor.alpha = alpha = numpy.zeros_like(reactor.alpha)
    # reactor.alpha1m = alpha1m = numpy.ones_like(reactor.alpha1m)

    if not os.path.exists('t_'+basename+'.npy') or not os.path.exists('rho_'+basename+'.npy'):
        t, rho = reactor.solve(tmax, gtol=1e-12, rtol=1e-12, atol=1e-12)
        numpy.save('t_'+basename+'.npy', t)
        numpy.save('rho_'+basename+'.npy', rho)
    else:
        t = numpy.load('t_'+basename+'.npy')
        rho = numpy.load('rho_'+basename+'.npy')

    n, dwdn = reactor.postprocess('dwdn', t=t, rho=rho)
    n, dwdlogn = reactor.postprocess('dwdlogn', t=t, rho=rho)
    # plot_populations(t, n, dwdlogn, alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png')
    # plot_populations(t, n, rho, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_'+basename+'.png', xlim=[1.0, nmax], font=16)
    # plot_populations(t, n, dwdlogn*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png', xlim=[1.0, nmax], font=16)
    # plot_populations(t, n, rho, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0], font=18)
    # plot_populations(t, n, dwdn, alpha1m, '$n$', r'$d\widetilde{W}/d{n}$', 'dwdn_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0], font=18)
    # xa = n[(n > 1.0) & (alpha1m < 0.5)].max()
    # xb = n[(n > 1.0) & (alpha1m > 0.5)].min()
    # ya = alpha1m[(n > 1.0) & (alpha1m < 0.5)].max()
    # yb = alpha1m[(n > 1.0) & (alpha1m > 0.5)].min()
    # print(temp, mass, xa+(xb-xa)/(yb-ya)*(0.5-ya), n[numpy.argmax(dwdn[:, -1])], n[numpy.argmax(dwdlogn[:, -1])])

    nn, nw, Dn = reactor.postprocess('D_logn', t=t, rho=rho)
    # plot_two_axes(t, nn, Dn, r'$\overline{n}$', '$'+u'\u0110'+'$', 'disp_'+basename+'.png', xlim=[0.0, 1.0])
    # print(temp, mass, nn[-1], nw[-1], Dn[-1])

    p, dpdn = reactor.postprocess('dpdn', t=t, rho=rho)
    p, dpdlogn = reactor.postprocess('dpdlogn', t=t, rho=rho)
    P, dPdn = reactor.postprocess('dPdn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)
    P, dPdlogn = reactor.postprocess('dPdlogn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)
    # plot_populations(t, n, dpdlogn*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{P}/d\log{n}$', 'dpdlogn_'+basename+'.png', xlim=[1.0, nmax])
    # plot_populations(t, n, dpdn, alpha1m, '$n$', r'$d\widetilde{P}/d{n}$', 'dpdn_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])
    # plot_curves([t], [p], r'$\widetilde{P}$', 'p_'+basename+'.png', xlim=[0.0, 1.0])
    # plot_curves([t], [P], 'Vapor Pressure (atm)', 'P_'+basename+'.png', xlim=[0.0, 1.0])

    rho_g, rho_l, rho_s = reactor.postprocess('rho_logn', t=t, rho=rho)
    wg, wl, ws = reactor.postprocess('w_logn', t=t, rho=rho)
    labels = ['Solid', 'Liquid', 'Gas']
    # plot_curves([t, t, t], [rho_s, rho_l, rho_g], r'$\widetilde{N}$', 'rho_gls_'+basename+'.png', labels=labels, loc='upper right', xlim=[0.0, 1.0], font=16)
    # plot_curves([t, t, t], [ws, wl, wg], r'$\widetilde{W}$', 'w_gls_'+basename+'.png', labels=labels, loc='upper right', xlim=[0.0, 1.0], font=16)
    # print(temp, mass, wg[-1], wl[-1], ws[-1])


for temp in temps:

    tt = []
    PP = []
    llabels = []

    for mass, mu, sigma in itertools.product(masses, mus, sigmas):

        nmin = 1e-2 # 1e-3
        nmax = 10.0**(mu+4.0*sigma)
        mesh = int(100.0*(numpy.log10(nmax)-numpy.log10(nmin)))
        n = numpy.logspace(numpy.log10(nmin), numpy.log10(nmax), mesh)
        concs = (1.0/n**2)*numpy.exp(-0.5*((numpy.log10(n)-mu)/(sigma))**2)
        tmax = 1.0

        reactor = BatchReactor(nmin=nmin, nmax=nmax, mesh=mesh, grid=grid, concs=concs, temp=temp, volume=volume, mass=mass, monomer=monomer, dens=dens, rand=1.0)
        n = reactor.n
        alpha, alpha1m = reactor.get_part()
        reactor.alpha = None
        reactor.alpha1m = None
        # reactor.alpha = alpha = numpy.zeros_like(reactor.alpha)
        # reactor.alpha1m = alpha1m = numpy.ones_like(reactor.alpha1m)

        if not os.path.exists('t_'+basename+'.npy') or not os.path.exists('rho_'+basename+'.npy'):
            t, rho = reactor.solve(tmax, gtol=1e-12, rtol=1e-12, atol=1e-12)
            numpy.save('t_'+basename+'.npy', t)
            numpy.save('rho_'+basename+'.npy', rho)
        else:
            t = numpy.load('t_'+basename+'.npy')
            rho = numpy.load('rho_'+basename+'.npy')

        P, dPdn = reactor.postprocess('dPdn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)
        P, dPdlogn = reactor.postprocess('dPdlogn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)

        tt.append(t)
        PP.append(P)
        llabels.append('$V_0 / W_0 =$'+'{:g}'.format(1000.0/mass)+' cm$^3$$\cdot$g$^{-1}$')

    # plot_curves(tt, PP, 'Vapor Pressure (atm)', 'P_'+basename+'.png', labels=llabels, loc='lower right', yscale='log', xlim=[0.0, 1.0], ylim=[10.0**(-0.20), 10.0**(+4.20)], font=18, size=(6.4, 4.8))

