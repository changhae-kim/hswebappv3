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

temps  = [ 573.15 ] # [ 423.15, 573.15 ]
masses = [ 100.0 ] # # [ 3.0, 30.0 ] + [ 1.0, 10.0, 100.0, 1000.0 ]
masses = [ (mass)/(1.0+mass/dens) for mass in masses ]
mus    = [ 3.0 ]
sigmas = [ 0.1 ]
fluxes = [ 1.0, 10.0 ] # [ 30.0 ] + [ 1.0, 10.0, 100.0 ]

# tt = []
# PP = []
# Labels = []

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

    reactor = CSTReactor(nmin=nmin, nmax=nmax, mesh=mesh, grid=grid, influx=influx, outflux=outflux, concs=concs, temp=temp, volume=volume, mass=mass, monomer=monomer, dens=dens, rand=1.0)
    n = reactor.n
    alpha = reactor.alpha
    alpha1m = reactor.alpha1m
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

    # n, dwdn = reactor.postprocess('dwdn', rho=rho)
    # n, dwdlogn = reactor.postprocess('dwdlogn', rho=rho)
    # plot_populations(t, n, dwdlogn, alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png')
    # plot_populations(t, n, rho, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_'+basename+'.png', xlim=[1.0, nmax])
    # plot_populations(t, n, dwdlogn*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png', xlim=[1.0, nmax])
    # plot_populations(t, n, rho, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])
    # plot_populations(t, n, dwdn, alpha1m, '$n$', r'$d\widetilde{W}/d{n}$', 'dwdn_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])

    # p, dpdn = reactor.postprocess('dpdn', t=t, rho=rho)
    # p, dpdlogn = reactor.postprocess('dpdlogn', t=t, rho=rho)
    # P, dPdn = reactor.postprocess('dPdn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)
    # P, dPdlogn = reactor.postprocess('dPdlogn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)
    # plot_populations(t, n, dpdlogn*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{P}/d\log{n}$', 'dpdlogn_'+basename+'.png', xlim=[1.0, nmax])
    # plot_populations(t, n, dpdn, alpha1m, '$n$', r'$d\widetilde{P}/d{n}$', 'dpdn_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])
    # plot_curves(t, [p], r'$\widetilde{P}$', 'p_'+basename+'.png')
    # plot_curves(t, [P], 'Vapor Pressure (atm)', 'P_'+basename+'.png')

    # n, dwdn1 = reactor.postprocess('dwdn', t=t, rho=rho1)
    # n, dwdlogn1 = reactor.postprocess('dwdlogn', t=t, rho=rho1)
    # plot_populations(t, n, rho1, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_'+basename+'.png', xlim=[1.0, nmax])
    # plot_populations(t, n, dwdlogn1*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png', xlim=[1.0, nmax])
    # plot_populations(t, n, rho1, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])
    # plot_populations(t, n, dwdn1, alpha1m, '$n$', r'$d\widetilde{W}/d{n}$', 'dwdn_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])

    # n, dwdn2 = reactor.postprocess('dwdn', t=t, rho=rho2)
    # n, dwdlogn2 = reactor.postprocess('dwdlogn', t=t, rho=rho2)
    # plot_populations(t, n, rho2, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_'+basename+'.png', xlim=[1.0, nmax])
    # plot_populations(t, n, dwdlogn2*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png', xlim=[1.0, nmax])
    # plot_populations(t, n, rho2, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])
    # plot_populations(t, n, dwdn2, alpha1m, '$n$', r'$d\widetilde{W}/d{n}$', 'dwdn_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])

    # nn, nw, Dn = reactor.postprocess('D_logn', t=t, rho=Gout)
    # plot_two_axes(t, nn, Dn, r'$\overline{n}$', '$'+u'\u0110'+'$', 'disp_'+basename+'.png')

    # n, dwdn3 = reactor.postprocess('dwdn', t=t, rho=rho3)
    # n, dwdlogn3 = reactor.postprocess('dwdlogn', t=t, rho=rho3)
    # plot_populations(t, n, rho3, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_'+basename+'.png', xlim=[1.0, nmax])
    # plot_populations(t, n, dwdlogn3*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png', xlim=[1.0, nmax])
    # plot_populations(t, n, rho3, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])
    # plot_populations(t, n, dwdn3, alpha1m, '$n$', r'$d\widetilde{W}/d{n}$', 'dwdn_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])

    # rho_g, rho_l, rho_s = reactor.postprocess('rho_logn', t=t, rho=rho3)
    # wg, wl, ws = reactor.postprocess('w_logn', t=t, rho=rho3)
    # labels = ['Solid (C$_{17'+u'\u2010'+'\infty}$)', 'Liquid (C$_{5'+u'\u2010'+'16}$)', 'Gas (C$_{1'+u'\u2010'+'4}$)']
    # plot_curves([t, t, t], [rho_s, rho_l, rho_g], r'$\widetilde{N}$', 'rho_gls_'+basename+'.png', labels=labels, loc='upper right')
    # plot_curves([t, t, t], [ws, wl, wg], r'$\widetilde{W}$', 'w_gls_'+basename+'.png', labels=labels, loc='upper right')

    # tt.append(t)
    # PP.append(P)
    # Labels.append(r'$\tilde{f}_\mathrm{out} = '+'{:g}'.format(flux)+'$')

# plot_curves(tt, PP, 'Vapor Pressure (atm)', 'P_'+basename+'.png', labels=Labels, loc='lower left', yscale='log', xlim=[0.0, 1.0], ylim=[10.0**(-0.30), 10.0**(+4.30)], size=(6.4, 4.8))

