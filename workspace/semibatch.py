import itertools
import numpy
import os, sys

from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 16})

from reactor import SemiBatchReactor
from utils import plot_curves, plot_two_axes, plot_populations


temp  = 573.15 # 423.15 573.15
mass  = 1.0/0.3 # 10.0/0.3 1.0/0.3 0.1/0.3 100.0 10.0 1.0
mu    = 3.0
sigma = 0.1
flux  = 100.0 # 1.0 10.0 100.0

volume  = 1.0
monomer = 14.027
dens    = 940.0

grid = 'logn'

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
rho_reac = ( reactor.rho + ( G + Gin - Gout ).T ).T
rho_cond = Gout
rho_comb = ( reactor.rho + ( G + Gin ).T ).T

# n, dwdn = reactor.postprocess('dwdn', t=t, rho=rho)
# n, dwdlogn = reactor.postprocess('dwdlogn', t=t, rho=rho)
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

# n, dwdn_reac = reactor.postprocess('dwdn', t=t, rho=rho_reac)
# n, dwdlogn_reac = reactor.postprocess('dwdlogn', t=t, rho=rho_reac)
# plot_populations(t, n, rho_reac, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_reac_'+basename+'.png', xlim=[1.0, nmax])
# plot_populations(t, n, dwdlogn_reac*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_reac_'+basename+'.png', xlim=[1.0, nmax])
# plot_populations(t, n, rho_reac, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_reac_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0], font=16)
# plot_populations(t, n, dwdn_reac, alpha1m, '$n$', r'$d\widetilde{W}/d{n}$', 'dwdn_reac_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0], font=16)

# n, dwdn_cond = reactor.postprocess('dwdn', t=t, rho=rho_cond)
# n, dwdlogn_cond = reactor.postprocess('dwdlogn', t=t, rho=rho_cond)
# plot_populations(t, n, rho_cond, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_cond_'+basename+'.png', xlim=[1.0, nmax])
# plot_populations(t, n, dwdlogn_cond*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_cond_'+basename+'.png', xlim=[1.0, nmax])
# plot_populations(t, n, rho_cond, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_cond_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0], font=16)
# plot_populations(t, n, dwdn_cond, alpha1m, '$n$', r'$d\widetilde{W}/d{n}$', 'dwdn_cond_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0], font=16)

# n, dwdn_comb = reactor.postprocess('dwdn', t=t, rho=rho_comb)
# n, dwdlogn_comb = reactor.postprocess('dwdlogn', t=t, rho=rho_comb)
# plot_populations(t, n, rho_comb, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_comb_'+basename+'.png', xlim=[1.0, nmax])
# plot_populations(t, n, dwdlogn_comb*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_comb_'+basename+'.png', xlim=[1.0, nmax])
# plot_populations(t, n, rho_comb, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_comb_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0], font=16)
# plot_populations(t, n, dwdn_comb, alpha1m, '$n$', r'$d\widetilde{W}/d{n}$', 'dwdn_comb_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0], font=16)
# xa = n[(n > 1.0) & (alpha1m < 0.5)].max()
# xb = n[(n > 1.0) & (alpha1m > 0.5)].min()
# ya = alpha1m[(n > 1.0) & (alpha1m < 0.5)].max()
# yb = alpha1m[(n > 1.0) & (alpha1m > 0.5)].min()
# print(flux, xa+(xb-xa)/(yb-ya)*(0.5-ya), n[numpy.argmax(dwdn_comb[:, -1])], n[numpy.argmax(dwdlogn_comb[:, -1])])

# nn, nw, Dn = reactor.postprocess('D_logn', t=t, rho=rho_comb)
# plot_two_axes(t, nn, Dn, r'$\widetilde{M}_N$', '$'+u'\u0110'+'$', 'disp_'+basename+'.png')
# print(mass, flux, nn[-1], nw[-1], Dn[-1])

# rho_g, rho_l, rho_s = reactor.postprocess('rho_logn', t=t, rho=rho_comb, state_cutoffs=[7.5, 25.5], renorm=True)
# wg, wl, ws = reactor.postprocess('w_logn', t=t, rho=rho_comb, state_cutoffs=[7.5, 25.5], renorm=True)
# labels = ['C$_{26}$$_{+}$', 'C$_{8}$$_{-}$$_{25}$', 'C$_{1}$$_{-}$$_{7}$', ]
# plot_curves([t, t, t], [rho_s, rho_l, rho_g], r'$\widetilde{N}$', 'rho_gls_'+basename+'.png', labels=labels, loc='upper right', xlim=[0.0, 1.0], font=18)
# plot_curves([t, t, t], [ws, wl, wg], r'$\widetilde{W}$', 'w_gls_'+basename+'.png', labels=labels, loc='upper right', xlim=[0.0, 1.0], font=18)
# print(mass, flux, wg[-1], wl[-1], ws[-1])

