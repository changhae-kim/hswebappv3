import itertools
import numpy
import os, sys

from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 16})

from reactor import BatchReactor
from utils import plot_curves, plot_two_axes, plot_populations


temp  = 573.15 # 423.15 573.15
mass  = 1.0/0.3 # 10.0/0.3 1.0/0.3 0.1/0.3 100.0 10.0 1.0
mu    = 3.0
sigma = 0.1

volume  = 1.0
monomer = 14.027
dens    = 940.0

grid = 'logn'

basename = '{temp:g}K_{mass:g}gpL_mu{mu:g}_sigma{sigma:g}'.format(temp=temp, mass=mass, mu=mu, sigma=sigma)

nmin = 1e-2
nmax = 10.0**(mu+4.0*sigma)
mesh = int(100.0*(numpy.log10(nmax)-numpy.log10(nmin)))
n = numpy.logspace(numpy.log10(nmin), numpy.log10(nmax), mesh)
concs = (1.0/n**2)*numpy.exp(-0.5*((numpy.log10(n)-mu)/(sigma))**2)
tmax = 1.0

reactor = BatchReactor(nmin=nmin, nmax=nmax, mesh=mesh, grid=grid, concs=concs, temp=temp, volume=volume, mass=mass, monomer=monomer, dens=dens, rand=1.0)
n = reactor.n
alpha, alpha1m = reactor.get_part()
reactor.W = None
reactor.V = None
reactor.alpha = None
reactor.alpha1m = None
# reactor.W = W = 1.0
# reactor.V = W = 0.0
# reactor.alpha = alpha = numpy.zeros_like(reactor.alpha)
# reactor.alpha1m = alpha1m = numpy.ones_like(reactor.alpha1m)

if not os.path.exists('t_'+basename+'.npy') or not os.path.exists('rho_'+basename+'.npy'):
    t, rho = reactor.solve(tmax, gtol=1e-12, rtol=1e-12, atol=1e-12)
    numpy.save('t_'+basename+'.npy', t)
    numpy.save('rho_'+basename+'.npy', rho)
else:
    t = numpy.load('t_'+basename+'.npy')
    rho = numpy.load('rho_'+basename+'.npy')

# n, dwdn = reactor.postprocess('dwdn', t=t, rho=rho)
n, dwdlogn = reactor.postprocess('dwdlogn', t=t, rho=rho)
# plot_populations(t, n, dwdlogn, alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png')
# plot_populations(t, n, rho, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_'+basename+'.png', xlim=[1.0, nmax], font=16)
plot_populations(t, n, dwdlogn*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_'+basename+'.png', xlim=[1.0, nmax], font=16)
# plot_populations(t, n, rho, alpha1m, '$n$', r'$\tilde{\rho}$', 'rho_n_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0], font=16)
# plot_populations(t, n, dwdn, alpha1m, '$n$', r'$d\widetilde{W}/d{n}$', 'dwdn_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0], font=16)
# xa = n[(n > 1.0) & (alpha1m < 0.5)].max()
# xb = n[(n > 1.0) & (alpha1m > 0.5)].min()
# ya = alpha1m[(n > 1.0) & (alpha1m < 0.5)].max()
# yb = alpha1m[(n > 1.0) & (alpha1m > 0.5)].min()
# print(temp, mass, xa+(xb-xa)/(yb-ya)*(0.5-ya), n[numpy.argmax(dwdn[:, -1])], n[numpy.argmax(dwdlogn[:, -1])])

# nn, nw, Dn = reactor.postprocess('D_logn', t=t, rho=rho)
# plot_two_axes(t, nn, Dn, r'$\widetilde{M}_N$', '$'+u'\u0110'+'$', 'disp_'+basename+'.png', xlim=[0.0, 1.0])
# print(temp, mass, nn[-1], nw[-1], Dn[-1])

# p, dpdn = reactor.postprocess('dpdn', t=t, rho=rho)
# p, dpdlogn = reactor.postprocess('dpdlogn', t=t, rho=rho)
# P, dPdn = reactor.postprocess('dPdn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)
# P, dPdlogn = reactor.postprocess('dPdlogn', t=t, rho=rho, temp=temp, volume=volume, mass=mass, monomer=monomer)
# plot_populations(t, n, dpdlogn*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{P}/d\log{n}$', 'dpdlogn_'+basename+'.png', xlim=[1.0, nmax])
# plot_populations(t, n, dpdn, alpha1m, '$n$', r'$d\widetilde{P}/d{n}$', 'dpdn_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])
# plot_curves([t], [p], r'$\widetilde{P}$', 'p_'+basename+'.png', xlim=[0.0, 1.0])
# plot_curves([t], [P], 'Vapor Pressure (atm)', 'P_'+basename+'.png', xlim=[0.0, 1.0])

# rho_g, rho_l, rho_s = reactor.postprocess('rho_logn', t=t, rho=rho, state_cutoffs=[7.5, 25.5], renorm=True)
# wg, wl, ws = reactor.postprocess('w_logn', t=t, rho=rho, state_cutoffs=[7.5, 25.5], renorm=True)
# labels = ['C$_{26}$$_{+}$', 'C$_{8}$$_{-}$$_{25}$', 'C$_{1}$$_{-}$$_{7}$', ]
# plot_curves([t, t, t], [rho_s, rho_l, rho_g], r'$\widetilde{N}$', 'rho_gls_'+basename+'.png', labels=labels, loc='upper right', xlim=[0.0, 1.0], font=18)
# plot_curves([t, t, t], [ws, wl, wg], r'$\widetilde{W}$', 'w_gls_'+basename+'.png', labels=labels, loc='upper right', xlim=[0.0, 1.0], font=18)
# print(temp, mass, wg[-1], wl[-1], ws[-1])

