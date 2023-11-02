import numpy
import os, sys

from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 16})

from reactor import CSTReactor
from reactor import SemiBatchReactor
from utils import plot_curves, plot_two_axes, plot_populations


temp    = 573.15
mass    = 1.0
volume  = 1.0
mu      = 3.0
sigma   = 0.1
influx  = 1.0
outflux = 1.0

monomer = 14.027
dens    = 940.0

grid = 'logn'

basename = '{temp:g}K_{gpL:g}gpL_mu{mu:g}_sigma{sigma:g}_in{influx:g}_out{outflux:g}'.format(temp=temp, gpL=mass/volume, mu=mu, sigma=sigma, influx=influx, outflux=outflux)

nmin = 1e-2
nmax = 10.0**(mu+4.0*sigma)
mesh = int(100.0*(numpy.log10(nmax)-numpy.log10(nmin)))
n = numpy.logspace(numpy.log10(nmin), numpy.log10(nmax), mesh)
concs = (1.0/n**2)*numpy.exp(-0.5*((numpy.log10(n)-mu)/(sigma))**2)
tmax = 1.0

influx = influx*concs
outflux = [outflux, 0.0]

reactor = SemiBatchReactor(nmin=nmin, nmax=nmax, mesh=mesh, grid=grid, influx=influx, outflux=outflux, concs=concs, temp=temp, volume=volume, mass=mass, monomer=monomer, dens=dens, rand=1.0)
n = reactor.n
#alpha, alpha1m = reactor.get_part()
reactor.W = None
reactor.V = None
reactor.alpha = None
reactor.alpha1m = None

if not os.path.exists('rho_'+basename+'.npy'):
    #rho = reactor.solve(gtom=1e-12)
    t, rho = reactor.solve(tmax, gtol=1e-12, rtol=1e-12, atol=1e-12)
    numpy.save('t_'+basename+'.npy', t)
    numpy.save('rho_'+basename+'.npy', rho)
else:
    t = numpy.load('t_'+basename+'.npy')
    rho = numpy.load('rho_'+basename+'.npy')

alpha, alpha1m = reactor.get_part(rho=rho[:, -1])

g, gin, gout, G, Gin, Gout = reactor.cointegrate(t=t, rho=rho, integrals_only=False)
#rho_reac = ( reactor.rho + ( G + Gin - Gout ).T ).T
#rho_cond = Gout
#rho_comb = ( reactor.rho + ( G + Gin ).T ).T
rho_reac = rho
rho_cond = gout

n, dwdn_reac = reactor.postprocess('dwdn', t=t, rho=rho_reac)
n, dwdlogn_reac = reactor.postprocess('dwdlogn', t=t, rho=rho_reac)
plot_populations(t, n, dwdlogn_reac*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_reac_'+basename+'.png', xlim=[1.0, nmax])
plot_populations(t, n, dwdn_reac, alpha1m, '$n$', r'$d\widetilde{W}/d{n}$', 'dwdn_reac_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])

n, dwdn_cond = reactor.postprocess('dwdn', t=t, rho=rho_cond)
n, dwdlogn_cond = reactor.postprocess('dwdlogn', t=t, rho=rho_cond)
plot_populations(t, n, dwdlogn_cond*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_cond_'+basename+'.png', xlim=[1.0, nmax])
plot_populations(t, n, dwdn_cond, alpha1m, '$n$', r'$d\widetilde{W}/d{n}$', 'dwdn_cond_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])

#n, dwdn_comb = reactor.postprocess('dwdn', t=t, rho=rho_comb)
#n, dwdlogn_comb = reactor.postprocess('dwdlogn', t=t, rho=rho_comb)
#plot_populations(t, n, dwdlogn_comb*numpy.log(10.0), alpha1m, '$n$', r'$d\widetilde{W}/d\log{n}$', 'dwdlogn_comb_'+basename+'.png', xlim=[1.0, nmax])
#plot_populations(t, n, dwdn_comb, alpha1m, '$n$', r'$d\widetilde{W}/d{n}$', 'dwdn_comb_'+basename+'.png', xscale='linear', xlim=[1.0, 29.0])

