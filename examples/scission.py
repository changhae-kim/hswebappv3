import numpy
import sys

from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 14})

sys.path.append('..')
from reactor import BatchReactor


nc = 10.0
t1max = 140.0
t2max = 4.0
'''
nc = 25.0
t1max = 100.0
t2max = 0.35
'''

nmax = 110.0
mesh = 500
grid = 'continuum'
n = numpy.linspace(1.0, nmax, mesh)
dn = n[1] - n[0]
concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
w = numpy.ones_like(n)
w[0] = w[-1] = 0.5
rho = ( concs ) / ( numpy.einsum('i,i,i->', w, n, concs) * dn )
tmax = t1max

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, rho=rho, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0)
n1 = reactor.n
a1 = reactor.alpha1m
t1, y1 = reactor.solve(tmax, gtol=1e-9, rtol=1e-9, atol=1e-9)

nmax = 110.0
mesh = 500
grid = 'continuum'
n = numpy.linspace(1.0, nmax, mesh)
dn = n[1] - n[0]
concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
w = numpy.ones_like(n)
w[0] = w[-1] = 0.5
rho = ( concs ) / ( numpy.einsum('i,i,i->', w, n, concs) * dn )
tmax = t2max

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, rho=rho, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0, rand=1.0)
n2 = reactor.n
a2 = reactor.alpha1m
t2, y2 = reactor.solve(tmax, gtol=1e-9, rtol=1e-9, atol=1e-9)

ia = numpy.argmax( ( n1 < nc ) * n1 )
ib = ia + 1
y1c = y1[ia] + ( y1[ib] - y1[ia] ) * ( nc - n1[ia] ) / ( n1[ib] - n1[ia] )
y2c = y2[ia] + ( y2[ib] - y2[ia] ) * ( nc - n2[ia] ) / ( n2[ib] - n2[ia] )

fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twiny()
color = 'tab:blue'
ax1.plot(t1, y1c, color=color, label='Chain-End')
ax1.set_xlabel('Chain-End Scission Time', color=color)
ax1.tick_params(axis='x', labelcolor=color)
color = 'tab:orange'
ax2.plot(t2, y2c, color=color, label='Random')
ax2.set_xlabel('Random Scission Time', color=color)
ax2.tick_params(axis='x', labelcolor=color)
ax1.set_ylabel('C$_{' + '{:.0f}'.format(nc) + '}$ Chain Concentration')
pyplot.tight_layout()
pyplot.savefig('scission_continuum_c{:.0f}.png'.format(nc))
pyplot.close()


nmax = 10.0**2.10
mesh = 500
grid = 'log_n'
n = numpy.logspace(0.0, numpy.log10(nmax), mesh)
concs = numpy.exp(-0.5*((numpy.log(n)-numpy.log(100.0))/(0.01*numpy.log(10.0)))**2)
tmax = t1max

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, concs=concs, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0)
n3 = reactor.n
a3 = reactor.alpha1m
t3, y3 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)

nmax = 10.0**2.10
mesh = 500
grid = 'log_n'
n = numpy.logspace(0.0, numpy.log10(nmax), mesh)
concs = numpy.exp(-0.5*((numpy.log(n)-numpy.log(100.0))/(0.01*numpy.log(10.0)))**2)
tmax = t2max

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, concs=concs, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0, rand=1.0)
n4 = reactor.n
a4 = reactor.alpha1m
t4, y4 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)

ia = numpy.argmax( ( n3 < nc ) * n3 )
ib = ia + 1
y3c = y3[ia] + ( y3[ib] - y3[ia] ) * ( nc - n3[ia] ) / ( n3[ib] - n3[ia] )
y4c = y4[ia] + ( y4[ib] - y4[ia] ) * ( nc - n4[ia] ) / ( n4[ib] - n4[ia] )

fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twiny()
color = 'tab:blue'
ax1.plot(t3, y3c, color=color, label='Chain-End')
ax1.set_xlabel('Chain-End Scission Time', color=color)
ax1.tick_params(axis='x', labelcolor=color)
color = 'tab:orange'
ax2.plot(t4, y4c, color=color, label='Random')
ax2.set_xlabel('Random Scission Time', color=color)
ax2.tick_params(axis='x', labelcolor=color)
ax1.set_ylabel('C$_{' + '{:.0f}'.format(nc) + '}$ Chain Concentration')
pyplot.tight_layout()
pyplot.savefig('scission_log_n_c{:.0f}.png'.format(nc))
pyplot.close()

