import numpy
import sys

from matplotlib import cm, pyplot
pyplot.rcParams.update({'font.size': 14})

sys.path.append('..')
from reactor import BatchReactor


print()
print('Examples of Batch Reactor with Random Scission')
print('These examples are the same as the examples in \'batch.py\',')
print('except that they use random scission instead of chain-end scission.')
print('You can tune the ratio of chain-end scission to random scission by passing different values of rand to the BatchReactor class.')
print('Here, we use rand=1.0 which corresponds to 100\% random scission.')


print()
print('Case 1: Discrete Equation with Phase Partition / Dimensional Input')
print('This example solves the original discrete population balance equations with nonzero phase partition coefficients.')
print('The discrete equations can be invoked by passing grid=\'discrete\' to the BatchReactor class.')
print('This example also demonstrates the use of dimensional input.')
print('We enter the temperature in K, the headspace volume in L, the melt mass in g, and the monomer mass in g/mol.')
print('Then, the code computes the nondimensional Henry\'s constants and the phase partition coefficients.')
print('The concentrations can be entered in arbitrary units.')
print('Then, the code normalizes them to the total number of monomer units.')
print('See lines 30-40 of \'batch_random.py\' script.')

nmax = 110
grid = 'discrete'
partition = 'static'
n = numpy.arange(1, nmax+1, 1)
concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
tmax = 0.1

reactor = BatchReactor(nmax=nmax, grid=grid, partition=partition, concs=concs, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0, rand=1.0)
n1 = reactor.n
a1 = reactor.alpha1m0
t1, y1 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)

prune = 1
nwin = 110
fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twinx()
cmap = cm.viridis(t1/t1.max())
for i, _ in enumerate(t1):
    if i % prune == 0:
        ax1.plot(n1[:nwin], y1[:nwin, i], color=cmap[i])
ax1.set_xlabel('Chain Length')
ax1.set_ylabel('Chain Concentration')
ax1.set_xlim(0, nwin)
ax2.plot(n1, a1, 'k--')
ylim = ax2.get_ylim()
ax2.set_ylabel('Liquid-Phase Partition')
pyplot.tight_layout()
pyplot.savefig('batch_random_partition_discrete.png')
pyplot.close()


print()
print('Case 2: Discrete Equation with No Phase Partition / Nondimensional Input')
print('This example solves the original discrete population balance equations with trivial phase partition coefficients.')
print('The discrete equations can be invoked by passing grid=\'discrete\' to the BatchReactor class.')
print('This example also demonstrates the use of nondimensional input.')
print('We provide the normalized concentrations and the phase partition coefficients.')
print('Then, the code no longer needs the temperature, volume, melt mass, and monomer mass.')
print('See lines 71-82 of \'batch_random.py\' script.')

nmax = 110
grid = 'discrete'
partition = 'static'
n = numpy.arange(1, nmax+1, 1)
concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
rho = concs / numpy.inner(n, concs)
tmax = 0.1

reactor = BatchReactor(nmax=nmax, grid=grid, partition=partition, rho=rho, alpha1m=numpy.ones(nmax), rand=1.0)
n2 = reactor.n
a2 = reactor.alpha1m0
t2, y2 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)

prune = 1
nwin = 110
fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twinx()
cmap = cm.viridis(t2/t2.max())
for i, _ in enumerate(t2):
    if i % prune == 0:
        ax1.plot(n2[:nwin], y2[:nwin, i], color=cmap[i])
ax1.set_xlabel('Chain Length')
ax1.set_ylabel('Chain Concentration')
ax1.set_xlim(0, nwin)
ax2.plot(n2, a2, 'k--')
ax2.set_ylim(ylim)
ax2.set_ylabel('Liquid-Phase Partition')
pyplot.tight_layout()
pyplot.savefig('batch_random_nopartition_discrete.png')
pyplot.close()


print()
print('Case 3: Continuum Approximation with Phase Partition / Mixed Dimensional & Nondimensional Input')
print('This example solves the continuum approximation of the population balance equations with nonzero phase partition coefficients.')
print('The continuum approximation can be invoked by passing grid=\'continuum\' and a nonzero value of mesh to the BatchReactor class.')
print('This example also demonstrates the use of partial nondimensional input.')
print('We provide the normalized concentrations,')
print('but we enter the temperature in K, the headspace volume in L, the melt mass in g, and the monomer mass in g/mol.')
print('Then, the code computes the nondimensional Henry\'s constants and the phase partition coefficients.')
print('See lines 114-129 of \'batch_random.py\' script.')

nmax = 110.0
mesh = 500
grid = 'continuum'
partition = 'static'
n = numpy.linspace(1.0, nmax, mesh)
dn = n[1] - n[0]
concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
w = numpy.ones_like(n)
w[0] = w[-1] = 0.5
rho = ( concs ) / ( numpy.einsum('i,i,i->', w, n, concs) * dn )
tmax = 0.1

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, partition=partition, rho=rho, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0, rand=1.0)
n3 = reactor.n
a3 = reactor.alpha1m0
t3, y3 = reactor.solve(tmax, gtol=1e-9, rtol=1e-9, atol=1e-9)

prune = 2
nwin = 110.0
fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twinx()
cmap = cm.viridis(t3/t3.max())
for i, _ in enumerate(t3):
    if i % prune == 0:
        ax1.plot(n3[n3 <= nwin], y3[n3 <= nwin][:, i], color=cmap[i])
ax1.set_xlabel('Chain Length')
ax1.set_ylabel('Chain Concentration')
ax1.set_xlim(0.0, nwin)
ax2.plot(n3, a3, 'k--')
ylim = ax2.get_ylim()
ax2.set_ylabel('Liquid-Phase Partition')
pyplot.tight_layout()
pyplot.savefig('batch_random_partition_continuum.png')
pyplot.close()


print()
print('Case 4: Continuum Approximation with No Phase Partition / Mixed Dimensional & Nondimensional Input')
print('This example solves the continuum approximation of the population balance equations with trivial phase partition coefficients.')
print('The continuum approximation can be invoked by passing grid=\'continuum\' and a nonzero value of mesh to the BatchReactor class.')
print('This example also demonstrates the use of partial nondimensional input.')
print('We provide the phase partition coefficients,')
print('but we enter the concentrations in arbitrary units.')
print('Then, the code normalizes them to the total number of monomer units.')
print('See lines 161-172 of \'batch_random.py\' script.')

nmax = 110.0
mesh = 500
grid = 'continuum'
partition = 'static'
n = numpy.linspace(1.0, nmax, mesh)
concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
tmax = 0.1

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, partition=partition, concs=concs, alpha1m=numpy.ones(mesh), rand=1.0)
n4 = reactor.n
a4 = reactor.alpha1m0
t4, y4 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)

prune = 1
nwin = 110.0
fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twinx()
cmap = cm.viridis(t4/t4.max())
for i, _ in enumerate(t4):
    if i % prune == 0:
        ax1.plot(n4[n4 <= nwin], y4[n4 <= nwin][:, i], color=cmap[i])
ax1.set_xlabel('Chain Length')
ax1.set_ylabel('Chain Concentration')
ax1.set_xlim(0.0, nwin)
ax2.plot(n4, a4, 'k--')
ax2.set_ylim(ylim)
ax2.set_ylabel('Liquid-Phase Partition')
pyplot.tight_layout()
pyplot.savefig('batch_random_nopartition_continuum.png')
pyplot.close()


print()
print('Case 5: Log Scale Equation with Phase Partition / Dimensional Input')
print('This example solves the population balance equations in a logarithmic scale with nonzero phase partition coefficients.')
print('The logarithmic equations can be invoked by passing grid=\'log_n\' to the BatchReactor class.')
print('This example also demonstrates the use of dimensional input.')
print('We enter the temperature in K, the headspace volume in L, the melt mass in g, and the monomer mass in g/mol.')
print('Then, the code computes the nondimensional Henry\'s constants and the phase partition coefficients.')
print('The concentrations can be entered in arbitrary units.')
print('Then, the code normalizes them to the total number of monomer units.')
print('For reference, we also solve the discrete equations.')
print('See lines 206-230 of \'batch_random.py\' script.')

nmax = 10.0**2.10
mesh = 500
grid = 'log_n'
partition = 'static'
n = numpy.logspace(0.0, numpy.log10(nmax), mesh)
concs = numpy.exp(-0.5*((numpy.log(n)-numpy.log(100.0))/(0.01*numpy.log(10.0)))**2)
tmax = 0.1

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, partition=partition, concs=concs, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0, rand=1.0)
n5 = reactor.n
a5 = reactor.alpha1m0
t5, y5 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)

nmax = int(10.0**2.10)+1
mesh = 0
grid = 'discrete'
partition = 'static'
n = numpy.arange(1, nmax+1, 1)
concs = (1.0/n) * numpy.exp(-0.5*((numpy.log(n)-numpy.log(100.0))/(0.01*numpy.log(10.0)))**2)
tmax = 0.1

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, partition=partition, concs=concs, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0, rand=1.0)
n6 = reactor.n
a6 = reactor.alpha1m0
t6, y6 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)

prune = 1
nwin = 10.0**2.10
fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twinx()
cmap = cm.viridis(t5/t5.max())
for i, _ in enumerate(t5):
    if i % prune == 0:
        ax1.plot(n5[n5 <= nwin], y5[n5 <= nwin][:, i], color=cmap[i])
ax1.set_xlabel('Chain Length')
ax1.set_ylabel('Chain Concentration')
ax1.set_xlim(1.0, nwin)
ax1.set_xscale('log')
ax2.plot(n5, a5, 'k--')
ax2.set_xscale('log')
ylim = ax2.get_ylim()
ax2.set_ylabel('Liquid-Phase Partition')
pyplot.tight_layout()
pyplot.savefig('batch_random_partition_log_n.png')
pyplot.close()

prune = 1
nwin = 10.0**2.10
fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twinx()
cmap = cm.viridis(t6/t6.max())
for i, _ in enumerate(t6):
    if i % prune == 0:
        ax1.plot(n6[n6 <= nwin], y6[n6 <= nwin][:, i], color=cmap[i])
ax1.set_xlabel('Chain Length')
ax1.set_ylabel('Chain Concentration')
ax1.set_xlim(1.0, nwin)
ax1.set_xscale('log')
ax2.plot(n6, a6, 'k--')
ax2.set_xscale('log')
ylim = ax2.get_ylim()
ax2.set_ylabel('Liquid-Phase Partition')
pyplot.tight_layout()
pyplot.savefig('batch_random_partition_log_n_compare.png')
pyplot.close()


print()
print('Case 6: Log Scale Equation with No Phase Partition / Mixed Dimensional & Nondimensional Input')
print('This example solves the population balance equations in a logarithmic scale with trivial phase partition coefficients.')
print('The logarithmic equations can be invoked by passing grid=\'log_n\' to the BatchReactor class.')
print('This example also demonstrates the use of partial nondimensional input.')
print('We provide the phase partition coefficients,')
print('but we enter the concentrations in arbitrary units.')
print('Then, the code normalizes them to the total number of monomer units.')
print('For reference, we also solve the discrete equations.')
print('See lines 286-310 of \'batch_random.py\' script.')

nmax = 10.0**2.10
mesh = 500
grid = 'log_n'
partition = 'static'
n = numpy.logspace(0.0, numpy.log10(nmax), mesh)
concs = numpy.exp(-0.5*((numpy.log(n)-numpy.log(100.0))/(0.01*numpy.log(10.0)))**2)
tmax = 0.1

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, partition=partition, concs=concs, alpha1m=numpy.ones(mesh), rand=1.0)
n7 = reactor.n
a7 = reactor.alpha1m0
t7, y7 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)

nmax = int(10.0**2.10)+1
mesh = 0
grid = 'discrete'
partition = 'static'
n = numpy.arange(1, nmax+1, 1)
concs = (1.0/n) * numpy.exp(-0.5*((numpy.log(n)-numpy.log(100.0))/(0.01*numpy.log(10.0)))**2)
tmax = 0.1

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, partition=partition, concs=concs, alpha1m=numpy.ones(nmax), rand=1.0)
n8 = reactor.n
a8 = reactor.alpha1m0
t8, y8 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)

prune = 1
nwin = 10.0**2.10
fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twinx()
cmap = cm.viridis(t7/t7.max())
for i, _ in enumerate(t7):
    if i % prune == 0:
        ax1.plot(n7[n7 <= nwin], y7[n7 <= nwin][:, i], color=cmap[i])
ax1.set_xlabel('Chain Length')
ax1.set_ylabel('Chain Concentration')
ax1.set_xlim(1.0, nwin)
ax1.set_xscale('log')
ax2.plot(n7, a7, 'k--')
ax2.set_xscale('log')
ax2.set_ylim(ylim)
ax2.set_ylabel('Liquid-Phase Partition')
pyplot.tight_layout()
pyplot.savefig('batch_random_nopartition_log_n.png')
pyplot.close()

prune = 1
nwin = 10.0**2.10
fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twinx()
cmap = cm.viridis(t8/t8.max())
for i, _ in enumerate(t8):
    if i % prune == 0:
        ax1.plot(n8[n8 <= nwin], y8[n8 <= nwin][:, i], color=cmap[i])
ax1.set_xlabel('Chain Length')
ax1.set_ylabel('Chain Concentration')
ax1.set_xlim(1.0, nwin)
ax1.set_xscale('log')
ax2.plot(n8, a8, 'k--')
ax2.set_xscale('log')
ax2.set_ylim(ylim)
ax2.set_ylabel('Liquid-Phase Partition')
pyplot.tight_layout()
pyplot.savefig('batch_random_nopartition_log_n_compare.png')
pyplot.close()


print()

