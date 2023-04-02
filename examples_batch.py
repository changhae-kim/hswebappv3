import numpy

from matplotlib import cm, pyplot
pyplot.rcParams.update({'font.size': 14})

from reactor_batch import BatchReactor


print()
print('Case 1: Discrete Equation with Phase Partition / Dimensional Input')
print('This example solves the original discrete population balance equations with nonzero phase partition coefficients.')
print('The discrete equations can be invoked by passing grid=0 to the BatchReactor class.')
print('This example also demonstrates the use of dimensional input.')
print('We enter the temperature in K, the headspace volume in L, the melt mass in g, and the monomer mass in g/mol.')
print('Then, the code computes the nondimensional Henry\'s constants and the phase partition coefficients.')
print('The concentrations can be entered in arbitrary units.')
print('Then, the code normalizes them to the total number of monomer units.')
print('See lines 20-29 of \'examples_batch.py\' script.')

nmax = 100
grid = 0
concs = numpy.zeros(nmax)
concs[-1] = 1.0
tmax = 100.0

reactor = BatchReactor(nmax=nmax, grid=grid, concs=concs, temp=573.15, volume=1.0, mass=10.0, monomer=14.027)
t1, y1 = reactor.solve(tmax, rtol=1e-12, atol=1e-12)
n1 = numpy.copy(reactor.n)
a1 = numpy.copy(reactor.alpha1m)


print()
print('Case 2: Discrete Equation with No Phase Partition / Nondimensional Input')
print('This example solves the original discrete population balance equations with trivial phase partition coefficients.')
print('The discrete equations can be invoked by passing grid=0 to the BatchReactor class.')
print('This example also demonstrates the use of nondimensional input.')
print('We provide the normalized concentrations and the phase partition coefficients.')
print('Then, the code no longer needs the temperature, volume, melt mass, and monomer mass.')
print('See lines 41-50 of \'examples_batch.py\' script.')

nmax = 100
grid = 0
rho = numpy.zeros(nmax)
rho[-1] = 0.01
tmax = 100.0

reactor = BatchReactor(nmax=nmax, grid=grid, rho=rho, alpha1m=numpy.ones(nmax))
t2, y2 = reactor.solve(tmax, rtol=1e-12, atol=1e-12)
n2 = numpy.copy(reactor.n)
a2 = numpy.copy(reactor.alpha1m)


print()
print('Case 3: Continuum Approximation with Phase Partition / Mixed Dimensional & Nondimensional Input')
print('This example solves the population balance equations under the continuum approximation on n with nonzero phase partition coefficients.')
print('The continuum approximation can be invoked by passing a nonzero value of grid to the BatchReactor class.')
print('This example also demonstrates the use of partial nondimensional input.')
print('We provide the normalized concentrations,')
print('but we enter the temperature in K, the headspace volume in L, the melt mass in g, and the monomer mass in g/mol.')
print('Then, the code computes the nondimensional Henry\'s constants and the phase partition coefficients.')
print('See lines 63-76 of \'examples_batch.py\' script.')

nmax = 105.0
grid = 521
n = numpy.linspace(1.0, nmax, grid)
dn = n[1] - n[0]
concs = numpy.exp(-0.5*((n-100.0)/(0.1))**2)
w = numpy.ones_like(n)
w[0] = w[-1] = 0.5
rho = concs / numpy.einsum('i,i,i->', w, n, concs) * dn
tmax = 100.0

reactor = BatchReactor(nmax=nmax, grid=grid, rho=rho, temp=573.15, volume=1.0, mass=10.0, monomer=14.027)
t3, y3 = reactor.solve(tmax, rtol=1e-12, atol=1e-12)
n3 = numpy.copy(reactor.n)
a3 = numpy.copy(reactor.alpha1m)


print()
print('Case 4: Continuum Approximation with No Phase Partition / Mixed Dimensional & Nondimensional Input')
print('This example solves the population balance equations under the continuum approximation on n with trivial phase partition coefficients.')
print('The continuum approximation can be invoked by passing a nonzero value of grid to the BatchReactor class.')
print('This example also demonstrates the use of partial nondimensional input.')
print('We provide the phase partition coefficients,')
print('but we enter the concentrations in arbitrary units.')
print('Then, the code normalizes them to the total number of monomer units.')
print('See lines 89-99 of \'examples_batch.py\' script.')

nmax = 105.0
grid = 521
n = numpy.linspace(1.0, nmax, grid)
dn = n[1] - n[0]
concs = numpy.exp(-0.5*((n-100.0)/(0.1))**2)
tmax = 100.0

reactor = BatchReactor(nmax=nmax, grid=grid, concs=concs, alpha1m=numpy.ones(grid))
t4, y4 = reactor.solve(tmax, rtol=1e-12, atol=1e-12)
n4 = numpy.copy(reactor.n)
a4 = numpy.copy(reactor.alpha1m)


print()

