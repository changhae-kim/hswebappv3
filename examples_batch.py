import numpy

from matplotlib import cm, pyplot
pyplot.rcParams.update({'font.size': 14})

from reactor import BatchReactor


print()
print('Case 1: Discrete Equation with Phase Partition / Dimensional Input')
print('This example solves the original discrete population balance equations with nonzero phase partition coefficients.')
print('The discrete equations can be invoked by passing grid=\'discrete\' to the BatchReactor class.')
print('This example also demonstrates the use of dimensional input.')
print('We enter the temperature in K, the headspace volume in L, the melt mass in g, and the monomer mass in g/mol.')
print('Then, the code computes the nondimensional Henry\'s constants and the phase partition coefficients.')
print('The concentrations can be entered in arbitrary units.')
print('Then, the code normalizes them to the total number of monomer units.')
print('See lines 20-29 of \'examples_batch.py\' script.')

nmax = 110
grid = 'discrete'
n = numpy.arange(1, nmax+1, 1)
concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
tmax = 100.0

reactor = BatchReactor(nmax=nmax, grid=grid, concs=concs, temp=573.15, volume=1.0, mass=10.0, monomer=14.027)
t1, y1 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
n1 = numpy.copy(reactor.n)
a1 = numpy.copy(reactor.alpha1m0)


print()
print('Case 2: Discrete Equation with No Phase Partition / Nondimensional Input')
print('This example solves the original discrete population balance equations with trivial phase partition coefficients.')
print('The discrete equations can be invoked by passing grid=\'discrete\' to the BatchReactor class.')
print('This example also demonstrates the use of nondimensional input.')
print('We provide the normalized concentrations and the phase partition coefficients.')
print('Then, the code no longer needs the temperature, volume, melt mass, and monomer mass.')
print('See lines 41-50 of \'examples_batch.py\' script.')

nmax = 110
grid = 'discrete'
n = numpy.arange(1, nmax+1, 1)
concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
rho = concs / numpy.inner(n, concs)
tmax = 100.0

reactor = BatchReactor(nmax=nmax, grid=grid, rho=rho, alpha1m=numpy.ones(nmax))
t2, y2 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
n2 = numpy.copy(reactor.n)
a2 = numpy.copy(reactor.alpha1m0)


print()
print('Case 3: Continuum Approximation with Phase Partition / Mixed Dimensional & Nondimensional Input')
print('This example solves the continuum approximation of the population balance equations with nonzero phase partition coefficients.')
print('The continuum approximation can be invoked by passing grid=\'continuum\' and a nonzero value of mesh to the BatchReactor class.')
print('This example also demonstrates the use of partial nondimensional input.')
print('We provide the normalized concentrations,')
print('but we enter the temperature in K, the headspace volume in L, the melt mass in g, and the monomer mass in g/mol.')
print('Then, the code computes the nondimensional Henry\'s constants and the phase partition coefficients.')
print('See lines 63-77 of \'examples_batch.py\' script.')

nmax = 110.0
mesh = 500
grid = 'continuum'
n = numpy.linspace(1.0, nmax, mesh)
dn = n[1] - n[0]
concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
w = numpy.ones_like(n)
w[0] = w[-1] = 0.5
rho = ( concs ) / ( numpy.einsum('i,i,i->', w, n, concs) * dn )
tmax = 100.0

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, rho=rho, temp=573.15, volume=1.0, mass=10.0, monomer=14.027)
t3, y3 = reactor.solve(tmax, gtol=1e-9, rtol=1e-9, atol=1e-9)
n3 = numpy.copy(reactor.n)
a3 = numpy.copy(reactor.alpha1m0)


print()
print('Case 4: Continuum Approximation with No Phase Partition / Mixed Dimensional & Nondimensional Input')
print('This example solves the continuum approximation of the population balance equations with trivial phase partition coefficients.')
print('The continuum approximation can be invoked by passing grid=\'continuum\' and a nonzero value of mesh to the BatchReactor class.')
print('This example also demonstrates the use of partial nondimensional input.')
print('We provide the phase partition coefficients,')
print('but we enter the concentrations in arbitrary units.')
print('Then, the code normalizes them to the total number of monomer units.')
print('See lines 90-100 of \'examples_batch.py\' script.')

nmax = 110.0
mesh = 500
grid = 'continuum'
n = numpy.linspace(1.0, nmax, mesh)
concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
tmax = 100.0

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, concs=concs, alpha1m=numpy.ones(mesh))
t4, y4 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
n4 = numpy.copy(reactor.n)
a4 = numpy.copy(reactor.alpha1m0)


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
print('See lines 115-137 of \'examples_batch.py\' script.')

nmax = 10.0**2.10
mesh = 500
grid = 'log_n'
n = numpy.logspace(0.0, numpy.log10(nmax), mesh)
concs = numpy.exp(-0.5*((numpy.log(n)-numpy.log(100.0))/(0.01*numpy.log(10.0)))**2)
tmax = 100.0

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, concs=concs, temp=573.15, volume=1.0, mass=10.0, monomer=14.027)
t5, y5 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
n5 = numpy.copy(reactor.n)
a5 = numpy.copy(reactor.alpha1m0)

nmax = int(10.0**2.10)+1
mesh = 0
grid = 'discrete'
n = numpy.arange(1, nmax+1, 1)
concs = (1.0/n) * numpy.exp(-0.5*((numpy.log(n)-numpy.log(100.0))/(0.01*numpy.log(10.0)))**2)
tmax = 100.0

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, concs=concs, temp=573.15, volume=1.0, mass=10.0, monomer=14.027)
t5b, y5b = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
n5b = numpy.copy(reactor.n)
a5b = numpy.copy(reactor.alpha1m0)


print()
print('Case 6: Log Scale Equation with No Phase Partition / Mixed Dimensional & Nondimensional Input')
print('This example solves the population balance equations in a logarithmic scale with trivial phase partition coefficients.')
print('The logarithmic equations can be invoked by passing grid=\'log_n\' to the BatchReactor class.')
print('This example also demonstrates the use of partial nondimensional input.')
print('We provide the phase partition coefficients,')
print('but we enter the concentrations in arbitrary units.')
print('Then, the code normalizes them to the total number of monomer units.')
print('For reference, we also solve the discrete equations.')
print('See lines 151-173 of \'examples_batch.py\' script.')

nmax = 10.0**2.10
mesh = 500
grid = 'log_n'
n = numpy.logspace(0.0, numpy.log10(nmax), mesh)
concs = numpy.exp(-0.5*((numpy.log(n)-numpy.log(100.0))/(0.01*numpy.log(10.0)))**2)
tmax = 100.0

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, concs=concs, alpha1m=numpy.ones(mesh))
t6, y6 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
n6 = numpy.copy(reactor.n)
a6 = numpy.copy(reactor.alpha1m0)

nmax = int(10.0**2.10)+1
mesh = 0
grid = 'discrete'
n = numpy.arange(1, nmax+1, 1)
concs = (1.0/n) * numpy.exp(-0.5*((numpy.log(n)-numpy.log(100.0))/(0.01*numpy.log(10.0)))**2)
tmax = 100.0

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, concs=concs, alpha1m=numpy.ones(nmax))
t6b, y6b = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
n6b = numpy.copy(reactor.n)
a6b = numpy.copy(reactor.alpha1m0)


print()

