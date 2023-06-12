import itertools
import numpy
import os, sys

from reactor import BatchReactor
from utils import plot_populations, plot_two_curves

nmax = 100
grid = 'discrete'
concs = numpy.ones(100)

temps = [ 423.15, 473.15, 523.15, 573.15 ]
volume = 1.0
masses = [ 10.0, 100.0, 1000.0 ]
monomer = 14.027
dens = 920.0

Mws = [ 22150, 17200, 70400, 420000, 115150, 4000, ]
Mns = [  8150, 15400, 64300, 158000,  33000, 2800, ]

X = []
for Mw, Mn in zip(Mws, Mns):
    nw = Mw / monomer
    nn = Mn / monomer
    sigma = numpy.log(nw/nn)**0.5 / numpy.log(10.0)
    mu = 0.5*numpy.log(nw*nn) / numpy.log(10.0)
    X.append([Mn, Mw/Mn, mu, sigma])
X = numpy.array(X)

print(X)

H0 = []
H1 = []
for temp in temps:
    H0i = []
    for mass in masses:
        reactor = BatchReactor(nmax=nmax, grid=grid, concs=concs, temp=temp, volume=volume, mass=mass, monomer=monomer, dens=dens)
        H0i.append(reactor.H0)
    H0.append(H0i)
    H1.append(reactor.H1)
H0 = numpy.array(H0)
H1 = numpy.array(H1)

print(H0)
print(H1)

