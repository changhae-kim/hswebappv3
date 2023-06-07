import numpy
import sys

sys.path.append('..')
from reactor import BatchReactor
from utils import plot_populations


author = sys.argv[1]


if author == 'celik':

    temp = 573.15
    volume = 0.3
    mass = 3.0
    monomer = 14.027
    dens = 920.0

    Mw = 22150.0 / monomer
    Mn = 8150.0 / monomer
    mu = 0.5*numpy.log(Mw*Mn)
    sigma = numpy.log(Mw/Mn)**0.5

elif author == 'yiyu':

    temp = 573.15
    volume = 0.3
    mass = 10.0
    monomer = 14.027
    dens = 940.0

    Mw = 4000.0 / monomer
    Mn = 2800.0 / monomer
    mu = 0.5*numpy.log(Mw*Mn)
    sigma = numpy.log(Mw/Mn)**0.5

else:

    print('available authors: celik, yiyu')
    exit()


nmax = 10.0**6.0 / monomer
mesh = 500
grid = 'logn'
n = numpy.logspace(0.0, numpy.log10(nmax), mesh)
concs = (1.0/n**2)*numpy.exp(-0.5*((numpy.log(n)-mu)/(sigma))**2)
tmax = 0.2

reactor = BatchReactor(nmax=nmax, mesh=mesh, grid=grid, concs=concs, temp=temp, volume=volume, mass=mass, monomer=monomer, dens=dens, rand=1.0)
n = reactor.n
alpha1m = reactor.alpha1m
#alpha1m = numpy.ones_like(reactor.alpha1m)
t, rho = reactor.solve(tmax, alpha1m=alpha1m, gtol=1e-12, rtol=1e-12, atol=1e-12)

n, dwdn = reactor.postprocess('dwdn')
n, dwdlogn = reactor.postprocess('dwdlogn')
n, dWdn = reactor.postprocess('dWdn')
n, dWdlogn = reactor.postprocess('dWdlogn')
M, dWdM = reactor.postprocess('dWdM')
M, dWdlogM = reactor.postprocess('dWdlogM')

plot_populations(t, n, rho, alpha1m, 'Chain Length', 'Chain Concentration', '{:s}_rho_n.png'.format(author))
plot_populations(t, M, rho, alpha1m, 'Molar Mass', 'Chain Concentration', '{:s}_rho_M.png'.format(author))
plot_populations(t, n, dwdn, alpha1m, 'Chain Length', r'$d\widetilde{W}/d{n}$', '{:s}_dwdn.png'.format(author))
plot_populations(t, M, dWdM, alpha1m, 'Molar Mass', '$d{W}/d{M}$', '{:s}_dWdM.png'.format(author))
plot_populations(t, n, dwdlogn, alpha1m, 'Chain Length', r'$d\widetilde{W}/d\log{n}$', '{:s}_dwdlogn.png'.format(author))
plot_populations(t, M, dWdlogM, alpha1m, 'Molar Mass', '$d{W}/d\log{M}$', '{:s}_dWdlogM.png'.format(author), ytick='%.1f')


if author == 'celik':

    y = dWdlogM
    ii = []
    for i, yi in enumerate(y.T):
        if 1e+3 < M[yi.argmax()] and M[yi.argmax()] < 2e+4:
            ii.append(i)
    s = t[ii]
    x = M
    y = ((y.T)[ii]).T
    plot_populations(s, x, y, alpha1m, 'Molar Mass', '$d{W}/d\log{M}$', '{:s}_dWdlogM_fit1.png'.format(author), ytick='%.1f')

    y = (alpha1m*dWdlogM.T).T
    ii = []
    for i, yi in enumerate(y.T):
        if 1e+3 < M[yi.argmax()] and M[yi.argmax()] < 2e+4:
            ii.append(i)
    s = t[ii]
    x = M
    y = ((y.T)[ii]).T
    plot_populations(s, x, y, alpha1m, 'Molar Mass', r'$(1-\alpha) d{W}/d\log{M}$', '{:s}_dWdlogM_fit2.png'.format(author), ytick='%.1f')

elif author == 'yiyu':

    plot_populations(t, n, dWdn, alpha1m, 'Chain Length', '$d{W}/d{n}$', '{:s}_dWdn.png'.format(author), xlim=[8,59], xscale='linear')

    y = dWdn
    ii = []
    for i, yi in enumerate(y.T):
        if 15.0 < n[yi.argmax()] and n[yi.argmax()] < 30.0:
            ii.append(i)
    s = t[ii]
    x = n
    y = ((y.T)[ii]).T
    plot_populations(s, x, y, alpha1m, 'Chain Length', '$d{W}/d{n}$', '{:s}_dWdn_fit1.png'.format(author), xlim=[8,59], xscale='linear')

    y = (alpha1m*dWdn.T).T
    ii = []
    for i, yi in enumerate(y.T):
        if 15.0 < n[yi.argmax()] and n[yi.argmax()] < 30.0:
            ii.append(i)
    s = t[ii]
    x = n
    y = ((y.T)[ii]).T
    plot_populations(s, x, y, alpha1m, 'Chain Length', r'$(1-\alpha) d{W}/d{n}$', '{:s}_dWdn_fit2.png'.format(author), xlim=[8,59], xscale='linear')

