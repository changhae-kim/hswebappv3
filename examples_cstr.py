import numpy

from matplotlib import cm, pyplot
pyplot.rcParams.update({'font.size': 14})

from reactor import CSTReactor


for flux in [0.001, 0.01, 0.1, 1.0]:

    nmax = 110
    grid = 'discrete'
    n = numpy.arange(1, nmax+1, 1)
    concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
    influx = flux * concs
    outflux = [flux, 0.0]
    tmax = 100.0

    reactor = CSTReactor(nmax=nmax, grid=grid, concs=concs, influx=influx, outflux=outflux, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0)
    t1, y1 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
    n1 = numpy.copy(reactor.n)
    a1 = numpy.copy(reactor.alpha1m0)

    prune = 2
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
    pyplot.savefig('examples_cstr_partition_discrete_io{:.0e}.png'.format(flux))


for flux in [0.001, 0.01, 0.1, 1.0]:

    nmax = 110
    grid = 'discrete'
    n = numpy.arange(1, nmax+1, 1)
    concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
    influx = numpy.zeros_like(concs)
    outflux = [flux, 0.0]
    tmax = 100.0

    reactor = CSTReactor(nmax=nmax, grid=grid, concs=concs, influx=influx, outflux=outflux, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0)
    t1, y1 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
    n1 = numpy.copy(reactor.n)
    a1 = numpy.copy(reactor.alpha1m0)

    prune = 2
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
    pyplot.savefig('examples_cstr_partition_discrete_o{:.0e}.png'.format(flux))


