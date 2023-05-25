import numpy
import sys

from matplotlib import cm, pyplot
pyplot.rcParams.update({'font.size': 14})

sys.path.append('..')
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
    n1 = reactor.n
    a1 = reactor.alpha1m0
    #t1, y1 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
    t1, y1 = reactor.solve(tmax, alpha=reactor.alpha0, alpha1m=reactor.alpha1m0, gtol=1e-6, rtol=1e-6, atol=1e-6)

    #G1, Gin1, Gout1 = reactor.cointegrate()
    G1, Gin1, Gout1 = reactor.cointegrate(alpha=reactor.alpha0, alpha1m=reactor.alpha1m0)

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
    pyplot.savefig('cstr_partition_discrete_io_{:.0e}.png'.format(flux))
    pyplot.close()

    prune = 2
    nwin = 110
    fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    cmap = cm.viridis(t1/t1.max())
    for i, _ in enumerate(t1):
        if i % prune == 0:
            ax1.plot(n1[:nwin], Gout1[:nwin, i], color=cmap[i])
    ax1.set_xlabel('Chain Length')
    ax1.set_ylabel('Chain Concentration')
    ax1.set_xlim(0, nwin)
    ax2.plot(n1, a1, 'k--')
    ylim = ax2.get_ylim()
    ax2.set_ylabel('Liquid-Phase Partition')
    pyplot.tight_layout()
    pyplot.savefig('cstr_partition_discrete_io_o_{:.0e}.png'.format(flux))
    pyplot.close()


for flux in [0.001, 0.01, 0.1, 1.0]:

    nmax = 110
    grid = 'discrete'
    n = numpy.arange(1, nmax+1, 1)
    concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
    influx = numpy.zeros_like(concs)
    outflux = [flux, 0.0]
    tmax = 100.0

    reactor = CSTReactor(nmax=nmax, grid=grid, concs=concs, influx=influx, outflux=outflux, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0)
    n2 = reactor.n
    a2 = reactor.alpha1m0
    #t2, y2 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
    t2, y2 = reactor.solve(tmax, alpha=reactor.alpha0, alpha1m=reactor.alpha1m0, gtol=1e-6, rtol=1e-6, atol=1e-6)

    #G2, Gin2, Gout2 = reactor.cointegrate()
    G2, Gin2, Gout2 = reactor.cointegrate(alpha=reactor.alpha0, alpha1m=reactor.alpha1m0)

    prune = 2
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
    ylim = ax2.get_ylim()
    ax2.set_ylabel('Liquid-Phase Partition')
    pyplot.tight_layout()
    pyplot.savefig('cstr_partition_discrete_o_{:.0e}.png'.format(flux))
    pyplot.close()

    prune = 2
    nwin = 110
    fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    cmap = cm.viridis(t2/t2.max())
    for i, _ in enumerate(t2):
        if i % prune == 0:
            ax1.plot(n2[:nwin], Gout2[:nwin, i], color=cmap[i])
    ax1.set_xlabel('Chain Length')
    ax1.set_ylabel('Chain Concentration')
    ax1.set_xlim(0, nwin)
    ax2.plot(n2, a2, 'k--')
    ylim = ax2.get_ylim()
    ax2.set_ylabel('Liquid-Phase Partition')
    pyplot.tight_layout()
    pyplot.savefig('cstr_partition_discrete_o_o_{:.0e}.png'.format(flux))
    pyplot.close()


for flux in [0.001, 0.01, 0.1, 1.0]:

    nmax = 110.0
    mesh = 500
    grid = 'continuum'
    n = numpy.linspace(1.0, nmax, mesh)
    concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
    influx = flux * concs
    outflux = [flux, 0.0]
    tmax = 100.0

    reactor = CSTReactor(nmax=nmax, mesh=mesh, grid=grid, concs=concs, influx=influx, outflux=outflux, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0)
    n3 = reactor.n
    a3 = reactor.alpha1m0
    #t3, y3 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
    t3, y3 = reactor.solve(tmax, alpha=reactor.alpha0, alpha1m=reactor.alpha1m0, gtol=1e-6, rtol=1e-6, atol=1e-6)

    #G3, Gin3, Gout3 = reactor.cointegrate()
    G3, Gin3, Gout3 = reactor.cointegrate(alpha=reactor.alpha0, alpha1m=reactor.alpha1m0)

    prune = 2
    nwin = 110
    fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    cmap = cm.viridis(t3/t3.max())
    for i, _ in enumerate(t3):
        if i % prune == 0:
            ax1.plot(n3[n3 <= nwin], y3[n3 <= nwin][:, i], color=cmap[i])
    ax1.set_xlabel('Chain Length')
    ax1.set_ylabel('Chain Concentration')
    ax1.set_xlim(0, nwin)
    ax2.plot(n3, a3, 'k--')
    ylim = ax2.get_ylim()
    ax2.set_ylabel('Liquid-Phase Partition')
    pyplot.tight_layout()
    pyplot.savefig('cstr_partition_continuum_io_{:.0e}.png'.format(flux))
    pyplot.close()

    prune = 2
    nwin = 110
    fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    cmap = cm.viridis(t3/t3.max())
    for i, _ in enumerate(t3):
        if i % prune == 0:
            ax1.plot(n3[n3 <= nwin], Gout3[n3 <= nwin][:, i], color=cmap[i])
    ax1.set_xlabel('Chain Length')
    ax1.set_ylabel('Chain Concentration')
    ax1.set_xlim(0, nwin)
    ax2.plot(n3, a3, 'k--')
    ylim = ax2.get_ylim()
    ax2.set_ylabel('Liquid-Phase Partition')
    pyplot.tight_layout()
    pyplot.savefig('cstr_partition_continuum_io_o_{:.0e}.png'.format(flux))
    pyplot.close()


for flux in [0.001, 0.01, 0.1, 1.0]:

    nmax = 110.0
    mesh = 500
    grid = 'continuum'
    n = numpy.linspace(1.0, nmax, mesh)
    concs = numpy.exp(-0.5*((n-100.0)/(2.0))**2)
    influx = numpy.zeros_like(concs)
    outflux = [flux, 0.0]
    tmax = 100.0

    reactor = CSTReactor(nmax=nmax, mesh=mesh, grid=grid, concs=concs, influx=influx, outflux=outflux, temp=573.15, volume=1.0, mass=10.0, monomer=14.027, dens=920.0)
    n4 = reactor.n
    a4 = reactor.alpha1m0
    #t4, y4 = reactor.solve(tmax, gtol=1e-6, rtol=1e-6, atol=1e-6)
    t4, y4 = reactor.solve(tmax, alpha=reactor.alpha0, alpha1m=reactor.alpha1m0, gtol=1e-6, rtol=1e-6, atol=1e-6)

    #G4, Gin4, Gout4 = reactor.cointegrate()
    G4, Gin4, Gout4 = reactor.cointegrate(alpha=reactor.alpha0, alpha1m=reactor.alpha1m0)

    prune = 2
    nwin = 110
    fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    cmap = cm.viridis(t4/t4.max())
    for i, _ in enumerate(t4):
        if i % prune == 0:
            ax1.plot(n4[n4 <= nwin], y4[n4 <= nwin][:, i], color=cmap[i])
    ax1.set_xlabel('Chain Length')
    ax1.set_ylabel('Chain Concentration')
    ax1.set_xlim(0, nwin)
    ax2.plot(n4, a4, 'k--')
    ylim = ax2.get_ylim()
    ax2.set_ylabel('Liquid-Phase Partition')
    pyplot.tight_layout()
    pyplot.savefig('cstr_partition_continuum_o_{:.0e}.png'.format(flux))
    pyplot.close()

    prune = 2
    nwin = 110
    fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    cmap = cm.viridis(t4/t4.max())
    for i, _ in enumerate(t4):
        if i % prune == 0:
            ax1.plot(n4[n4 <= nwin], Gout4[n4 <= nwin][:, i], color=cmap[i])
    ax1.set_xlabel('Chain Length')
    ax1.set_ylabel('Chain Concentration')
    ax1.set_xlim(0, nwin)
    ax2.plot(n4, a4, 'k--')
    ylim = ax2.get_ylim()
    ax2.set_ylabel('Liquid-Phase Partition')
    pyplot.tight_layout()
    pyplot.savefig('cstr_partition_continuum_o_o_{:.0e}.png'.format(flux))
    pyplot.close()

