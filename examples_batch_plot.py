import numpy

from matplotlib import cm, pyplot
pyplot.rcParams.update({'font.size': 14})

from reactor import BatchReactor
from examples_batch import t1, y1, n1, a1, t2, y2, n2, a2, t3, y3, n3, a3, t4, y4, n4, a4
from examples_batch import t5, y5, n5, a5#, t6, y6, n6, a6

prune = 25
nwin = 100
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
pyplot.savefig('examples_batch_partition_exact.png')

prune = 25
nwin = 100
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
pyplot.savefig('examples_batch_nopartition_exact.png')

prune = 25
nwin = 105.0
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
pyplot.savefig('examples_batch_partition_continuum.png')

prune = 25
nwin = 105.0
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
pyplot.savefig('examples_batch_nopartition_continuum.png')

prune = 25
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
pyplot.savefig('examples_batch_partition_log.png')
'''
prune = 25
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
pyplot.savefig('examples_batch_nopartition_log.png')
'''
