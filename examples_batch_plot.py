import numpy

from matplotlib import cm, pyplot
pyplot.rcParams.update({'font.size': 14})

from reactor_batch import BatchReactor
from examples_batch import t1, y1, n1, a1, t2, y2, n2, a2, t3, y3, n3, a3, t4, y4, n4, a4

prune = 5
nwin = 100 # 25
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

prune = 5
nwin = 100 # 25
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
nwin = 105.0 # 25.0
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

prune = 250
nwin = 105.0 # 25.0
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
ax1.set_ylim(1e-10, y4[n4 <= nwin].max())
ax1.set_yscale('log')
ax2.plot(n4, a4, 'k--')
ax2.set_ylim(ylim)
ax2.set_ylabel('Liquid-Phase Partition')
pyplot.tight_layout()
pyplot.savefig('examples_batch_nopartition_continuum.png')

