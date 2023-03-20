import numpy

from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 14})

from simple_examples import calc, y1, w1, y2, w2, y3, w3, y4, w4

x = calc.names

fig = pyplot.figure(figsize=(9.6, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twinx()
color = 'tab:blue'
ax1.bar(x, y1, width=-0.4, align='edge', color=color)
ax1.set_ylabel('Gas-Phase Mole Fractions', color=color)
ax1.tick_params(axis='y', labelcolor=color)
color = 'tab:orange'
ax2.bar(x, w1, width=+0.4, align='edge', color=color)
ax2.set_yscale('log')
ax2.set_ylabel('Liquid-Phase Mass Fractions', color=color)
ax2.tick_params(axis='y', labelcolor=color)
pyplot.title('Gas-Phase Mole Fractions to Liquid-Phase Mass Fractions')
pyplot.tight_layout()
pyplot.savefig('simple_examples_y1w1.png')

fig = pyplot.figure(figsize=(9.6, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twinx()
color = 'tab:blue'
ax1.bar(x, y2, width=-0.4, align='edge', color=color)
ax1.set_yscale('log')
ax1.set_ylabel('Gas-Phase Mole Fractions', color=color)
ax1.tick_params(axis='y', labelcolor=color)
color = 'tab:orange'
ax2.bar(x, w2, width=+0.4, align='edge', color=color)
ax2.set_ylabel('Liquid-Phase Mass Fractions', color=color)
ax2.tick_params(axis='y', labelcolor=color)
pyplot.title('Liquid-Phase Mass Fractions to Gas-Phase Mole Fractions')
pyplot.tight_layout()
pyplot.savefig('simple_examples_y2w2.png')

fig = pyplot.figure(figsize=(9.6, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twinx()
color = 'tab:blue'
ax1.bar(x, y3, width=-0.4, align='edge', color=color)
ax1.set_yscale('log')
ax1.set_ylabel('Gas-Phase Mole Fractions', color=color)
ax1.tick_params(axis='y', labelcolor=color)
color = 'tab:orange'
ax2.bar(x, w3, width=+0.4, align='edge', color=color)
ax2.set_ylabel('Liquid-Phase Mass Fractions', color=color)
ax2.tick_params(axis='y', labelcolor=color)
pyplot.title('Gas-Phase Mole Fractions to Liquid-Phase Mass Fractions')
pyplot.tight_layout()
pyplot.savefig('simple_examples_y3w3.png')

fig = pyplot.figure(figsize=(9.6, 4.8), dpi=150)
ax1 = fig.subplots()
ax2 = ax1.twinx()
color = 'tab:blue'
ax1.bar(x, y4, width=-0.4, align='edge', color=color)
ax1.set_yscale('log')
ax1.set_ylabel('Gas-Phase Mole Fractions', color=color)
ax1.tick_params(axis='y', labelcolor=color)
color = 'tab:orange'
ax2.bar(x, w4, width=+0.4, align='edge', color=color)
ax2.set_ylabel('Liquid-Phase Mass Fractions', color=color)
ax2.tick_params(axis='y', labelcolor=color)
pyplot.title('Liquid-Phase Mass Fractions to Gas-Phase Mole Fractions')
pyplot.tight_layout()
pyplot.savefig('simple_examples_y4w4.png')

