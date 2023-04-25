import numpy
import sys

sys.path.append('..')
from calculator import Calculator


print()
print('Phase Partition Coefficients')
print('We compute the gas-phase and liquid-phase partition coefficients inside a reactor of a given volume.')
print('For now, we assume that the reactor contains a small amount of reagents.')
print('See lines 14-22 of \'partition.py\' script.')

T = 473.15
P = 1.0
W = 100.0
V = 10.0

calc = Calculator(temp=T, pressure=P, mass=W, volume=V)

alpha   = calc.get_gaspart()
alpha1m = calc.get_liquidpart()

print()
print('Gas-Phase and Liquid-Phase Partition Coefficients')
print('T = {:g} K'.format(T))
print('P = {:g} atm'.format(P))
print('W = {:g} g'.format(W))
print('V = {:g} L'.format(V))
print('{:>5s} {:>10s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'MW', 'alpha', '1-alpha'))
print('-' * 49)
for ( name, Hi, Mi, value1, value2 ) in zip( calc.names, calc.Hv, calc.MW, alpha, alpha1m ):
    print('{:>5s} {:10.1f} {:10.3f} {:10f} {:10f}'.format( name, Hi, Mi, value1, value2 ))
print()

