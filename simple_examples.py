import numpy

from calculator import Calculator


print()
print('Case 1: Uniform Mole/Mass Fractions')
print('Suppose that there is an equal amount of each species in the gas/liquid phase.')
print('For an example with gas-phase mole fractions as the starting point,')
print('imagine a headspace where the 11 species (H2 C1-C8 C2H4 iC4) have mole fractions of 1/11.')
print('For an example with liquid-phase mass fractions as the starting point,')
print('imagine a melt where each of the 11 species have a mass fraction of 0.01.')
print('Indeed, the liquid-phase mass fractions should add to w < 1,')
print('since a lot of the melt should be long hydrocarbons that do not vaporize.')
print('Lines 17-26 of \'simple_examples.py\' script.')

T = 473.15
P = 1.0

calc = Calculator(temp=T, pressure=P)

y1 = numpy.ones(11) / 11.
w1 = calc.get_liquidphase(y1)

w2 = numpy.full(11, 0.01)
y2, P2 = calc.get_gasphase(w2, get_pressure=True)

print()
print('Gas-Phase Mole Fractions (y1) to Liquid-Phase Mass Fractions (w1)')
print('T = {:g} K'.format(T))
print('P = {:g} atm'.format(P))
print('{:>5s} {:>10s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'MW', 'y1', 'w1'))
print('-' * 49)
for ( name, Hi, Mi, value1, value2 ) in zip( calc.names, calc.Hv, calc.MW, y1, w1 ):
    print('{:>5s} {:10.1f} {:10.3f} {:10f} {:10f}'.format( name, Hi, Mi, value1, value2 ))
print('-' * 49)
print('{:>5s} {:>10s} {:>10s} {:10f} {:10f}'.format( 'Total', '', '', numpy.sum(y1), numpy.sum(w1) ))
print()
print('Liquid-Phase Mass Fractions (w2) to Gas-Phase Mole Fractions (y2)')
print('T = {:g} K'.format(T))
print('P = {:g} atm (est.)'.format(P2))
print('{:>5s} {:>10s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'MW', 'w2', 'y2'))
print('-' * 49)
for ( name, Hi, Mi, value1, value2 ) in zip( calc.names, calc.Hv, calc.MW, w2, y2 ):
    print('{:>5s} {:10.1f} {:10.3f} {:10f} {:10f}'.format( name, Hi, Mi, value1, value2 ))
print('-' * 49)
print('{:>5s} {:>10s} {:>10s} {:10f} {:10f}'.format( 'Total', '', '', numpy.sum(w2), numpy.sum(y2) ))
print()


print()
print('Case 2: Linear/Reciprocal Mole/Mass Fractions')
print('For an example with gas-phase mole fractions as the starting point,')
print('suppose that the mole fraction of each species in the gas phase is proportional to the reciprocal molar mass.')
print('For an example with liquid-phase mass fractions as the starting point,')
print('suppose that the mass fraction of each species in the liquid phase is proportional to the molar mass.')
print('We normalize so that total mass fraction of the volatiles species in the melt does not exceed 0.11.')
print('Lines 60-71 of \'simple_examples.py\' script.')

T = 473.15
P = 1.0

calc = Calculator(temp=T, pressure=P)

m = numpy.array([ 2.016, 16.04, 30.07, 44.09, 58.12, 72.15, 86.17, 100.20, 114.22, 28.05, 58.12 ])

y3 = 1.0/m / numpy.sum(1.0/m)
w3 = calc.get_liquidphase(y3)

w4 = 0.11 * m / numpy.sum(m)
y4, P4 = calc.get_gasphase(w4, get_pressure=True)

print()
print('Gas-Phase Mole Fractions (y3) to Liquid-Phase Mass Fractions (w3)')
print('T = {:g} K'.format(T))
print('P = {:g} atm'.format(P))
print('{:>5s} {:>10s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'MW', 'y3', 'w3'))
print('-' * 49)
for ( name, Hi, Mi, value1, value2 ) in zip( calc.names, calc.Hv, calc.MW, y3, w3 ):
    print('{:>5s} {:10.1f} {:10.3f} {:10f} {:10f}'.format( name, Hi, Mi, value1, value2 ))
print('-' * 49)
print('{:>5s} {:>10s} {:>10s} {:10f} {:10f}'.format( 'Total', '', '', numpy.sum(y3), numpy.sum(w3) ))
print()
print('Liquid-Phase Mass Fractions (w4) to Gas-Phase Mole Fractions (y4)')
print('T = {:g} K'.format(T))
print('P = {:g} atm (est.)'.format(P4))
print('{:>5s} {:>10s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'MW', 'w4', 'y4'))
print('-' * 49)
for ( name, Hi, Mi, value1, value2 ) in zip( calc.names, calc.Hv, calc.MW, w4, y4 ):
    print('{:>5s} {:10.1f} {:10.3f} {:10f} {:10f}'.format( name, Hi, Mi, value1, value2 ))
print('-' * 49)
print('{:>5s} {:>10s} {:>10s} {:10f} {:10f}'.format( 'Total', '', '', numpy.sum(w4), numpy.sum(y4) ))
print()

