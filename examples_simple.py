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
print('See lines 17-26 of \'examples_simple.py\' script.')

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
print('Case 2: Poisson/Gaussian Mole/Mass Fractions')
print('For an example with gas-phase mole fractions as the starting point,')
print('suppose that the mole fractions in the gas phase exhibit a Poisson distribution in the molar masses (lambda = 14.0266).')
print('For an example with liquid-phase mass fractions as the starting point,')
print('suppose that the mass fractions in the liquid phase exhibit a Gaussian distribution in the molar masses (mu = 30*14.0266, sigma = 10*14.266).')
print('We normalize so that the mass fractions of the volatiles species in the liquid phase add to 0.11.')
print('See lines 60-73 of \'examples_simple.py\' script.')

T = 473.15
P = 1.0

calc = Calculator(temp=T, pressure=P)

m = calc.MW

y3 = numpy.exp(-m/14.0266)
y3 = y3 / numpy.sum(y3)
w3 = calc.get_liquidphase(y3)

w4 = numpy.exp(-0.5*((m-30*14.0266)/(10*14.0266))**2)
w4 = 0.11 * w4 / numpy.sum(w4)
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

