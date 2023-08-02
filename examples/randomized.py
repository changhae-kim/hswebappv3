import numpy
import sys

sys.path.append('..')
from calculator import Calculator


print()
print('Case 1: Array-like Input, Array-like Output')
print('For the input, construct an array-like object with mole/mass fractions in the order:')
print('H2 C1 C2 C3 C4 C5 C6 C7 C8 N2 CO2 VinAce C2H4 iC4 Benzene Toluene')
print('For an example with gas-phase mole fractions as the starting point,')
print('we choose a random number n of species and generate n random numbers that add to 1.')
print('For an example with liquid-phase mass fractions as the starting point,')
print('we choose a random number n of species and generate n random numbers that add to a random number w < 1.')
print('Indeed, a lot of the melt should be long hydrocarbons (C20-C30) that do not vaporize.')
print('See lines 19-39 of \'randomized.py\' script.')

T = 473.15
P = 1.0

calc = Calculator(temp=T, pressure=P)

n1 = numpy.random.randint(1, 17)
i1 = list(sorted( numpy.random.choice( list(range(16)), n1, replace=False ) ))
y1 = numpy.zeros(16)
y1[i1] = numpy.random.rand(n1)
y1 = y1 / numpy.sum(y1)

w1 = calc.get_liquidphase(y1)

n2 = numpy.random.randint(1, 17)
i2 = list(sorted( numpy.random.choice( list(range(16)), n2, replace=False ) ))
W2 = numpy.random.rand(1)
w2 = numpy.zeros(16)
w2[i2] = numpy.random.rand(n2)
w2 = W2 * w2 / numpy.sum(w2)

y2, P2 = calc.get_gasphase(w2, get_pressure=True)

print()
print('Gas-Phase Mole Fractions (y1) to Liquid-Phase Mass Fractions (w1)')
print('T = {:g} K'.format(T))
print('P = {:g} atm'.format(P))
print('{:>10s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'y1', 'w1'))
print('-' * 43)
for ( name, Hi, value1, value2 ) in zip( calc.names, calc.Hv, y1, w1 ):
    print('{:>10s} {:10.1f} {:10f} {:10f}'.format( name, Hi, value1, value2 ))
print('-' * 43)
print('{:>10s} {:>10s} {:10f} {:10f}'.format( 'Total', '', numpy.sum(y1), numpy.sum(w1) ))
print()
print('Liquid-Phase Mass Fractions (w2) to Gas-Phase Mole Fractions (y2)')
print('T = {:g} K'.format(T))
print('P = {:g} atm (est.)'.format(P2))
print('{:>10s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'w2', 'y2'))
print('-' * 43)
for ( name, Hi, value1, value2 ) in zip( calc.names, calc.Hv, w2, y2 ):
    print('{:>10s} {:10.1f} {:10f} {:10f}'.format( name, Hi, value1, value2 ))
print('-' * 43)
print('{:>10s} {:>10s} {:10f} {:10f}'.format( 'Total', '', numpy.sum(w2), numpy.sum(y2) ))
print()


print()
print('Case 2: Dictionary Input, Dictionary Output')
print('For the input, construct a dictionary object.')
print('The keys must be chosen from:')
print('H2 C1 C2 C3 C4 C5 C6 C7 C8 N2 CO2 VinAce C2H4 iC4 Benzene Toluene')
print('The values must be numbers.')
print('If the mass/mole fraction of a species is 0, then you may omit its entry.')
print('For an example with gas-phase mole fractions as the starting point,')
print('we choose a random number n of species and generate n random numbers that add to 1.')
print('For an example with liquid-phase mass fractions as the starting point,')
print('we choose a random number n of species and generate n random numbers that add to a random number w < 1.')
print('Indeed, a lot of the melt should be long hydrocarbons (C20-C30) that do not vaporize.')
print('See lines 78-102 of \'randomized.py\' script.')

T = 473.15
P = 1.0

calc = Calculator(temp=T, pressure=P)

names = [ 'H2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'N2', 'CO2', 'VinAce', 'C2H4', 'iC4', 'Benzene', 'Toluene', ]

n3 = numpy.random.randint(1, 17)
i3 = list(sorted( numpy.random.choice( list(range(16)), n3, replace=False ) ))
y3 = numpy.zeros(16)
y3[i3] = numpy.random.rand(n3)
y3 = y3 / numpy.sum(y3)
y3 = { names[i]: y3[i] for i in i3 }

w3 = calc.get_liquidphase(y3, diction=True)

n4 = numpy.random.randint(1, 17)
i4 = list(sorted( numpy.random.choice( list(range(16)), n4, replace=False ) ))
W2 = numpy.random.rand(1)
w4 = numpy.zeros(16)
w4[i4] = numpy.random.rand(n4)
w4 = W2 * w4 / numpy.sum(w4)
w4 = { names[i]: w4[i] for i in i4 }

y4, P2 = calc.get_gasphase(w4, diction=True, get_pressure=True)

print()
print('Gas-Phase Mole Fractions (y3) to Liquid-Phase Mass Fractions (w3)')
print('T = {:g} K'.format(T))
print('P = {:g} atm'.format(P))
print('{:>10s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'y3', 'w3'))
print('-' * 43)
for ( Hi, ( name, value1 ), ( _, value2 ) ) in zip( calc.Hv, y3.items(), w3.items() ):
    print('{:>10s} {:10.1f} {:10f} {:10f}'.format( name, Hi, value1, value2 ))
print('-' * 43)
print('{:>10s} {:>10s} {:10f} {:10f}'.format( 'Total', '', sum(y3.values()), sum(w3.values()) ))
print()
print('Liquid-Phase Mass Fractions (w4) to Gas-Phase Mole Fractions (y4)')
print('T = {:g} K'.format(T))
print('P = {:g} atm (est.)'.format(P2))
print('{:>10s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'w4', 'y4'))
print('-' * 43)
for ( Hi, ( name, value1 ), ( _, value2 ) ) in zip( calc.Hv, w4.items(), y4.items() ):
    print('{:>10s} {:10.1f} {:10f} {:10f}'.format( name, Hi, value1, value2 ))
print('-' * 43)
print('{:>10s} {:>10s} {:10f} {:10f}'.format( 'Total', '', sum(w4.values()), sum(y4.values()) ))
print()

