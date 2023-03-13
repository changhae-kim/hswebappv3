import numpy

from calculator import Calculator


print()
print('Case 1: Array-like Input, Array-like Output')
print('For the input, construct an array-like object with mass/mole fractions in the order:')
print('H2 C1 C2 C3 C4 C5 C6 C7 C8 C2H4 iC4')
print('For examples of gas-phase mole fractions, we choose a random number n of species and generate n random numbers that add to 1.')
print('For examples of liquid-phase mass fractions, we choose a random number n of species and generate n random numbers that add to a random number w < 1;')
print('indeed, most of the liquid-phase mass should be long-chain hydrocarbons that do not vaporize.')
print('See Lines 15-35 of \'examples.py\' script.')

T = 473.15
P = 1.0

calc = Calculator(temp=T, pressure=P)

n1 = numpy.random.randint(1, 12)
i1 = list(sorted( numpy.random.choice( list(range(11)), n1, replace=False ) ))
y1 = numpy.zeros(11)
y1[i1] = numpy.random.rand(n1)
y1 = y1 / numpy.sum(y1)

w1 = calc.get_liquidphase(y1)

n2 = numpy.random.randint(1, 12)
i2 = list(sorted( numpy.random.choice( list(range(11)), n2, replace=False ) ))
W2 = numpy.random.rand(1)
w2 = numpy.zeros(11)
w2[i2] = numpy.random.rand(n2)
w2 = W2 * w2 / numpy.sum(w2)

y2 = calc.get_gasphase(w2)

print()
print('Gas-Phase Mole Fractions (y1) to Liquid-Phase Mass Fractions (w1)')
print('T = {:g} K'.format(T))
print('P = {:g} atm'.format(P))
print('{:>5s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'y1', 'w1'))
print('-' * 38)
for ( name, Hi, value1, value2 ) in zip( calc.names, calc.Hv, y1, w1 ):
    print('{:>5s} {:10.1f} {:10f} {:10f}'.format( name, Hi, value1, value2 ))
print('-' * 38)
print('{:>5s} {:>10s} {:10f} {:10f}'.format( 'Total', '', numpy.sum(y1), numpy.sum(w1) ))
print()
print('Liquid-Phase Mass Fractions (w2) to Gas-Phase Mole Fractions (y2)')
print('T = {:g} K'.format(T))
print('P = TBD')
print('{:>5s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'w2', 'y2'))
print('-' * 38)
for ( name, Hi, value1, value2 ) in zip( calc.names, calc.Hv, w2, y2 ):
    print('{:>5s} {:10.1f} {:10f} {:10f}'.format( name, Hi, value1, value2 ))
print('-' * 38)
print('{:>5s} {:>10s} {:10f} {:10f}'.format( 'Total', '', numpy.sum(w2), numpy.sum(y2) ))
print()


print()
print('Case 2: Dictionary Input, Dictionary Output')
print('For the input, construct a dictionary object.')
print('The keys must be chosen from:')
print('H2 C1 C2 C3 C4 C5 C6 C7 C8 C2H4 iC4')
print('The values must be numbers.')
print('If the mass/mole fraction of a species is 0, then you may omit its entry.')
print('For examples of gas-phase mole fractions, we choose a random number n of species and generate n random numbers that add to 1.')
print('For examples of liquid-phase mass fractions, we choose a random number n of species and generate n random numbers that add to a random number w < 1;')
print('indeed, most of the liquid-phase mass should be long-chain hydrocarbons that do not vaporize.')
print('See Lines 72-96 of \'examples.py\' script.')

T = 473.15
P = 1.0

calc = Calculator(temp=T, pressure=P)

names = ['H2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C2H4', 'iC4']

n1 = numpy.random.randint(1, 12)
i1 = list(sorted( numpy.random.choice( list(range(11)), n1, replace=False ) ))
y1 = numpy.zeros(11)
y1[i1] = numpy.random.rand(n1)
y1 = y1 / numpy.sum(y1)
y1 = { names[i]: y1[i] for i in i1 }

w1 = calc.get_liquidphase(y1, diction=True)

n2 = numpy.random.randint(1, 12)
i2 = list(sorted( numpy.random.choice( list(range(11)), n2, replace=False ) ))
W2 = numpy.random.rand(1)
w2 = numpy.zeros(11)
w2[i2] = numpy.random.rand(n2)
w2 = W2 * w2 / numpy.sum(w2)
w2 = { names[i]: w2[i] for i in i2 }

y2 = calc.get_gasphase(w2, diction=True)

print()
print('Gas-Phase Mole Fractions (y1) to Liquid-Phase Mass Fractions (w1)')
print('T = {:g} K'.format(T))
print('P = {:g} atm'.format(P))
print('{:>5s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'y1', 'w1'))
print('-' * 38)
for ( Hi, ( name, value1 ), ( _, value2 ) ) in zip( calc.Hv, y1.items(), w1.items() ):
    print('{:>5s} {:10.1f} {:10f} {:10f}'.format( name, Hi, value1, value2 ))
print('-' * 38)
print('{:>5s} {:>10s} {:10f} {:10f}'.format( 'Total', '', sum(y1.values()), sum(w1.values()) ))
print()
print('Liquid-Phase Mass Fractions (w2) to Gas-Phase Mole Fractions (y2)')
print('T = {:g} K'.format(T))
print('P = TBD')
print('{:>5s} {:>10s} {:>10s} {:>10s}'.format('', 'Hv', 'w2', 'y2'))
print('-' * 38)
for ( Hi, ( name, value1 ), ( _, value2 ) ) in zip( calc.Hv, w2.items(), y2.items() ):
    print('{:>5s} {:10.1f} {:10f} {:10f}'.format( name, Hi, value1, value2 ))
print('-' * 38)
print('{:>5s} {:>10s} {:10f} {:10f}'.format( 'Total', '', sum(w2.values()), sum(y2.values()) ))
print()

