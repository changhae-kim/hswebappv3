import numpy


compounds = {
        'H2'  : { 'H0': 109019.2869    , 'Hsol/R':  +622., 'MW':    2.016 },
        'C1'  : { 'H0':   6009.797611  , 'Hsol/R':   -14., 'MW':   16.04  },
        'C2'  : { 'H0':   2478.83417   , 'Hsol/R':  -609., 'MW':   30.07  },
        'C3'  : { 'H0':    950.4375513 , 'Hsol/R': -1091., 'MW':   44.09  },
        'C4'  : { 'H0':    563.801216  , 'Hsol/R': -1819., 'MW':   58.12  },
        'C5'  : { 'H0':    376.6074988 , 'Hsol/R': -2211., 'MW':   72.15  },
        'C6'  : { 'H0':    190.5018328 , 'Hsol/R': -2804., 'MW':   86.17  },
        'C7'  : { 'H0':    116.0282428 , 'Hsol/R': -3279., 'MW':  100.20  },
        'C8'  : { 'H0':     80.99987344, 'Hsol/R': -3724., 'MW':  114.22  },
        'C2H4': { 'H0':  22689.02157   , 'Hsol/R':  -661., 'MW':   28.05  },
        'iC4' : { 'H0':    626.8092598 , 'Hsol/R': -1631., 'MW':   58.12  },
        }


class Calculator():

    def __init__( self, temp=573.0, pressure=1.0 ):
        self.temp      = temp
        self.pressure  = pressure
        self.names     = [ key for key in compounds.keys() ]
        self.H0        = numpy.array([ value['H0'    ] for value in compounds.values() ])
        self.Hsol_R    = numpy.array([ value['Hsol/R'] for value in compounds.values() ])
        self.MW        = numpy.array([ value['MW'    ] for value in compounds.values() ])
        self.Hv        = self.get_henrysconst()
        return

    def get_henrysconst( self, temp=None ):
        if temp is None:
            temp = self.temp
        return self.H0 * numpy.exp( self.Hsol_R * ( 1.0/temp - 1.0/573.0 ) )

    def get_vector(self, diction):
        vector = numpy.zeros_like(self.Hv)
        for (key, value) in diction.items():
            assert key in self.names
            for i, name in enumerate(self.names):
                if key == name:
                    vector[i] = value
        return vector

    def get_diction(self, vector):
        diction = { name: vector[i] for i, name in enumerate(self.names) if vector[i] > 0.0 }
        return diction

    def get_gasphase( self, w, temp=None, diction=False, get_pressure=False, set_pressure=False ):

        if temp is None:
            temp = self.temp
            Hv   = self.Hv
        else:
            temp = temp
            Hv   = self.get_henrysconst(temp)

        if isinstance(w, dict):
            w = self.get_vector(w)
        else:
            w = numpy.array(w)

        p = w * Hv
        P = numpy.sum(p)
        y = p / P

        if set_pressure:
            self.pressure = P

        if diction:
            if get_pressure:
                return self.get_diction(y), P
            else:
                return self.get_diction(y)
        else:
            if get_pressure:
                return y, P
            else:
                return y

    def get_liquidphase( self, y, temp=None, pressure=None, diction=False ):

        if temp is None:
            temp = self.temp
            Hv   = self.Hv
        else:
            temp = temp
            Hv   = self.get_henrysconst(temp)

        if pressure is None:
            pressure = self.pressure
        else:
            pressure = pressure

        if isinstance(y, dict):
            y = self.get_vector(y)
        else:
            y = numpy.array(y)

        p = y * pressure
        w = p / Hv

        if diction:
            return self.get_diction(w)
        else:
            return w

