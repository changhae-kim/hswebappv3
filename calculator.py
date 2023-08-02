import numpy


compounds = {
    'H2'      : { 'H0': 109019.2869     , 'Hsol/R':  +622. , 'MW':   2.016 },
    'C1'      : { 'H0':   6009.797611   , 'Hsol/R':   -14. , 'MW':  16.04  },
    'C2'      : { 'H0':   2478.83417    , 'Hsol/R':  -609. , 'MW':  30.07  },
    'C3'      : { 'H0':    950.4375513  , 'Hsol/R': -1091. , 'MW':  44.09  },
    'C4'      : { 'H0':    563.801216   , 'Hsol/R': -1819. , 'MW':  58.12  },
    'C5'      : { 'H0':    376.6074988  , 'Hsol/R': -2211. , 'MW':  72.15  },
    'C6'      : { 'H0':    190.5018328  , 'Hsol/R': -2804. , 'MW':  86.17  },
    'C7'      : { 'H0':    116.0282428  , 'Hsol/R': -3279. , 'MW': 100.20  },
    'C8'      : { 'H0':     80.99987344 , 'Hsol/R': -3724. , 'MW': 114.22  },
    'N2'      : { 'H0':   3093.63534    , 'Hsol/R':  +469. , 'MW':  28.01  },
    'CO2'     : { 'H0':  17687.2176     , 'Hsol/R': -1010. , 'MW':  44.01  },
    'VinAce'  : { 'H0':  19089.7895     , 'Hsol/R': -2500. , 'MW':  86.09  },
    'C2H4'    : { 'H0':   6793.60087    , 'Hsol/R':  -627. , 'MW':  28.05  },
    'iC4'     : { 'H0':  10496.8464     , 'Hsol/R': -1627. , 'MW':  58.12  },
    'Benzene' : { 'H0':  20283.7286     , 'Hsol/R': -2920. , 'MW':  78.11  },
    'Toluene' : { 'H0':  32950.3153     , 'Hsol/R': -3461. , 'MW':  92.14  },
}

compounds_v2 = {
    'H2'      : { 'MW':   2.016 , 'dHsol/R': +472.8315525 , 'lnH0': 8.124149532 },
    'C1'      : { 'MW':  16.04  , 'dHsol/R': +2.735977354 , 'lnH0': 8.731344842 },
    'C2'      : { 'MW':  30.07  , 'dHsol/R': -600.1989700 , 'lnH0': 8.778734219 },
    'C3'      : { 'MW':  44.09  , 'dHsol/R': -1090.128588 , 'lnH0': 8.719472698 },
    'C4'      : { 'MW':  58.12  , 'dHsol/R': -1818.230692 , 'lnH0': 9.421184710 },
    'C5'      : { 'MW':  72.15  , 'dHsol/R': -2209.744754 , 'lnH0': 9.760611249 },
    'C6'      : { 'MW':  86.17  , 'dHsol/R': -2805.190625 , 'lnH0': 10.09889626 },
    'C7'      : { 'MW': 100.20  , 'dHsol/R': -3282.775276 , 'lnH0': 10.41519594 },
    'C8'      : { 'MW': 114.22  , 'dHsol/R': -3728.364057 , 'lnH0': 10.85028070 },
    'N2'      : { 'MW':  28.01  , 'dHsol/R': +469.3227179 , 'lnH0': 8.037102165 },
    'CO2'     : { 'MW':  44.01  , 'dHsol/R': -1010.182899 , 'lnH0': 9.780597487 },
    'VinAce'  : { 'MW':  86.09  , 'dHsol/R': -2500.220269 , 'lnH0': 9.856908889 },
    'C2H4'    : { 'MW':  28.05  , 'dHsol/R': -626.6910178 , 'lnH0': 8.823736400 },
    'iC4'     : { 'MW':  58.12  , 'dHsol/R': -1627.016052 , 'lnH0': 9.258830151 },
    'Benzene' : { 'MW':  78.11  , 'dHsol/R': -2919.958665 , 'lnH0': 9.917574296 },
    'Toluene' : { 'MW':  92.14  , 'dHsol/R': -3461.185768 , 'lnH0': 10.40275611 },
}


class Calculator():

    def __init__( self, temp=573.15, pressure=1.0, volume=1.0, mass=10.0, version=2 ):

        self.temp     = temp
        self.pressure = pressure
        self.volume   = volume
        self.mass     = mass

        self.version  = version

        self.names  = [ key for key in compounds.keys() ]
        if self.version == 1:
            self.MW     = numpy.array([ value[ 'MW'     ] for value in compounds.values() ])
            self.Hsol_R = numpy.array([ value[ 'Hsol/R' ] for value in compounds.values() ])
            self.H0     = numpy.array([ value[ 'H0'     ] for value in compounds.values() ])
        else:
            self.MW     = numpy.array([ value[ 'MW'      ] for value in compounds_v2.values() ])
            self.dHsolR = numpy.array([ value[ 'dHsol/R' ] for value in compounds_v2.values() ])
            self.lnH0   = numpy.array([ value[ 'lnH0'    ] for value in compounds_v2.values() ])
        self.Hv     = self.get_henrysconst()

        self.R = 0.082057366080960 # L atm / mol K

        return

    def get_henrysconst( self, temp=None ):
        if temp is None:
            temp = self.temp
        if self.version == 1:
            return self.H0 * numpy.exp( self.Hsol_R * ( 1.0/temp - 1.0/573.0 ) )
        else:
            return numpy.exp( self.lnH0 + self.dHsolR / temp )

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

    def get_gaspart( self, temp=None, pressure=None, mass=None, volume=None ):

        if temp is None:
            temp = self.temp
            Hv   = self.Hv
        else:
            temp = temp
            Hv   = self.get_henrysconst(temp)

        if pressure is None:
            pressure = self.pressure
        if mass is None:
            mass = self.mass
        if volume is None:
            volume = self.volume

        alpha = 1.0 / ( 1.0 + ( mass * self.R * temp ) / ( self.MW * self.Hv * volume ) )

        return alpha

    def get_liquidpart( self, temp=None, pressure=None, mass=None, volume=None ):

        if temp is None:
            temp = self.temp
            Hv   = self.Hv
        else:
            temp = temp
            Hv   = self.get_henrysconst(temp)

        if pressure is None:
            pressure = self.pressure
        if mass is None:
            mass = self.mass
        if volume is None:
            volume = self.volume

        alpha1m = 1.0 / ( 1.0 + ( self.MW * self.Hv * volume ) / ( mass * self.R * temp ) )

        return alpha1m

