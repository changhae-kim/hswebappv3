from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import InputRequired
import numpy

app = Flask(__name__)
app.config['SECRET_KEY'] = '2389dh01'

compound_list = ['H2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C2H4', 'iC4']
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

wtfrac = {}

class Calc():

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

class Calculator(FlaskForm):
    H2 = FloatField('H2', validators=[InputRequired()])
    C1 = FloatField('C1', validators=[InputRequired()])
    C2 = FloatField('C2', validators=[InputRequired()])
    C2H4 = FloatField('C2H4', validators=[InputRequired()])
    C3 = FloatField('C3', validators=[InputRequired()])
    C4 = FloatField('C4', validators=[InputRequired()])
    iC4 = FloatField('iC4', validators=[InputRequired()])
    C5 = FloatField('C5', validators=[InputRequired()])
    C6 = FloatField('C6', validators=[InputRequired()])
    C7 = FloatField('C7', validators=[InputRequired()])
    C8 = FloatField('C8', validators=[InputRequired()])

    TField = FloatField('T', validators=[InputRequired()])
    PtField = FloatField('Pt', validators=[InputRequired()])

    run = SubmitField('Run')
    reset = SubmitField('Reset')

@app.route('/')
def index():
    calculator = Calculator()

    return render_template('index.html', calculator=calculator, wtfrac=wtfrac, compound_list=compound_list)

@app.route('/calculate', methods=['POST'])
def calculate():
    calculator = Calculator()

    T = calculator.TField.data
    Pt = calculator.PtField.data # Pt for total pressure

    calc = Calc(temp=T, pressure=Pt)
    y = []

    if calculator.run.data:
        # do wt frac calculation
        for compound in compound_list:
            y.append(calculator[compound].data) # Array of each gas-phase mole frac.

        w = calc.get_liquidphase(y) # Array of liquid-phase mass frac.
        for i in range(len(compound_list)):
            wtfrac.update({compound_list[i]: w[i]}) # Dict of compound : liq. mass frac.

    return render_template('index.html', calculator=calculator, wtfrac=wtfrac, compound_list=compound_list)

@app.route('/reset', methods=['POST'])
def reset():
    calculator = Calculator()

    if calculator.reset.data:
        wtfrac.clear()
        return render_template('index.html', calculator=calculator, wtfrac=wtfrac, compound_list=compound_list)

if __name__ == '__main__':
    app.run(debug=True)