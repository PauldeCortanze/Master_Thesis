import numpy as np
import openmdao.api as om

from hydesign.wind.wind import get_rotor_d


class HPPMappingComp(om.ExplicitComponent):
    """
    Component to map high-level design variables to low-level OpenMDAO model inputs.
    """

    @property
    def input_keys(self):
        return ['clearance', 'sp', 'p_rated', 'Nwt', 'wind_MW_per_km2', 'solar_MW', 'surface_tilt', 'surface_azimuth', 'DC_AC_ratio', 'b_P', 'b_E_h', 'cost_of_battery_P_fluct_in_peak_price_ratio']

    @property
    def output_keys(self):
        return ['hh', 'd', 'Awpp', 'b_E', 'p_rated_out', 'Nwt_out', 'solar_MW_out', 'surface_tilt_out', 'surface_azimuth_out', 'DC_AC_ratio_out', 'b_P_out', 'cost_of_battery_P_fluct_in_peak_price_ratio_out']

    def setup(self):
        # Inputs: high-level design variables
        self.add_input('clearance', units='m',
                       desc='Distance from ground to blade tip')
        self.add_input('sp', units='W/m**2', desc='Specific power')
        self.add_input('p_rated', units='MW',
                       desc='Rated power per turbine')
        self.add_input('Nwt', desc='Number of wind turbines')
        self.add_input('wind_MW_per_km2',
                       units='MW/km**2', desc='Wind power density')
        self.add_input('solar_MW',
                       units='MW', desc='Solar capacity')
        self.add_input('surface_tilt', units='deg', desc='Solar panel tilt')
        self.add_input('surface_azimuth', units='deg',
                       desc='Solar panel azimuth')
        self.add_input('DC_AC_ratio', desc='DC to AC ratio')
        self.add_input('b_P', units='MW', desc='Battery power')
        self.add_input('b_E_h', units='h',
                       desc='Battery energy hours')
        self.add_input('cost_of_battery_P_fluct_in_peak_price_ratio',
                       desc='Battery cost ratio')

        # Outputs: low-level model inputs (computed or pass-through)
        self.add_output('hh', val=80.0, units='m', desc='Hub height')
        self.add_output('d', val=100.0, units='m', desc='Rotor diameter')
        self.add_output('Awpp', val=20.0, units='km**2', desc='Wind farm area')
        self.add_output('b_E', val=300.0, desc='Battery energy')
        # Pass-through outputs
        self.add_output('p_rated_out', val=5.0, units='MW')
        self.add_output('Nwt_out', val=100)
        self.add_output('solar_MW_out', val=100.0, units='MW')
        self.add_output('surface_tilt_out', val=25.0, units='deg')
        self.add_output('surface_azimuth_out', val=180.0, units='deg')
        self.add_output('DC_AC_ratio_out', val=1.0)
        self.add_output('b_P_out', val=50.0, units='MW')
        self.add_output(
            'cost_of_battery_P_fluct_in_peak_price_ratio_out', val=10.0)

    def compute(self, inputs, outputs):
        # Compute derived values
        clearance = inputs['clearance']
        sp = inputs['sp']
        p_rated = inputs['p_rated']
        Nwt = inputs['Nwt']
        wind_MW_per_km2 = inputs['wind_MW_per_km2']
        b_E_h = inputs['b_E_h']
        b_P = inputs['b_P']

        # Rotor diameter
        d = get_rotor_d(p_rated * 1e6 / sp)
        outputs['d'] = d

        # Hub height
        outputs['hh'] = d / 2 + clearance

        # Wind farm area
        wind_MW = Nwt * p_rated
        outputs['Awpp'] = wind_MW / wind_MW_per_km2

        # Battery energy
        outputs['b_E'] = b_E_h * b_P

        # Pass-through
        outputs['p_rated_out'] = p_rated
        outputs['Nwt_out'] = Nwt
        outputs['solar_MW_out'] = inputs['solar_MW']
        outputs['surface_tilt_out'] = inputs['surface_tilt']
        outputs['surface_azimuth_out'] = inputs['surface_azimuth']
        outputs['DC_AC_ratio_out'] = inputs['DC_AC_ratio']
        outputs['b_P_out'] = b_P
        outputs['cost_of_battery_P_fluct_in_peak_price_ratio_out'] = inputs['cost_of_battery_P_fluct_in_peak_price_ratio']
