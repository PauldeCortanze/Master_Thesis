# =============================================================================
#  hpp_assembly_NSGA2_Parallel.py
#  Multi-objective HPP assembly: NPV/CAPEX (economic) + Annual Carbon Offset
#  (environmental).
#
#  How the two objectives interact
#  --------------------------------
#  The ALPSO outer loop explores the Pareto front by varying `emission_weight`
#  [€/tCO2eq].  At each evaluation the EMS LP sees an effective electricity
#  price of  price[t] + emission_weight/1000 * mdf[t]  [€/MWh], which biases
#  dispatch toward high-MDF (high-carbon-displacement) hours.  The outer loop
#  then maximises both NPV_over_CAPEX and annual_carbon_offset simultaneously.
#
#  Setting emission_weight=0 reproduces the original economic-only behaviour.
# =============================================================================

import datetime
import os
import traceback

import numpy as np
import openmdao.api as om
import pandas as pd
import xarray as xr
import yaml

os.environ['OPENMDAO_REPORTS'] = '0'

from hydesign.assembly.hpp_mapping_comp import HPPMappingComp
from hydesign.battery_degradation import (
    battery_degradation_comp as battery_degradation,
    battery_loss_in_capacity_due_to_temp_comp as battery_loss_in_capacity_due_to_temp,
)
from hydesign.costs.costs import (
    battery_cost_comp as battery_cost,
    pvp_cost_comp as pvp_cost,
    shared_cost_comp as shared_cost,
    wpp_cost_comp as wpp_cost,
)
from hydesign.ems.ems_new import (
    ems_comp as ems,
    ems_long_term_operation_comp as ems_long_term_operation,
)
from hydesign.finance.finance import finance_comp as finance
from hydesign.look_up_tables import lut_filepath
from hydesign.pv.pv import (
    pvp_comp as pvp,
    pvp_with_degradation_comp as pvp_with_degradation,
)
from hydesign.reliability import (
    battery_with_reliability_comp as battery_with_reliability,
    pvp_with_reliability_comp as pvp_with_reliability,
    wpp_with_reliability_comp as wpp_with_reliability,
)
from hydesign.weather.weather import ABL_comp as ABL
from hydesign.weather.weather import extract_weather_for_HPP, select_years
from hydesign.wind.wind import (
    genericWake_surrogate_comp as genericWake_surrogate,
    genericWT_surrogate_comp as genericWT_surrogate,
    get_rotor_d,
    wpp_comp as wpp,
    wpp_with_degradation_comp as wpp_with_degradation,
)
from hydesign.vepso.vepso_driver import VESPODriver


# =============================================================================
#  Carbon Offset Component
# =============================================================================

class CarbonOffsetComp(om.ExplicitComponent):
    """
    Computes the mean annual carbon offset [tCO2eq/yr] by multiplying the
    lifetime HPP power export series by the (tiled) marginal displacement
    factor (MDF) at each corresponding hour.

    Both inputs have shape (life_h,) where life_h = life_y * 8760, which
    matches the existing promoted 'hpp_t' output of ems_comp.

    Inputs
    ------
    hpp_t_co : np.ndarray, shape (life_h,)
        Net power exported to the grid [MW] — connected explicitly from
        the promoted 'hpp_t' (lifetime series from ems_comp).
    mdf_t_co : np.ndarray, shape (life_h,)
        Marginal displacement factor [kgCO2eq/MWh] tiled over the lifetime.
        (Uses a distinct promoted name 'mdf_t_co' to avoid colliding with the
        N_time-shaped 'mdf_t' input of ems_comp.)

    Outputs
    -------
    annual_carbon_offset : float
        Mean annual carbon offset [tCO2eq/yr] = sum(hpp_t_co * mdf_t_co) / life_y / 1e3
    """

    def initialize(self):
        self.options.declare('life_h', types=int,
                             desc='Total lifetime hours = life_y * 8760')
        self.options.declare('life_y', types=(int, float),
                             desc='Plant lifetime in years')

    def setup(self):
        life_h = self.options['life_h']
        self.add_input('hpp_t_co',  shape=(life_h,), units='MW', val=0.0)
        self.add_input('mdf_t_co',  shape=(life_h,),              val=0.0)
        self.add_output('annual_carbon_offset', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        hpp_t  = inputs['hpp_t_co']      # MW  (= MWh per hour)
        mdf_t  = inputs['mdf_t_co']      # kgCO2eq / MWh
        life_y = self.options['life_y']
        # kgCO2eq summed over lifetime, divided by years → kg/yr → t/yr
        outputs['annual_carbon_offset'] = (hpp_t * mdf_t).sum() / life_y / 1e3


# =============================================================================
#  MDF loading helper
# =============================================================================

def load_mdf_aligned(mdf_fn, weather_index, weeks_per_season_per_year,
                     seed, work_dir, verbose=True):
    """
    Load the MDF CSV, align it to the weather time index using a
    (month, day, hour) lookup over the available years (2023-2024), apply
    the same week-sampling used for the weather, and return a 1-D numpy
    array of length N_time (the final simulation length after sampling).

    Parameters
    ----------
    mdf_fn : str
        Path to the MDF CSV.  Must have a datetime index in DD/MM/YYYY HH:MM
        format and one numeric column (e.g. 'MDFwind_kgCO2eperMWh').
    weather_index : pd.DatetimeIndex
        Full (pre-sampling) weather time index.
    weeks_per_season_per_year : int or None
    seed : int
    work_dir : str
    verbose : bool

    Returns
    -------
    mdf_aligned : np.ndarray, shape (N_time,)
        MDF values in kgCO2eq/MWh aligned to the final simulation timesteps.
    """
    # ------------------------------------------------------------------
    # 1. Load CSV — European date format (day first)
    # ------------------------------------------------------------------
    df_mdf = pd.read_csv(mdf_fn, index_col=0, parse_dates=True, dayfirst=True)

    # Pick first numeric column
    mdf_col = df_mdf.select_dtypes(include=[np.number]).columns[0]
    if verbose:
        n = len(df_mdf)
        t0 = df_mdf.index[0].strftime('%d/%m/%Y %H:%M')
        t1 = df_mdf.index[-1].strftime('%d/%m/%Y %H:%M')
        print(f"Loaded MDF data from '{mdf_fn}' with {n} timesteps, "
              f"from {t0} to {t1}")
        print(f"MDF column used: '{mdf_col}'")

    mdf_series = df_mdf[mdf_col].copy()
    mdf_series.index = pd.to_datetime(mdf_series.index)

    # ------------------------------------------------------------------
    # 2. Build (month, day, hour) → mean MDF lookup across available years.
    #    This allows the MDF to be mapped onto any weather year.
    # ------------------------------------------------------------------
    mdf_lookup = (
        mdf_series
        .groupby([mdf_series.index.month,
                  mdf_series.index.day,
                  mdf_series.index.hour])
        .mean()
    )
    mdf_lookup.index.names = ['month', 'day', 'hour']

    def _get_mdf(dt):
        try:
            return mdf_lookup.loc[(dt.month, dt.day, dt.hour)]
        except KeyError:
            # Feb 29 in a year not in MDF → use Feb 28 value
            return mdf_lookup.loc[(dt.month, 28, dt.hour)]

    mdf_full = pd.Series(
        [_get_mdf(dt) for dt in weather_index],
        index=weather_index,
        name='mdf_t',
    )

    # ------------------------------------------------------------------
    # 3. Apply the same week-sampling as the weather (if used)
    # ------------------------------------------------------------------
    if weeks_per_season_per_year is not None:
        mdf_sampled = select_years(
            mdf_full.to_frame(),
            seed=seed,
            weeks_per_season_per_year=weeks_per_season_per_year,
        )
        mdf_aligned = mdf_sampled['mdf_t'].values
    else:
        mdf_aligned = mdf_full.values

    if verbose:
        print(f"MDF aligned: {len(mdf_aligned)} timesteps, "
              f"mean={mdf_aligned.mean():.1f} kgCO2eq/MWh, "
              f"min={mdf_aligned.min():.1f}, max={mdf_aligned.max():.1f}")

    return mdf_aligned


# =============================================================================
#  hpp_base
# =============================================================================

class hpp_base:
    def __init__(self, sim_pars_fn, variables, defaults={}, **kwargs):
        self.sim_pars_fn = sim_pars_fn
        self.variables = variables

        # Build sim_pars: defaults → yml file → kwargs
        sim_pars = self.get_defaults()
        sim_pars.update(defaults)
        with open(sim_pars_fn) as file:
            sim_pars.update(yaml.load(file, Loader=yaml.FullLoader))
        sim_pars.update(kwargs)
        self.check_inputs(sim_pars)

        # Resolve wind turbine look-up table paths
        genWT_fn   = os.path.join(sim_pars["gen_lut_dir"], sim_pars["genWT_fn"])
        genWake_fn = os.path.join(sim_pars["gen_lut_dir"], sim_pars["genWake_fn"])
        sim_pars["genWT_fn"]   = genWT_fn
        sim_pars["genWake_fn"] = genWake_fn

        work_dir   = sim_pars["work_dir"]
        altitude   = sim_pars["altitude"]
        latitude   = sim_pars["latitude"]
        longitude  = sim_pars["longitude"]
        verbose    = sim_pars["verbose"]
        input_ts_fn               = sim_pars["input_ts_fn"]
        price_fn                  = sim_pars["price_fn"]
        name                      = sim_pars["name"]
        ppa_price                 = sim_pars["ppa_price"]
        weeks_per_season_per_year = sim_pars["weeks_per_season_per_year"]
        seed                      = sim_pars["seed"]
        max_num_batteries_allowed = sim_pars["max_num_batteries_allowed"]

        work_dir = mkdir(work_dir)

        # Altitude from elevation map if not provided
        if altitude is None:
            elevation_fn = sim_pars["elevation_fn"]
            elevation_ds = xr.open_dataset(elevation_fn, engine="h5netcdf")
            altitude = (
                elevation_ds["elev"]
                .interp(latitude=latitude, longitude=longitude,
                        kwargs={"fill_value": 0.0})
                .values
            )

        if verbose:
            print("\nFixed parameters on the site")
            print("-------------------------------")
            print("longitude =", longitude)
            print("latitude  =", latitude)
            print("altitude  =", altitude)

        # Life time
        if "life_y" in sim_pars:
            life_y = sim_pars["life_y"]
        elif "N_life" in sim_pars:
            life_y = sim_pars["N_life"]

        # Wind degradation curves
        wind_deg           = sim_pars.get("wind_deg",            [0, 0])
        wind_deg_yr        = sim_pars.get("wind_deg_yr",         [0, 25])
        share_WT_deg_types = sim_pars.get("share_WT_deg_types",  0.5)

        # Year range for ERA5 extraction
        year_start = sim_pars.get("year_start", sim_pars.get("year"))
        year_end   = sim_pars.get("year_end",   sim_pars.get("year"))

        # ------------------------------------------------------------------
        # Weather time series
        # ------------------------------------------------------------------
        if input_ts_fn is None:
            era5_zarr      = sim_pars["era5_zarr"]
            ratio_gwa_era5 = sim_pars["ratio_gwa_era5"]
            era5_ghi_zarr  = sim_pars["era5_ghi_zarr"]

            weather = extract_weather_for_HPP(
                longitude=longitude, latitude=latitude, altitude=altitude,
                era5_zarr=era5_zarr, ratio_gwa_era5=ratio_gwa_era5,
                era5_ghi_zarr=era5_ghi_zarr,
                year_start=year_start, year_end=year_end,
            )
            if isinstance(price_fn, str):
                price_df = pd.read_csv(price_fn, index_col=0, parse_dates=True)
            else:
                price_df = price_fn
            try:
                weather["Price"] = price_df.loc[weather.index].bfill()
            except Exception:
                raise ValueError("Price timeseries does not match the weather index.")

            N_time = len(weather)
            if np.mod(N_time, 365 * 24) != 0:
                N_sel  = N_time - np.mod(N_time, 365 * 24)
                weather = weather.iloc[:N_sel]

            input_ts_fn = f"{work_dir}input_ts{name}.csv"
            print(f"\ninput_ts_fn extracted and stored in {input_ts_fn}")
            weather.to_csv(input_ts_fn)
            N_time = len(weather)

        else:
            weather = pd.read_csv(input_ts_fn, index_col=0, parse_dates=True)
            N_time  = len(weather)

        # Ensure complete years
        if np.mod(N_time, 365 * 24) != 0:
            N_sel   = N_time - np.mod(N_time, 365 * 24)
            weather = weather.iloc[:N_sel]
            input_ts_fn = f"{work_dir}input_ts_modified.csv"
            print("\ninput_ts_fn length is not a complete number of years "
                  "(hyDesign handles years as 365 days).")
            print(f"The file has been modified and stored in {input_ts_fn}")
            weather.to_csv(input_ts_fn)
            N_time = len(weather)

        # Price array
        if ppa_price is None:
            price = weather["Price"].values
        else:
            price = ppa_price * np.ones_like(weather["Price"])

        # Keep full index before sampling (needed for MDF alignment)
        weather_index_full = weather.index

        # Week sampling
        if weeks_per_season_per_year is not None:
            weather = select_years(
                weather,
                seed=seed,
                weeks_per_season_per_year=weeks_per_season_per_year,
            )
            N_time = len(weather)
            input_ts_fn = f"{work_dir}input_ts_sel.csv"
            print(f"\n\nSelected input time series based on "
                  f"{weeks_per_season_per_year} weeks per season stored in "
                  f"{input_ts_fn}")
            weather.to_csv(input_ts_fn)

        # Number of wind-speed points in look-up tables
        with xr.open_dataset(genWT_fn, engine="h5netcdf") as ds:
            self.N_ws = len(ds.ws.values)

        sim_pars["time_str"] = datetime.datetime.now().strftime("%y_%m_%H_%M_%S")

        if "H2_demand_fn" in sim_pars:
            H2_demand_fn = sim_pars["H2_demand_fn"]
            if isinstance(H2_demand_fn, str):
                H2_demand_data = pd.read_csv(
                    H2_demand_fn, index_col=0, parse_dates=True
                ).loc[weather.index, :]
                H2_demand = H2_demand_data["H2_demand"]
            elif H2_demand_fn is None:
                H2_demand = None
            elif np.size(H2_demand_fn) == 1:
                H2_demand = H2_demand_fn * np.ones(N_time)
            else:
                H2_demand = H2_demand_fn
            sim_pars["H2_demand"] = H2_demand

        # ------------------------------------------------------------------
        # MDF loading (carbon offset / emission-aware dispatch signal)
        # ------------------------------------------------------------------
        mdf_t  = None
        mdf_fn = sim_pars.get("mdf_fn", None)

        if mdf_fn is not None:
            try:
                mdf_t = load_mdf_aligned(
                    mdf_fn=mdf_fn,
                    weather_index=weather_index_full,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                    seed=seed,
                    work_dir=work_dir,
                    verbose=verbose,
                )
                if len(mdf_t) != N_time:
                    raise ValueError(
                        f"MDF aligned length ({len(mdf_t)}) != N_time ({N_time}). "
                        f"Check weeks_per_season_per_year and seed consistency."
                    )
            except Exception as e:
                print(f"WARNING: Could not load MDF file '{mdf_fn}': {e}")
                print("         Carbon offset objective will be disabled.")
                mdf_t = None
        else:
            if verbose:
                print("\nNo 'mdf_fn' in sim_pars — carbon offset objective disabled.")

        self.mdf_t = mdf_t   # None → single-objective mode

        # ------------------------------------------------------------------
        # Store attributes
        # ------------------------------------------------------------------
        self.weather              = weather
        self.N_time               = N_time
        self.wind_deg             = wind_deg
        self.wind_deg_yr          = wind_deg_yr
        self.share_WT_deg_types   = share_WT_deg_types
        self.price                = price
        self.wpp_efficiency       = sim_pars["wpp_efficiency"]
        self.max_num_batteries_allowed = max_num_batteries_allowed
        self.input_ts_fn          = input_ts_fn
        self.longitude            = longitude
        self.latitude             = latitude
        self.altitude             = altitude
        self.life_y               = life_y

        sim_pars["N_ws"]               = self.N_ws
        sim_pars["N_time"]             = N_time
        sim_pars["wind_deg"]           = wind_deg
        sim_pars["wind_deg_yr"]        = wind_deg_yr
        sim_pars["share_WT_deg_types"] = share_WT_deg_types
        sim_pars["price"]              = price
        sim_pars["input_ts_fn"]        = input_ts_fn
        sim_pars["life_y"]             = life_y

        self.sim_pars = sim_pars

    # ------------------------------------------------------------------
    def get_defaults(self):
        return dict(
            work_dir="./",
            max_num_batteries_allowed=3,
            ems_type="cplex",
            weeks_per_season_per_year=None,
            seed=0,
            input_ts_fn=None,
            price_fn=None,
            gen_lut_dir=lut_filepath,
            genWT_fn="genWT_v3.nc",
            genWake_fn="genWake_v3.nc",
            verbose=True,
            name="",
            ppa_price=None,
            reliability_ts_battery=None,
            reliability_ts_trans=None,
            reliability_ts_wind=None,
            reliability_ts_pv=None,
            save_finance_ts=False,
            mdf_fn=None,          # path to MDF CSV — None disables carbon offset
            emission_weight=0.0,  # €/tCO2eq — 0 = economic-only EMS dispatch
        )

    def check_inputs(self, sim_pars):
        for var in ["altitude", "latitude", "longitude"]:
            if var not in sim_pars:
                raise ValueError(
                    f"Variable '{var}' must be provided in the yml file or at "
                    f"instantiation."
                )
            if sim_pars[var] is None:
                raise ValueError(f"Variable '{var}' cannot be None.")

    def check_outputs(self, x_opt, outs):
        if x_opt is None:
            if hasattr(self, "inputs"):
                x_opt = self.inputs
            else:
                raise ValueError("No design inputs available.")
        if outs is None:
            if hasattr(self, "outputs"):
                outs = self.outputs
            else:
                raise ValueError("No outputs available.")
        return x_opt, outs

    def print_design(self, x_opt=None, outs=None):
        x_opt, outs = self.check_outputs(x_opt, outs)
        print("\nDesign:")
        print("---------------")
        for i_v, var in enumerate(self.list_vars):
            v = x_opt[i_v]
            if np.size(v) == 1:
                print(f"{var}: {v:.3f}")
            else:
                print(f"{var}: mean={np.mean(v):.3f}, std={np.std(v):.3f}, "
                      f"min={np.min(v):.3f}, max={np.max(v):.3f}")
        print()
        for i_v, var in enumerate(self.list_out_vars):
            try:
                print(f"{var}: {outs[i_v]:.3f}")
            except Exception:
                print(f"{var}: {outs[i_v]}")
        print()

    def evaluation_in_df(self, x_opt=None, outs=None):
        x_opt, outs = self.check_outputs(x_opt, outs)
        design_df = pd.DataFrame(
            columns=["longitude", "latitude", "altitude"]
                    + self.list_vars + self.list_out_vars,
            index=range(1),
        )
        design_df.iloc[0] = (
            [self.longitude, self.latitude, self.altitude]
            + list(x_opt) + list(outs)
        )
        return design_df

    def evaluation_in_csv(self, name_file, design_df=None, x_opt=None, outs=None):
        if design_df is None:
            design_df = self.evaluation_in_df(x_opt, outs)
        design_df.to_csv(f"{name_file}.csv")

    @staticmethod
    def get_prob(comps):
        """
        Build an OpenMDAO Problem from a list of component tuples.

        Each tuple is either:
          (name, component)               → promotes=['*']
          (name, component, key_map_dict) → uses input_keys / output_keys
                                            from hyDesign ComponentWrapper
        """
        prob = om.Problem()

        for c in comps:
            if len(c) == 2:
                prob.model.add_subsystem(c[0], c[1], promotes=["*"])
            elif len(c) == 3:
                key_map_dict = c[2]
                input_list = [
                    (k, key_map_dict.get(k, k)) for k in c[1].input_keys
                ]
                prob.model.add_subsystem(
                    c[0], c[1],
                    promotes_inputs=input_list,
                    promotes_outputs=list(c[1].output_keys),
                )

        prob.model.set_input_defaults('surface_tilt',    val=28.125, units='deg')
        prob.model.set_input_defaults('surface_azimuth', val=191.25, units='deg')
        return prob

    def load_simulation_parameters(self, sim_pars_fn, defaults, kwargs):
        sim_pars = self.get_defaults()
        sim_pars.update(defaults)
        with open(sim_pars_fn) as file:
            sim_pars.update(yaml.load(file, Loader=yaml.FullLoader))
        sim_pars.update(kwargs)
        self.check_inputs(sim_pars)
        return sim_pars


# =============================================================================
#  hpp_model
# =============================================================================

class hpp_model(hpp_base):
    """HPP design evaluator with optional multi-objective carbon offset.

    The `emission_weight` parameter [€/tCO2eq] is the scalarisation knob that
    controls the EMS dispatch trade-off:
      - emission_weight = 0   → purely economic dispatch (original behaviour)
      - emission_weight > 0   → EMS adds a carbon bonus of
                                 emission_weight/1000 * mdf[t] [€/MWh]
                                 to the effective electricity price, biasing
                                 dispatch toward high-carbon-displacement hours.
    The ALPSO outer loop varies emission_weight across runs to trace the
    NPV_over_CAPEX  vs  annual_carbon_offset  Pareto front.
    """

    def __init__(self, sim_pars_fn, **kwargs):
        hpp_base.__init__(self, sim_pars_fn=sim_pars_fn, **kwargs)

        N_time             = self.N_time
        N_ws               = self.N_ws
        wpp_efficiency     = self.wpp_efficiency
        sim_pars           = self.sim_pars
        life_y             = self.life_y
        wind_deg_yr        = self.wind_deg_yr
        wind_deg           = self.wind_deg
        share_WT_deg_types = self.share_WT_deg_types
        price              = self.price
        mdf_t              = self.mdf_t   # None if mdf_fn was not provided

        input_ts_fn               = sim_pars["input_ts_fn"]
        genWT_fn                  = sim_pars["genWT_fn"]
        genWake_fn                = sim_pars["genWake_fn"]
        latitude                  = sim_pars["latitude"]
        longitude                 = sim_pars["longitude"]
        altitude                  = sim_pars["altitude"]
        weeks_per_season_per_year = sim_pars["weeks_per_season_per_year"]
        ems_type                  = sim_pars["ems_type"]
        max_num_batteries_allowed = sim_pars["max_num_batteries_allowed"]
        reliability_ts_battery    = sim_pars["reliability_ts_battery"]
        reliability_ts_trans      = sim_pars["reliability_ts_trans"]
        reliability_ts_wind       = sim_pars["reliability_ts_wind"]
        reliability_ts_pv         = sim_pars["reliability_ts_pv"]
        battery_price_reduction_per_year = sim_pars["battery_price_reduction_per_year"]

        # emission_weight from kwargs takes priority, then sim_pars default (0.0)
        self.emission_weight = float(kwargs.get(
            'emission_weight', sim_pars.get('emission_weight', 0.0)
        ))

        # Whether the carbon offset objective is active (requires MDF data)
        self.use_carbon_offset = mdf_t is not None

        # Lifetime hours — matches the shape of hpp_t from ems_comp
        life_h = int(life_y * 8760)
        self.life_h = life_h

        # ------------------------------------------------------------------
        # Component list (identical to original single-objective assembly)
        # ------------------------------------------------------------------
        comps = [
            (
                "mapping",
                HPPMappingComp(),
                {
                    "p_rated_out": "p_rated",
                    "Nwt_out": "Nwt",
                    "solar_MW_out": "solar_MW",
                    "surface_tilt_out": "surface_tilt",
                    "surface_azimuth_out": "surface_azimuth",
                    "DC_AC_ratio_out": "DC_AC_ratio",
                    "b_P_out": "b_P",
                    "cost_of_battery_P_fluct_in_peak_price_ratio_out":
                        "cost_of_battery_P_fluct_in_peak_price_ratio",
                },
            ),
            ("abl",         ABL(weather_fn=input_ts_fn, N_time=N_time)),
            ("genericWT",   genericWT_surrogate(genWT_fn=genWT_fn, N_ws=N_ws)),
            ("genericWake", genericWake_surrogate(genWake_fn=genWake_fn, N_ws=N_ws)),
            ("wpp",         wpp(N_time=N_time, N_ws=N_ws,
                                wpp_efficiency=wpp_efficiency)),
            (
                "pvp",
                pvp(
                    weather_fn=input_ts_fn, N_time=N_time,
                    latitude=latitude, longitude=longitude, altitude=altitude,
                    tracking=sim_pars["tracking"],
                ),
            ),
            (
                "ems",
                ems(
                    N_time=N_time,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                    life_y=life_y,
                    ems_type=ems_type,
                ),
            ),
            (
                "battery_degradation",
                battery_degradation(
                    weather_fn=input_ts_fn,
                    num_batteries=max_num_batteries_allowed,
                    life_y=life_y,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                ),
            ),
            (
                "battery_loss_in_capacity_due_to_temp",
                battery_loss_in_capacity_due_to_temp(
                    weather_fn=input_ts_fn,
                    life_y=life_y,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                ),
            ),
            (
                "wpp_with_degradation",
                wpp_with_degradation(
                    N_time=N_time, N_ws=N_ws,
                    wpp_efficiency=wpp_efficiency,
                    life_y=life_y,
                    wind_deg_yr=wind_deg_yr, wind_deg=wind_deg,
                    share_WT_deg_types=share_WT_deg_types,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                ),
            ),
            (
                "pvp_with_degradation",
                pvp_with_degradation(
                    life_y=life_y,
                    pv_deg_yr=sim_pars["pv_deg_yr"],
                    pv_deg=sim_pars["pv_deg"],
                ),
            ),
            (
                "battery_with_reliability",
                battery_with_reliability(
                    life_y=life_y,
                    reliability_ts_battery=reliability_ts_battery,
                    reliability_ts_trans=reliability_ts_trans,
                ),
            ),
            (
                "wpp_with_reliability",
                wpp_with_reliability(
                    life_y=life_y,
                    reliability_ts_wind=reliability_ts_wind,
                    reliability_ts_trans=reliability_ts_trans,
                ),
                {"wind_t": "wind_t_ext_deg"},
            ),
            (
                "pvp_with_reliability",
                pvp_with_reliability(
                    life_y=life_y,
                    reliability_ts_pv=reliability_ts_pv,
                    reliability_ts_trans=reliability_ts_trans,
                ),
                {"solar_t": "solar_t_ext_deg"},
            ),
            (
                "ems_long_term_operation",
                ems_long_term_operation(N_time=N_time, life_y=life_y),
                {
                    "SoH":             "SoH_all",
                    "wind_t_ext_deg":  "wind_t_rel",
                    "solar_t_ext_deg": "solar_t_rel",
                    "b_t":             "b_t_rel",
                },
            ),
            (
                "wpp_cost",
                wpp_cost(
                    wind_turbine_cost=sim_pars["wind_turbine_cost"],
                    wind_civil_works_cost=sim_pars["wind_civil_works_cost"],
                    wind_fixed_onm_cost=sim_pars["wind_fixed_onm_cost"],
                    wind_variable_onm_cost=sim_pars["wind_variable_onm_cost"],
                    d_ref=sim_pars["d_ref"],
                    hh_ref=sim_pars["hh_ref"],
                    p_rated_ref=sim_pars["p_rated_ref"],
                    N_time=N_time,
                ),
            ),
            (
                "pvp_cost",
                pvp_cost(
                    solar_PV_cost=sim_pars["solar_PV_cost"],
                    solar_hardware_installation_cost=sim_pars[
                        "solar_hardware_installation_cost"],
                    solar_inverter_cost=sim_pars["solar_inverter_cost"],
                    solar_fixed_onm_cost=sim_pars["solar_fixed_onm_cost"],
                ),
            ),
            (
                "battery_cost",
                battery_cost(
                    battery_energy_cost=sim_pars["battery_energy_cost"],
                    battery_power_cost=sim_pars["battery_power_cost"],
                    battery_BOP_installation_commissioning_cost=sim_pars[
                        "battery_BOP_installation_commissioning_cost"],
                    battery_control_system_cost=sim_pars["battery_control_system_cost"],
                    battery_energy_onm_cost=sim_pars["battery_energy_onm_cost"],
                    life_y=life_y,
                    battery_price_reduction_per_year=battery_price_reduction_per_year,
                ),
            ),
            (
                "shared_cost",
                shared_cost(
                    hpp_BOS_soft_cost=sim_pars["hpp_BOS_soft_cost"],
                    hpp_grid_connection_cost=sim_pars["hpp_grid_connection_cost"],
                    land_cost=sim_pars["land_cost"],
                ),
            ),
            (
                "finance",
                finance(
                    N_time=N_time,
                    depreciation_yr=sim_pars["depreciation_yr"],
                    depreciation=sim_pars["depreciation"],
                    inflation_yr=sim_pars["inflation_yr"],
                    inflation=sim_pars["inflation"],
                    ref_yr_inflation=sim_pars["ref_yr_inflation"],
                    phasing_yr=sim_pars["phasing_yr"],
                    phasing_CAPEX=sim_pars["phasing_CAPEX"],
                    life_y=life_y,
                ),
                {
                    "CAPEX_el": "CAPEX_sh",
                    "OPEX_el":  "OPEX_sh",
                    "penalty_t": "penalty_t_with_deg",
                },
            ),
        ]

        # ------------------------------------------------------------------
        # Build the OpenMDAO problem (without CarbonOffsetComp)
        # ------------------------------------------------------------------
        prob = hpp_base.get_prob(comps)

        # ------------------------------------------------------------------
        # Add CarbonOffsetComp outside get_prob to avoid .input_keys issue.
        #
        # Naming convention:
        #   'mdf_t'    (shape N_time) → EMS input  (promoted from ems_comp)
        #   'mdf_t_co' (shape life_h) → CarbonOffsetComp input (tiled version)
        #
        # The two arrays are different sizes so they must use distinct names.
        # CarbonOffsetComp.mdf_t_co is set directly via prob.set_val() below.
        # ------------------------------------------------------------------
        if self.use_carbon_offset:
            prob.model.add_subsystem(
                "carbon_offset",
                CarbonOffsetComp(life_h=life_h, life_y=life_y),
                promotes_inputs=["mdf_t_co"],         # lifetime-length MDF
                promotes_outputs=["annual_carbon_offset"],
            )
            # Connect promoted 'hpp_t' (lifetime, shape life_h) from ems_comp
            # to the carbon offset component's hpp_t_co input.
            prob.model.connect("hpp_t", "carbon_offset.hpp_t_co")

        self.prob = prob

        # ------------------------------------------------------------------
        # Setup optimization (adds objectives, driver, recorder, prob.setup)
        # ------------------------------------------------------------------
        self.setup_optimization(
            seed=kwargs.get('seed', 0),
            PopSize=kwargs.get('PopSize', 15),
            MaxGen=kwargs.get('MaxGen', 10),
            Pc=kwargs.get('Pc', 0.8),
            Pm=kwargs.get('Pm', 0.1),
            emission_weight=self.emission_weight,
        )

        # ------------------------------------------------------------------
        # Set fixed/constant values
        # ------------------------------------------------------------------
        prob.set_val("price_t",   price)
        prob.set_val("G_MW",      sim_pars["G_MW"])
        prob.set_val("battery_depth_of_discharge",
                     sim_pars["battery_depth_of_discharge"])
        prob.set_val("battery_charge_efficiency",
                     sim_pars["battery_charge_efficiency"])
        prob.set_val("peak_hr_quantile",      sim_pars["peak_hr_quantile"])
        prob.set_val("n_full_power_hours_expected_per_day_at_peak_price",
                     sim_pars["n_full_power_hours_expected_per_day_at_peak_price"])
        prob.set_val("min_LoH",               sim_pars["min_LoH"])
        prob.set_val("wind_WACC",             sim_pars["wind_WACC"])
        prob.set_val("solar_WACC",            sim_pars["solar_WACC"])
        prob.set_val("battery_WACC",          sim_pars["battery_WACC"])
        prob.set_val("tax_rate",              sim_pars["tax_rate"])
        prob.set_val("land_use_per_solar_MW", sim_pars["land_use_per_solar_MW"])

        # ------------------------------------------------------------------
        # Wire MDF and emission_weight into the EMS (ems_comp inputs)
        # ------------------------------------------------------------------
        # 'mdf_t' is the N_time-shaped input consumed by ems_comp.
        # 'emission_weight' is the carbon shadow-price scalar.
        # Both default to 0 (no carbon signal) when mdf_t is None.
        if self.use_carbon_offset:
            prob.set_val("mdf_t", mdf_t)                          # shape (N_time,)
            prob.set_val("emission_weight", self.emission_weight)  # scalar [€/tCO2eq]

            # 'mdf_t_co' is the life_h-shaped input of CarbonOffsetComp.
            # It is the per-simulation-year MDF tiled over the full lifetime.
            mdf_tiled = np.tile(mdf_t, int(life_y))   # (N_time,) → (life_h,)
            prob.set_val("mdf_t_co", mdf_tiled)
        else:
            # No MDF data: EMS sees zero carbon signal (original behaviour)
            prob.set_val("emission_weight", 0.0)

        # Initialise design variables to random values within bounds
        for var in self.variables.keys():
            if self.variables[var]['var_type'] == 'design':
                lower, upper = self.variables[var]['limits']
                prob.set_val(var, np.random.uniform(lower, upper))

        self.prob = prob

        # ------------------------------------------------------------------
        # Output variable names
        # ------------------------------------------------------------------
        self.list_out_vars = [
            "NPV_over_CAPEX",
            "NPV [MEuro]",
            "IRR",
            "LCOE [Euro/MWh]",
            "Revenues [MEuro]",
            "CAPEX [MEuro]",
            "OPEX [MEuro]",
            "Wind CAPEX [MEuro]",
            "Wind OPEX [MEuro]",
            "PV CAPEX [MEuro]",
            "PV OPEX [MEuro]",
            "Batt CAPEX [MEuro]",
            "Batt OPEX [MEuro]",
            "Shared CAPEX [MEuro]",
            "Shared OPEX [MEuro]",
            "penalty lifetime [MEuro]",
            "Mean Annual Electricity Sold [GWh]",
            "GUF",
            "grid [MW]",
            "wind [MW]",
            "solar [MW]",
            "Battery Energy [MWh]",
            "Battery Power [MW]",
            "Total curtailment [GWh]",
            "Total curtailment with deg [GWh]",
            "Awpp [km2]",
            "Apvp [km2]",
            "Plant area [km2]",
            "Rotor diam [m]",
            "Hub height [m]",
            "Number of batteries used in lifetime",
            "Break-even PPA price [Euro/MWh]",
            "Capacity factor wind [-]",
            "AEP [GWh]",
            "AEP with degradation [GWh]",
        ]
        if self.use_carbon_offset:
            self.list_out_vars.append("Annual carbon offset [tCO2eq/yr]")

        self.list_vars = [
            "clearance [m]",
            "sp [W/m2]",
            "p_rated [MW]",
            "Nwt",
            "wind_MW_per_km2 [MW/km2]",
            "solar_MW [MW]",
            "surface_tilt [deg]",
            "surface_azimuth [deg]",
            "DC_AC_ratio",
            "b_P [MW]",
            "b_E_h [h]",
            "cost_of_battery_P_fluct_in_peak_price_ratio",
        ]

    # ------------------------------------------------------------------
    def evaluate(
        self,
        clearance, sp, p_rated, Nwt, wind_MW_per_km2,
        solar_MW, surface_tilt, surface_azimuth, DC_AC_ratio,
        b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio,
    ):
        """Run the OpenMDAO model for a given design and return all outputs.

        Note: emission_weight and mdf_t are fixed inputs set during __init__
        and persist across evaluate() calls.  To change the carbon trade-off
        level, create a new hpp_model instance with a different emission_weight.
        """
        self.inputs = [
            clearance, sp, p_rated, Nwt, wind_MW_per_km2,
            solar_MW, surface_tilt, surface_azimuth, DC_AC_ratio,
            b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio,
        ]

        prob = self.prob

        d     = get_rotor_d(p_rated * 1e6 / sp)
        hh    = (d / 2) + clearance
        wind_MW = Nwt * p_rated
        Awpp  = wind_MW / wind_MW_per_km2
        b_E   = b_E_h * b_P

        prob.set_val("hh",    hh)
        prob.set_val("d",     d)
        prob.set_val("p_rated", p_rated)
        prob.set_val("Nwt",   Nwt)
        prob.set_val("Awpp",  Awpp)
        prob.set_val("surface_tilt",    surface_tilt)
        prob.set_val("surface_azimuth", surface_azimuth)
        prob.set_val("DC_AC_ratio",     DC_AC_ratio)
        prob.set_val("solar_MW",        solar_MW)
        prob.set_val("b_P",   b_P)
        prob.set_val("b_E",   b_E)
        prob.set_val("cost_of_battery_P_fluct_in_peak_price_ratio",
                     cost_of_battery_P_fluct_in_peak_price_ratio)

        prob.run_model()
        self.prob = prob

        cf_wind = (
            np.nan if Nwt == 0
            else prob.get_val("wpp_with_degradation.wind_t_ext_deg").mean()
                 / p_rated / Nwt
        )
        AEP = (prob["wind_t"].mean() + prob["solar_t"].mean()) * 1e-3 * 24 * 365
        AEP_deg = (
            (prob["wind_t_ext_deg"].mean() + prob["solar_t_ext_deg"].mean())
            * 1e-3 * 24 * 365
        )

        outputs = np.hstack([
            prob["NPV_over_CAPEX"],
            prob["NPV"] / 1e6,
            prob["IRR"],
            prob["LCOE"],
            prob["revenues"] / 1e6,
            prob["CAPEX"] / 1e6,
            prob["OPEX"] / 1e6,
            prob.get_val("finance.CAPEX_w") / 1e6,
            prob.get_val("finance.OPEX_w")  / 1e6,
            prob.get_val("finance.CAPEX_s") / 1e6,
            prob.get_val("finance.OPEX_s")  / 1e6,
            prob.get_val("finance.CAPEX_b") / 1e6,
            prob.get_val("finance.OPEX_b")  / 1e6,
            prob.get_val("finance.CAPEX_el") / 1e6,
            prob.get_val("finance.OPEX_el")  / 1e6,
            prob["penalty_lifetime"] / 1e6,
            prob["mean_AEP"] / 1e3,
            prob["mean_AEP"] / (self.sim_pars["G_MW"] * 365 * 24),
            self.sim_pars["G_MW"],
            wind_MW,
            solar_MW,
            b_E,
            b_P,
            prob["total_curtailment"]          / 1e3,
            prob["total_curtailment_with_deg"] / 1e3,
            Awpp,
            prob.get_val("shared_cost.Apvp"),
            max(Awpp, prob.get_val("shared_cost.Apvp")),
            d,
            hh,
            prob.get_val("battery_degradation.n_batteries") * (b_P > 0),
            prob["break_even_PPA_price"],
            cf_wind,
            AEP,
            AEP_deg,
        ])

        if self.use_carbon_offset:
            outputs = np.hstack([
                outputs,
                prob.get_val("annual_carbon_offset"),
            ])

        self.outputs = outputs
        return outputs

    # ------------------------------------------------------------------
    def setup_optimization(
        self,
        seed=0,
        PopSize=15,
        MaxGen=10,
        Pc=0.6,
        Pm=0.2,
        emission_weight=0.0,
    ):
        """Configure the OpenMDAO driver, objectives, and recorder.

        Objectives
        ----------
        NPV_over_CAPEX        : maximised (scaler=-1) — economic performance
        annual_carbon_offset  : maximised (scaler=-1) — only when MDF is active

        ALPSO handles multiple objectives natively by evolving a swarm toward
        the Pareto front.  The EMS's emission_weight biases each inner dispatch
        solve; the outer loop then evaluates both objectives and retains the
        non-dominated solutions.
        """
        model = self.prob.model
        prob  = self.prob

        # Design variables
        for var in self.variables.keys():
            if self.variables[var]['var_type'] == 'design':
                lower, upper = self.variables[var]['limits']
                model.add_design_var(var, lower=lower, upper=upper)

        # -------------------------------------------------------------------
        # Objectives — both are maximised (scaler=-1 inverts for minimisation)
        # -------------------------------------------------------------------

        if self.use_carbon_offset:
            model.add_objective('annual_carbon_offset', scaler=-1)
        model.add_objective('NPV_over_CAPEX',       scaler=-1)

        # -------------------------------------------------------------------
        # NSGA2 driver
        # -------------------------------------------------------------------
        # prob.driver = om.pyOptSparseDriver()
        # prob.driver.options['optimizer']          = 'NSGA2'
        # prob.driver.opt_settings['PopSize']       = PopSize
        # prob.driver.opt_settings['maxGen']        = MaxGen
        # prob.driver.opt_settings['pCross_real']   = Pc
        # prob.driver.opt_settings['pMut_real']     = Pm
        # # prob.driver.opt_settings['seed']          = seed

        # ── VESPODriver ───────────────────────────────────────────────────
        output_dir = 'optimization_results_multi-objective_VEPSO'
        os.makedirs(output_dir, exist_ok=True)
 
        history_csv = os.path.join(
            output_dir,
            f"swarm_history_VEPSO_seed{seed}_ew{emission_weight}.csv"
        )
 
        prob.driver = VESPODriver()
        prob.driver.options['swarm_size']   = PopSize   # re-use same argument
        prob.driver.options['max_iter']     = MaxGen    # re-use same argument
        prob.driver.options['omega']        = 0.7
        prob.driver.options['c1']           = 1.5
        prob.driver.options['c2']           = 1.5
        prob.driver.options['omega_damp']   = 0.99
        prob.driver.options['archive_size'] = 100
        prob.driver.options['seed']         = seed
        prob.driver.options['verbose']      = True
        prob.driver.options['history_csv']  = history_csv


        # ── SqliteRecorder (keeps save_results() working unchanged) ───────
        self.filepath = os.path.join(
            output_dir,
            f"results_VESPO_seed{seed}.sql"
        )
        self.filepath_csv = os.path.join(
            output_dir,
            f"history_VESPO_seed{seed}.csv"
        )
 
        recorder = om.SqliteRecorder(self.filepath)
        prob.driver.add_recorder(recorder)
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_desvars']    = True
 
        prob.setup()


        # Apply fixed variable values
        for var in self.variables.keys():
            if self.variables[var]['var_type'] == 'fixed':
                prob.set_val(var, self.variables[var]['value'])

    # ------------------------------------------------------------------
    def run_optimization(self):
        prob = self.prob
        prob.run_driver()
        prob.cleanup()

        self.save_results(self.filepath, self.filepath_csv)

        results = {}
        for var in self.variables.keys():
            results[var] = self.prob.get_val(var)
        results['NPV_over_CAPEX'] = float(self.prob.get_val('NPV_over_CAPEX')[0])
        if self.use_carbon_offset:
            results['annual_carbon_offset'] = float(
                self.prob.get_val('annual_carbon_offset')[0])

        print("Optimization completed")
        print("Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

        return results

    # ------------------------------------------------------------------
    def save_results(self, sqlite_path, csv_path):
        """Read the ALPSO recorder database and save a flat CSV history."""
        cr = om.CaseReader(sqlite_path)
        driver_cases = cr.get_cases('driver')

        data = []
        for i, entry in enumerate(driver_cases):
            case = cr.get_case(entry) if isinstance(entry, str) else entry
            row = {
                'iteration': i,
                'timestamp': case.timestamp,
                'success':   case.success,
            }
            for name, val in case.get_objectives().items():
                row[name] = val[0] if isinstance(val, (np.ndarray, list)) else val
            for name, val in case.get_design_vars().items():
                row[name] = val[0] if isinstance(val, (np.ndarray, list)) else val
            for name, val in case.get_constraints().items():
                row[name] = val[0] if isinstance(val, (np.ndarray, list)) else val
            data.append(row)

        if not data:
            print("Warning: No cases found in the recorder.")
            return

        df = pd.DataFrame(data)
        df['time_sec'] = df['timestamp'] - df['timestamp'].iloc[0]
        df.to_csv(csv_path, index=False)
        print(f"Successfully saved {len(df)} iterations to {csv_path}")


# =============================================================================
#  Auxiliary functions
# =============================================================================

def mkdir(dir_):
    if str(dir_).startswith("~"):
        dir_ = str(dir_).replace("~", os.path.expanduser("~"))
    try:
        os.stat(dir_)
    except Exception:
        try:
            os.mkdir(dir_)
        except Exception:
            pass
    return dir_


def run_optimization_seed(args):
    """
    Worker function for parallel multi-seed runs.

    The `emission_weight` argument controls the EMS dispatch trade-off for
    this particular run.  By sweeping it across parallel workers you generate
    multiple Pareto-optimal solutions in one multiprocessing call.

    args tuple
    ----------
    seed, Swarmsize, MaxIter, Pc, Pm,
    latitude, longitude, altitude,
    sim_pars_fn, input_ts_fn, variables,
    mdf_fn, emission_weight
    """
    try:
        (seed, PopSize, MaxGen, Pc, Pm,
         latitude, longitude, altitude,
         sim_pars_fn, input_ts_fn, variables,
         mdf_fn, emission_weight) = args

        print(
            f"[Run {seed}] seed={seed}  emission_weight={emission_weight}  "
            f"PID={os.getpid()}",
            flush=True,
        )

        base_dir   = os.path.abspath('optimization_results_multi')
        worker_dir = os.path.join(
            base_dir,
            f'Multi_worker_run_Pc{Pc}_Pm{Pm}_seed{seed}_ew{emission_weight}_pid{os.getpid()}')
        os.makedirs(worker_dir, exist_ok=True)
        os.chdir(worker_dir)

        hpp = hpp_model(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            sim_pars_fn=sim_pars_fn,
            input_ts_fn=input_ts_fn,
            variables=variables,
            seed=seed,
            PopSize=PopSize,
            MaxGen=MaxGen,
            Pc=Pc,
            Pm=Pm,
            worker_dir=worker_dir,
            mdf_fn=mdf_fn,
            emission_weight=emission_weight,
        )

        hpp.run_optimization()

        result = {
            'run':             seed,
            'unique_seed':     seed,
            'emission_weight': emission_weight,
        }

        result['NPV_over_CAPEX'] = float(hpp.prob.get_val('NPV_over_CAPEX')[0])
        if hpp.use_carbon_offset:
            result['annual_carbon_offset'] = float(
                hpp.prob.get_val('annual_carbon_offset')[0])

        print(
            f"[Run {seed}] Done → NPV/CAPEX={result['NPV_over_CAPEX']:.4f}",
            flush=True,
        )
        if hpp.use_carbon_offset:
            print(
                f"[Run {seed}]        annual_carbon_offset="
                f"{result['annual_carbon_offset']:.0f} tCO2eq/yr",
                flush=True,
            )

        pareto_csv = os.path.join(
            os.path.dirname(hpp.filepath),
            f"pareto_front_VESPO_seed{seed}_ew{emission_weight}.csv"
        )
        import csv as _csv
        if hpp.prob.driver.pareto_obj:
            keys = (list(hpp.prob.driver.pareto_obj[0].keys()) +
                    list(hpp.prob.driver.pareto_pos[0].keys()))
            with open(pareto_csv, 'w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for obj, pos in zip(hpp.prob.driver.pareto_obj,
                                    hpp.prob.driver.pareto_pos):
                    row = {**obj,
                           **{k: float(np.atleast_1d(v)[0])
                              for k, v in pos.items()}}
                    w.writerow(row)
            print(f"[Run {seed}] Pareto front ({len(hpp.prob.driver.pareto_obj)} "
                  f"solutions) saved → {pareto_csv}", flush=True)

        return result

    except Exception:
        print(" WORKER CRASHED", flush=True)
        traceback.print_exc()
        raise


# =============================================================================
#  Main
# =============================================================================

if __name__ == "__main__":
    import multiprocessing
    import time
    from hydesign.examples import examples_filepath

    name = "France_good_wind"
    examples_sites = pd.read_csv(
        f"{examples_filepath}examples_sites.csv", index_col=0, sep=";"
    )
    ex_site = examples_sites.loc[examples_sites.name == name]

    longitude = ex_site["longitude"].values[0]
    latitude  = ex_site["latitude"].values[0]
    altitude  = ex_site["altitude"].values[0]

    sim_pars_fn = examples_filepath + ex_site["sim_pars_fn"].values[0]
    input_ts_fn = examples_filepath + ex_site["input_ts_fn"].values[0]

    # ------------------------------------------------------------------
    # Path to the MDF CSV.  Set to None to fall back to single-objective.
    # The file must have a datetime index in DD/MM/YYYY HH:MM format and
    # one numeric column (e.g. 'MDFwind_kgCO2eperMWh'), covering 2023-2024.
    # ------------------------------------------------------------------
    mdf_fn = r"C:\Users\pauld\OneDrive\Documents\Mes documents\DTU\4th Master Thesis\HyDesign_Code\Master_Thesis\hydesign\emission\df_mdf.csv"
    # mdf_fn = None   # ← uncomment to run single-objective only

    variables = {
        "clearance":    {"var_type": "design", "limits": [10, 120],  "types": "int"},
        "sp":           {"var_type": "design", "limits": [200, 360], "types": "int"},
        "p_rated":      {"var_type": "design", "limits": [5, 20],    "types": "int"},
        "Nwt":          {"var_type": "design", "limits": [5, 50],    "types": "int"},
        "wind_MW_per_km2": {"var_type": "design", "limits": [1, 10], "types": "float"},
        "solar_MW":     {"var_type": "design", "limits": [30, 200],  "types": "float"},
        "surface_tilt": {"var_type": "design", "limits": [0, 90],    "types": "float"},
        "surface_azimuth": {"var_type": "design", "limits": [150, 210], "types": "float"},
        "DC_AC_ratio":  {"var_type": "fixed", "value": 1.479},
        "b_P":          {"var_type": "fixed", "value": 10*20},
        "b_E_h":        {"var_type": "fixed", "value": 10*4},
        "cost_of_battery_P_fluct_in_peak_price_ratio": {"var_type": "fixed", "value": 8.75},
    }

    # ------------------------------------------------------------------
    # Single run (uncomment to test)
    # ------------------------------------------------------------------
    run_optimization_seed(
        (0, 15, 10, 0.6, 0.2,
        latitude, longitude, altitude,
        sim_pars_fn, input_ts_fn, variables,
        mdf_fn, 0.0)   # purely economic
        )

    # ------------------------------------------------------------------
    # Parallel Pareto sweep: one run per emission_weight value.
    # Each run is independent; together they trace the Pareto front.
    # emission_weight=0 → purely economic (baseline)
    # emission_weight>0 → increasing carbon preference
    # ------------------------------------------------------------------
    
    '''
    seed      = 0
    PopSize = 40
    MaxGen   = 30
    Pc, Pm    = 0.6, 0.2

    # Values to sweep for the Pareto front.
    # Adjust the upper end to match your MDF scale (kgCO2eq/MWh).
    # A rough guide: if mean MDF ~ 200 kgCO2eq/MWh and price ~ 50 €/MWh,
    # the carbon bonus equals the electricity price at ~250 €/tCO2eq.
    emission_weight_list = [0, 100, 200, 500, 1000]

    args_list = [
        (seed, PopSize, MaxGen, Pc, Pm,
         latitude, longitude, altitude,
         sim_pars_fn, input_ts_fn, variables,
         mdf_fn, ew)
        for ew in emission_weight_list
    ]

    start = time.time()
    n_processes = min(len(args_list), os.cpu_count() - 2)
    print(f"Launching {len(args_list)} Pareto runs on {n_processes} processes...",
          flush=True)

    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(run_optimization_seed, args_list)

    end = time.time()
    print(f"\n=== Pareto-front results (total time: {(end - start) / 60:.1f} min) ===")
    print(f"{'emission_weight':>16}  {'NPV/CAPEX':>10}  {'CO2 offset [tCO2eq/yr]':>22}")
    print("-" * 54)
    for r in results:
        co2_str = (f"{r['annual_carbon_offset']:>22.0f}"
                   if 'annual_carbon_offset' in r else f"{'N/A':>22}")
        print(f"{r['emission_weight']:>16.1f}  {r['NPV_over_CAPEX']:>10.4f}  {co2_str}")
    '''