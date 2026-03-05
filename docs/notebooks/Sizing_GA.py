import importlib

import openmdao.api as om
from hydesign.assembly.hpp_assembly import *
from hydesign.examples import examples_filepath
import pandas as pd
import os

import time

name = "France_good_wind"
examples_sites = pd.read_csv(
    f"{examples_filepath}examples_sites.csv", index_col=0, sep=";"
)
ex_site = examples_sites.loc[examples_sites.name == name]

longitude = ex_site["longitude"].values[0]
latitude = ex_site["latitude"].values[0]
altitude = ex_site["altitude"].values[0]

sim_pars_fn = examples_filepath + ex_site["sim_pars_fn"].values[0]
input_ts_fn = examples_filepath + ex_site["input_ts_fn"].values[0]

hpp = hpp_model(
    latitude=latitude,
    longitude=longitude,
    altitude=altitude,
    sim_pars_fn=sim_pars_fn,
    input_ts_fn=input_ts_fn,
)

start = time.time()

# x = [
#     55.0,
#     257.0,
#     10.000000000000002,
#     10.0,
#     5.916666666666667,
#     75.0,
#     28.125,
#     191.25,
#     1.4791666666666665,
#     27.0,
#     4.0,
#     8.75,
# ]

# outs = hpp.evaluate(*x)

# hpp.print_design()

# end = time.time()
# print("exec. time [min]:", (end - start) / 60)

# print(hpp.prob["NPV_over_CAPEX"])

hpp.test()
