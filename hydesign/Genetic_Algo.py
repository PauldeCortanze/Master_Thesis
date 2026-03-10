# -*- coding: utf-8 -*-
"""
Created on March 9 2026

"""
import os
import time
from multiprocessing import Pool
from sys import version_info

import numpy as np
import pandas as pd
import smt
from numpy import newaxis as na
from openmdao.core.driver import Driver

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import optimize
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from smt.applications.ego import Evaluator
from smt.applications.mixed_integer import (
    MixedIntegerContext,
    MixedIntegerSurrogateModel,
)

# SMT imports
from smt.design_space import (
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
)
from smt.sampling_methods import LHS, FullFactorial, Random
from smt.surrogate_models import GEKPLS, KPLS, KPLSK, KRG

# HyDesign imports
from hydesign.examples import examples_filepath

smt_version = smt.__version__.split(".")
smt_major, smt_minor = smt_version[:2]
# import platform


def extreme_around_point(x):
    ndims = x.shape[1]
    xcand = np.tile(x.T, ndims * 2).T
    for i in range(ndims):
        xcand[i, i] = 0.0
    for i in range(ndims):
        xcand[i + ndims, i] = 1.0
    return xcand


def perturbe_around_point(x, step=0.1):
    ndims = x.shape[1]
    xcand = np.tile(x.T, ndims * 2).T
    for i in range(ndims):
        xcand[i, i] += step
    for i in range(ndims):
        xcand[i + ndims, i] -= step

    xcand = np.maximum(xcand, 0)
    xcand = np.minimum(xcand, 1.0)
    return xcand


def get_design_vars(variables):
    return [
        var_ for var_ in variables.keys() if variables[var_]["var_type"] == "design"
    ], [var_ for var_ in variables.keys() if variables[var_]["var_type"] == "fixed"]


def get_limits(variables, design_var=[]):
    if len(design_var) == 0:
        design_var, fixed_var = get_design_vars(variables)
    return np.array([variables[var_]["limits"] for var_ in design_var])


def get_xlimits(variables, design_var=[]):
    if len(design_var) == 0:
        design_var, fixed_var = get_design_vars(variables)
    return np.array([variables[var_]["limits"] for var_ in design_var])


def get_xtypes(variables, design_var=[]):
    if len(design_var) == 0:
        design_var, fixed_var = get_design_vars(variables)
    return [variables[var_]["types"] for var_ in design_var]


def cast_to_mixint(x, variables):
    design_var, fixed_var = get_design_vars(variables)
    types_ = get_xtypes(variables)
    for i, ty in enumerate(types_):
        if ty == "int":
            x[:, i] = np.round(x[:, i])
        elif ty == "resolution":
            res = variables[design_var[i]]["resolution"]
            x[:, i] = np.round(x[:, i] / res, decimals=0) * res
    return x


def drop_duplicates(x, y, decimals=3):
    x_rounded = np.around(x, decimals=decimals)
    _, indices = np.unique(x_rounded, axis=0, return_index=True)
    x_unique = x[indices, :]
    y_unique = y[indices, :]
    return x_unique, y_unique


def concat_to_existing(x, y, xnew, ynew):
    x_concat, y_concat = drop_duplicates(
        np.vstack([x, xnew]), np.vstack([y, ynew]))
    return x_concat, y_concat


def get_mixint_context(variables, seed=None, criterion="maximin"):
    design_var, fixed_var = get_design_vars(variables)
    list_vars_doe = []
    for var_ in design_var:
        if variables[var_]["types"] == "int":
            list_vars_doe += [IntegerVariable(*variables[var_]["limits"])]
        elif variables[var_]["types"] == "float":
            list_vars_doe += [FloatVariable(*variables[var_]["limits"])]
        else:
            dtype = type(variables[var_]["resolution"])
            val_list = list(
                np.arange(
                    variables[var_]["limits"][0],
                    variables[var_]["limits"][1] +
                    variables[var_]["resolution"],
                    variables[var_]["resolution"],
                    dtype=dtype,
                )
            )
            list_vars_doe += [OrdinalVariable(val_list)]
    if int(smt_major) == 2:
        if int(smt_minor) < 4:
            mixint = MixedIntegerContext(DesignSpace(list_vars_doe, seed=seed))
        else:
            ds = DesignSpace(list_vars_doe, random_state=seed)
            ds.sampler = LHS(
                xlimits=ds.get_unfolded_num_bounds(),
                random_state=int(seed),
                criterion=criterion,
            )
            mixint = MixedIntegerContext(ds)
    return mixint


def get_sampling(mixint, seed, criterion="maximin"):
    if int(smt_major) == 2:
        if int(smt_minor) < 1:
            sampling = mixint.build_sampling_method(
                LHS, criterion=criterion, random_state=int(seed)
            )
        elif int(smt_minor) >= 1:
            mixint._design_space.sampler = LHS(
                xlimits=mixint.get_unfolded_xlimits(),
                criterion=criterion,
                random_state=int(seed),
            )
            sampling = mixint.build_sampling_method(random_state=int(seed))
        # else:
        #     sampling = mixint.build_sampling_method(random_state=int(seed))
        return sampling


def expand_x_for_model_eval(x, kwargs):

    list_vars = kwargs["list_vars"]
    variables = kwargs["variables"]
    design_vars = kwargs["design_vars"]
    fixed_vars = kwargs["fixed_vars"]

    x_eval = np.zeros([x.shape[0], len(list_vars)])

    for ii, var in enumerate(list_vars):
        if var in design_vars:
            x_eval[:, ii] = x[:, design_vars.index(var)]
        elif var in fixed_vars:
            x_eval[:, ii] = variables[var]["value"]

    return x_eval


def model_evaluation(inputs):  # Evaluates the model
    x, kwargs = inputs
    hpp_m = kwargs["hpp_model"](**kwargs, verbose=False)

    x = kwargs["scaler"].inverse_transform(x)
    x_eval = expand_x_for_model_eval(x, kwargs)
    try:
        return np.array(
            kwargs["opt_sign"] *
            hpp_m.evaluate(*x_eval[0, :])[kwargs["op_var_index"]]
        )
    except:
        print("There was an error with this case (or potentially memory error): ")
        print("x=[" + ", ".join(map(str, x_eval[0, :])) + "]")


class ParallelEvaluator(Evaluator):
    """
    Implement Evaluator interface using multiprocessing Pool object (Python 3 only).
    """

    def __init__(self, n_procs=31):
        self.n_procs = n_procs

    def run_ydoe(self, fun, x, **kwargs):
        n_procs = self.n_procs
        if version_info.major == 2:
            raise ("version_info.major==2")

        with Pool(n_procs) as p:
            return np.array(
                p.map(fun, [(x[[i], :], kwargs) for i in range(x.shape[0])])
            ).reshape(-1, 1)

    def run_both(self, fun, i, **kwargs):
        n_procs = self.n_procs
        if version_info.major == 2:
            raise ("version_info.major==2")

        with Pool(n_procs) as p:
            return p.map(
                fun,
                [
                    ((n + i * n_procs) * 100 + kwargs["n_seed"], kwargs)
                    for n in np.arange(n_procs)
                ],
            )

    def run_xopt_iter(self, fun, x, **kwargs):
        n_procs = self.n_procs
        if version_info.major == 2:
            raise ("version_info.major==2")

        with Pool(n_procs) as p:
            return np.vstack(
                p.map(fun, [(x[[ii], :], kwargs) for ii in range(x.shape[0])])
            )


def check_types(kwargs):
    # Only convert if key exists
    for x in ["num_batteries", "n_procs", "n_doe", "n_clusters", "n_seed", "max_iter"]:
        if x in kwargs:
            kwargs[x] = int(kwargs[x])

    # Only set default if missing
    if "final_design_fn" not in kwargs or kwargs["final_design_fn"] is None:
        kwargs["final_design_fn"] = (
            f'{kwargs["work_dir"]}design_hpp_{kwargs["name"]}_{kwargs["opt_var"]}.csv'
        )

    for x in ["opt_var", "final_design_fn"]:
        if x in kwargs:
            kwargs[x] = str(kwargs[x])

    return kwargs


class GeneticAlgorithmDriver(Driver):

    def __init__(self, **kwargs):
        os.environ["OPENMDAO_USE_MPI"] = "0"
        kwargs = check_types(kwargs)
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        for k, v in self.kwargs.items():
            self.options.declare(k, v)

    def run(self):
        kwargs = self.kwargs
        recorder = {
            "time": [],
            "yopt": [],
        }

        # Input
        start_total = time.time()

        variables = kwargs["variables"]
        design_vars, fixed_vars = get_design_vars(variables)
        xlimits = get_xlimits(variables, design_vars)
        xtypes = get_xtypes(variables, design_vars)

        # Scale design variables
        scaler = MinMaxScaler()
        scaler.fit(xlimits.T)

        # HPP model
        name = kwargs["name"]
        print("\n\n\n")
        print(f"Sizing a HPP plant at {name}:")
        print()
        list_minimize = ["LCOE [Euro/MWh]"]

        # Get index of output var to optimize
        # Get sign to always write the optimization as minimize
        opt_var = kwargs["opt_var"]
        opt_sign = -1
        if opt_var in list_minimize:
            opt_sign = 1

        kwargs["opt_sign"] = opt_sign
        kwargs["scaler"] = scaler
        kwargs["xtypes"] = xtypes
        kwargs["xlimits"] = xlimits

        hpp_m = kwargs["hpp_model"](**kwargs)
        # Update kwargs to use input file generated when extracting weather
        kwargs["input_ts_fn"] = hpp_m.input_ts_fn
        kwargs["altitude"] = hpp_m.altitude
        kwargs["price_fn"] = None

        print("\n\n")

        # Lists of all possible outputs, inputs to the hpp model
        # -------------------------------------------------------
        list_vars = hpp_m.list_vars
        list_out_vars = hpp_m.list_out_vars
        op_var_index = list_out_vars.index(opt_var)
        kwargs.update({"op_var_index": op_var_index})
        # Stablish types for design variables

        kwargs["list_vars"] = list_vars
        kwargs["design_vars"] = design_vars
        kwargs["fixed_vars"] = fixed_vars

        # GA parameters
        pop_size = kwargs["n_doe"]
        generations = kwargs["max_iter"]
        mutation_sigma = 0.05
        crossover_rate = 0.8

        # ----------------------------
        # Initial population (LHS)
        # ----------------------------
        mixint = get_mixint_context(kwargs["variables"], kwargs["n_seed"])
        sampling = get_sampling(
            mixint, seed=kwargs["n_seed"], criterion="maximin")

        X = sampling(pop_size)
        X = np.array(mixint.design_space.decode_values(X))
        X = scaler.transform(X)

        # Parallel evaluator
        start = time.time()
        n_procs = kwargs["n_procs"]
        PE = ParallelEvaluator(n_procs=n_procs)

        print("\nInitial population evaluation\n")

        Y = PE.run_ydoe(fun=model_evaluation, x=X, **kwargs)

        best_idx = np.argmin(Y)
        xbest = X[[best_idx]]
        ybest = Y[[best_idx]]

        lapse = np.round((time.time() - start) / 60, 2)
        print(f"Initial {X.shape[0]} simulations took {lapse} minutes")

        print(f"Initial best value: {float(np.squeeze(ybest))}")

        # GA main loop

        for gen in range(generations):
            start_gen = time.time()
            # Selection (tournament)
            parents = []
            for _ in range(pop_size):
                i, j = np.random.randint(pop_size, size=2)
                winner = i if Y[i] < Y[j] else j
                parents.append(X[winner])
            parents = np.array(parents)

            # Crossover
            children = []
            for i in range(0, pop_size, 2):
                p1 = parents[i]
                p2 = parents[(i + 1) % pop_size]
                if np.random.rand() < crossover_rate:
                    alpha = np.random.rand()
                    child1 = alpha * p1 + (1 - alpha) * p2
                    child2 = alpha * p2 + (1 - alpha) * p1
                else:
                    child1 = p1.copy()
                    child2 = p2.copy()
                children.append(child1)
                children.append(child2)

            children = np.array(children[:pop_size])

            # Mutation
            mutation = np.random.normal(0, mutation_sigma, children.shape)
            children = children + mutation

            children = np.clip(children, 0, 1)

            # Cast integer / resolution variables
            children_unscaled = scaler.inverse_transform(children)
            children_unscaled = cast_to_mixint(children_unscaled, variables)
            children = scaler.transform(children_unscaled)

            # Evaluate offspring
            Y_children = PE.run_ydoe(
                fun=model_evaluation, x=children, **kwargs)

            # Combine populations
            X = np.vstack([X, children])
            Y = np.vstack([Y, Y_children])

            idx = np.argsort(Y.flatten())[:pop_size]

            X = X[idx]
            Y = Y[idx]

            best_idx = np.argmin(Y)
            xbest = X[[best_idx]]
            ybest = Y[[best_idx]]

            lapse = np.round((time.time() - start_gen) / 60, 2)

            print(
                f"Generation {gen+1} | Best {opt_var} = {float(np.squeeze(ybest)):.4E} | time {lapse} min"
            )

        # Final evaluation
        xbest = scaler.inverse_transform(xbest)
        xbest = expand_x_for_model_eval(xbest, kwargs)

        outs = hpp_m.evaluate(*xbest[0, :])

        hpp_m.print_design(xbest[0, :], outs)

        lapse = np.round((time.time() - start_total) / 60, 2)

        print(f"\nGA optimization finished in {lapse} minutes\n")

        # Store results
        design_df = pd.DataFrame(columns=list_vars, index=[kwargs["name"]])

        for iv, var in enumerate(list_vars):
            design_df[var] = xbest[0, iv]

        for iv, var in enumerate(list_out_vars):
            design_df[var] = outs[iv]

        design_df["design obj"] = opt_var
        design_df["opt time [min]"] = lapse

        design_df.T.to_csv(kwargs["final_design_fn"])

        self.result = design_df
        self.hpp_m = hpp_m
        '''
        # Evaluate model at initial doe
        start = time.time()
        n_procs = kwargs["n_procs"]
        PE = ParallelEvaluator(n_procs=n_procs)
        ydoe = PE.run_ydoe(fun=model_evaluation, x=xdoe, **kwargs)

        lapse = np.round((time.time() - start) / 60, 2)
        print(f"Initial {xdoe.shape[0]} simulations took {lapse} minutes")

        # Initialize iterative optimization
        itr = 0
        error = 1e10
        conv_iter = 0
        xopt = xdoe[[np.argmin(ydoe)], :]
        yopt = ydoe[[np.argmin(ydoe)], :]
        kwargs["yopt"] = yopt
        yold = np.copy(yopt)
        # xold = None
        print(
            f"  Current solution {opt_sign}*{opt_var} = {float(np.squeeze(yopt)):.3E}".replace(
                "1*", ""
            )
        )
        print(f"  Current No. model evals: {xdoe.shape[0]}\n")
        recorder["time"].append(time.time())
        recorder["yopt"].append(float(np.squeeze(yopt)))
        sm_args = {"n_comp": min(len(design_vars), 4)}
        sm_args.update(
            {k: v for k, v in kwargs.items() if k in ["theta_bounds", "n_comp"]}
        )
        while itr < kwargs["max_iter"]:
            # Iteration
            start_iter = time.time()

            # Train surrogate model
            np.random.seed(kwargs["n_seed"])
            sm = get_sm(xdoe, ydoe, **sm_args)
            kwargs["sm"] = sm

            # Evaluate surrogate model in a large number of design points
            # in parallel
            start = time.time()
            both = PE.run_both(surrogate_evaluation, itr, **kwargs)
            # with Pool(n_procs) as p:
            #     both = ( p.map(fun_par, (np.arange(n_procs)+itr*100) * 100 + itr) )
            xpred = np.vstack([both[ii][0] for ii in range(len(both))])
            ypred_LB = np.vstack([both[ii][1] for ii in range(len(both))])

            # Get candidate points from clustering all sm evalautions
            n_clusters = kwargs["n_clusters"]
            xnew = get_candiate_points(
                xpred, ypred_LB, n_clusters=n_clusters, quantile=1e-2  # n_clusters - 1,
            )  # 1/(kwargs['npred']/n_clusters) )
            # request candidate points based on global evaluation of current surrogate
            # returns best designs in n_cluster of points with outputs bellow a quantile
            lapse = np.round((time.time() - start) / 60, 2)
            print(f"Update sm and extract candidate points took {lapse} minutes")

            # -------------------
            # Refinement
            # -------------------
            # # optimize the sm starting on the cluster based candidates and the best design
            # xnew, _ = concat_to_existing(xnew, _, xopt, _)
            # xopt_iter = PE.run_xopt_iter(surrogate_optimization, xnew, **kwargs)

            # 2C)
            if np.abs(error) < kwargs["tol"]:
                # add refinement around the opt
                np.random.seed(
                    kwargs["n_seed"] * 100 + itr
                )  # to have a different refinement per iteration
                step = np.random.uniform(low=0.05, high=0.25, size=1)
                xopt_iter = perturbe_around_point(xopt, step=step)
            else:
                # add extremes on each opt_var (one at a time) around the opt
                xopt_iter = extreme_around_point(xopt)

            xopt_iter = scaler.inverse_transform(xopt_iter)
            xopt_iter = cast_to_mixint(xopt_iter, kwargs["variables"])
            xopt_iter = scaler.transform(xopt_iter)
            xopt_iter, _ = drop_duplicates(xopt_iter, np.zeros_like(xopt_iter))
            xopt_iter, _ = concat_to_existing(
                xnew, np.zeros_like(xnew), xopt_iter, np.zeros_like(xopt_iter)
            )

            # run model at all candidate points
            start = time.time()
            yopt_iter = PE.run_ydoe(fun=model_evaluation, x=xopt_iter, **kwargs)

            lapse = np.round((time.time() - start) / 60, 2)
            print(
                f"Check-optimal candidates: new {xopt_iter.shape[0]} simulations took {lapse} minutes"
            )

            # update the db of model evaluations, xdoe and ydoe
            xdoe_upd, ydoe_upd = concat_to_existing(xdoe, ydoe, xopt_iter, yopt_iter)
            xdoe_upd, ydoe_upd = drop_duplicates(xdoe_upd, ydoe_upd)

            # Drop yopt if it is not better than best design seen
            xopt = xdoe_upd[[np.argmin(ydoe_upd)], :]
            yopt = ydoe_upd[[np.argmin(ydoe_upd)], :]

            recorder["time"].append(time.time())
            recorder["yopt"].append(float(np.squeeze(yopt)))

            # if itr > 0:
            error = opt_sign * float(1 - (np.squeeze(yold) / np.squeeze(yopt)))

            xdoe = np.copy(xdoe_upd)
            ydoe = np.copy(ydoe_upd)
            # xold = np.copy(xopt)
            yold = np.copy(yopt)
            itr = itr + 1
            lapse = np.round((time.time() - start_iter) / 60, 2)

            print(
                f"  Current solution {opt_sign}*{opt_var} = {float(np.squeeze(yopt)):.3E}".replace(
                    "1*", ""
                )
            )
            print(f"  Current No. model evals: {xdoe.shape[0]}")
            print(f"  rel_yopt_change = {error:.2E}")
            print(f"Iteration {itr} took {lapse} minutes\n")

            if np.abs(error) < kwargs["tol"]:
                conv_iter += 1
                if conv_iter >= kwargs["min_conv_iter"]:
                    print("Surrogate based optimization is converged.")
                    break
            else:
                conv_iter = 0

        xopt = scaler.inverse_transform(xopt)
        xopt = expand_x_for_model_eval(xopt, kwargs)

        # Re-Evaluate the last design to get all outputs
        outs = hpp_m.evaluate(*xopt[0, :])
        yopt = np.array(opt_sign * outs[[op_var_index]])[:, na]
        hpp_m.print_design(xopt[0, :], outs)

        recorder["time"].append(time.time())
        recorder["yopt"].append(float(np.squeeze(yopt)))

        n_model_evals = xdoe.shape[0]

        lapse = np.round((time.time() - start_total) / 60, 2)
        print(
            f"Optimization with {itr} iterations and {n_model_evals} model evaluations took {lapse} minutes\n"
        )

        # Store results
        # -----------------
        design_df = pd.DataFrame(columns=list_vars, index=[name])
        for var_ in ["name", "longitude", "latitude", "altitude"]:
            design_df[var_] = kwargs[var_]
        for iv, var in enumerate(list_vars):
            design_df[var] = xopt[0, iv]
        for iv, var in enumerate(list_out_vars):
            design_df[var] = outs[iv]

        design_df["design obj"] = opt_var
        design_df["opt time [min]"] = lapse
        design_df["n_model_evals"] = n_model_evals

        design_df.T.to_csv(kwargs["final_design_fn"])
        self.result = design_df
        # store final model, to check or extract additional variables
        self.hpp_m = hpp_m
        self.recorder = recorder
        '''


if __name__ == "__main__":
    from hydesign.assembly.hpp_assembly import hpp_model

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

    inputs = {
        # HPP Model Inputs
        "name": name,
        "longitude": longitude,
        "latitude": latitude,
        "altitude": altitude,
        "input_ts_fn": input_ts_fn,
        "sim_pars_fn": sim_pars_fn,
        "num_batteries": 10,
        "work_dir": "./",
        "hpp_model": hpp_model,
        # Genetic algorithm inputs
        "opt_var": "NPV_over_CAPEX",
        "n_procs": 4,
        "n_doe": 20,
        "n_seed": 0,
        "max_iter": 10,
        "final_design_fn": "hydesign_design_0.csv",
        # Design Variables
        "variables": {
            # "clearance [m]": {"var_type": "design", "limits": [10, 60], "types": "int"},
            "clearance [m]": {"var_type": "fixed", "value":28},
            # "sp [W/m2]": {"var_type": "design", "limits": [200, 360], "types": "int"},
            "sp [W/m2]": {"var_type": "fixed", "value": 360},
            "p_rated [MW]": {"var_type": "fixed", "value": 6},
            "Nwt": {"var_type": "fixed", "value": 200},
            "wind_MW_per_km2 [MW/km2]": {"var_type": "fixed", "value": 7},
            "solar_MW [MW]": {"var_type": "fixed", "value": 200},
            "surface_tilt [deg]": {"var_type": "fixed", "value": 25},
            "surface_azimuth [deg]": {
                "var_type": "design",
                "limits": [150, 210],
                "types": "float",
            },
            "DC_AC_ratio": {
                "var_type": "fixed",
                "value": 1.0,
            },
            "b_P [MW]": {"var_type": "fixed", "value": 50},
            "b_E_h [h]": {"var_type": "fixed", "value": 6},
            "cost_of_battery_P_fluct_in_peak_price_ratio": {
                "var_type": "fixed",
                "value": 10,
            },
        },
    }
    GA = GeneticAlgorithmDriver(**inputs)
    GA.run()
    result = GA.result

    print("\nOptimization result:\n")
    print(result)

    # import pickle
    # with open('recording.pkl', 'wb') as f:
    #     pickle.dump(GA.recorder, f)
