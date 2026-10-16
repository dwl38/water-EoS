#! /usr/bin/env python

_PROG_DESC_HEADER = '''
==============================================================================
 This program takes in a sequence of canonical-ensemble thermodynamic data for
 liquid water (which may be generated from Molecular Dynamics or Monte Carlo
 simulations), and constructs an Equation of State for self-consistent
 interpolation of thermodynamic properties at unseen state points. Use as:

 >    python build_equation_of_state.py <input_file> [-o <output_dir>]

 All plots will be saved to an output directory (specified by the -o option),
 which defaults to "EoS" within the parent directory that this program is
 executed from. The resulting FreeEnergySurface object is also saved to the
 output directory, allowing the fitted Equation of State to be reused easily.

 To be specific, the Equation of State is modelled in terms of the Helmholtz
 free energy as a function of volume per particle and temperature `F(V, T)`,
 which is internally expressed in terms of a nondimensionalized function
 `F(V, T) = kB * T * f(x, y)` where `x` is the reduced density and `y` is the
 reduced inverse temperature. The nondimensionalized function `f(x, y)` is
 then fitted using cardinal B-splines, enabling fully flexible yet regularized
 local interpolation.

 The input file should contain the following thermodynamic data:

  - V: The volume per particle, in Å³.
  - T: The temperature, in K.
  - p: The pressure corresponding to V and T, in bar.
  - p_err: Uncertainties of p, in bar.
  - U: The internal (total) energy corresponding to V and T, in eV.
  - U_err: Uncertainties of U, in eV.

 The input file should be provided in one of the following formats:

  - YAML or JSON file. File extension ending in ".yaml", ".yml", or ".json".
    Should represent a dictionary with all of the fields "V", "T", "p",
    "p_err", "U", and "U_err" present, which should all be lists or otherwise
    iterables of floats of the same length.

  - Plaintext DAT file. File extension ending in ".dat" or ".txt". The first
    two lines will be ignored as comments, and all remaining lines will be
    interpreted as space-separated values, in the following order:

    >>> # Filename, title, other comments, etc.
    >>> #   V     T     p   p_err   U   U_err
    >>>   xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx
    >>>   xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx
    >>>    ...   ...   ...   ...   ...   ...

  - Numpy archive. File extension ending in ".npz". Should contain the fields
    "V", "T", "p", "p_err", "U", and "U_err", which are all expected to be 1D
    ndarrays of floats of the same shape.

==============================================================================
'''

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from pathlib import Path

from fast_water_EoS.io import read_file
from fast_water_EoS.misc import elapsed_time, SI_prefix
from fast_water_EoS.free_energy_surface import FreeEnergySurface
from fast_water_EoS.constants import (
    WATER_MASS_DENSITY_CONVERSION,
    PRESSURE_CONVERSION,
    K_B,
    AMBIENT_VOLUME,
    AMBIENT_TEMPERATURE,
)

_N_PLOT_POINTS = 200
_ISOCHORE_DENSITIES = [920.0, 960.0, 1000.0, 1040.0, 1080.0, 1120.0, 1160.0]
_ISOTHERM_TEMPS = [250.0, 275.0, 300.0, 325.0, 350.0, 375.0]
_ISOBAR_PRESSURES = [-1000.0, 0.0, 1000.0, 2000.0, 3000.0, 4000.0]
_DENSITY_RANGE = [880.0, 1180.0]
_TEMP_RANGE = [250.1, 374.9]
_PRESSURE_RANGE = [-1999.0, 9999.0]
_HEAT_CAPACITY_RANGE = [0.0, 0.00199]
_COMPRESSIBILITY_RANGE = [0.0, 0.00015]
_THERMAL_PRESSURE_RANGE = [-35.0, 35.0]
_THERMAL_EXPANSION_RANGE = [-0.00112, 0.00112]
_SAVEFIG_PARAMS = {'dpi': 300.0, 'bbox_inches': 'tight', 'pad_inches': 0.05}

def main() -> None:

    #-------------------------------------------------------------------------
    # Generate program description and parse input arguments

    prog_desc = ''
    for line in _PROG_DESC_HEADER.splitlines()[2:-1]:
        prog_desc += line + '\n'

    parser = argparse.ArgumentParser(
        prog='build_equation_of_state',
        description=prog_desc,
        usage='%(prog)s input_file [-o <output_dir>]',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'input_file',
        help='input file to read data from',
    )
    parser.add_argument(
        '-o', '--output',
        default='EoS',
        dest='output_dir',
        help='output folder to save results and graphical outputs to',
    )
    args = parser.parse_args()

    if not Path(args.input_file).is_file():
        raise RuntimeError(f'File "{args.input_file}" not found.')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #-------------------------------------------------------------------------
    # Read input file and fit free energy surface

    time_start_0 = time.time()

    data = read_file(args.input_file)
    V_obs, T_obs, p_obs, p_obs_err, U_obs, U_obs_err = data
    rho_obs = WATER_MASS_DENSITY_CONVERSION / V_obs
    fes = FreeEnergySurface()
    fes.fit_to_data(V_obs, T_obs, p_obs, p_obs_err, U_obs, U_obs_err)
    fes.save_to_file(output_dir / 'EoS.yaml')

    print(f'Equation of state fitted in {elapsed_time(time_start_0)}.')

    #-------------------------------------------------------------------------
    # Plot internal energy surface

    time_start_1 = time.time()

    rho_range = np.linspace(*_DENSITY_RANGE, _N_PLOT_POINTS)
    V_range = WATER_MASS_DENSITY_CONVERSION / rho_range
    T_range = np.linspace(*_TEMP_RANGE, _N_PLOT_POINTS)
    V_plot, T_plot = np.meshgrid(V_range, T_range)
    V_plot, T_plot = V_plot.ravel(), T_plot.ravel()
    rho_plot = WATER_MASS_DENSITY_CONVERSION / V_plot

    mean_surface = fes.internal_energy(V_plot, T_plot)
    unc_surface = fes.internal_energy_unc(V_plot, T_plot)
    vmin = min(np.min(mean_surface), np.min(U_obs))
    vmax = max(np.max(mean_surface), np.max(U_obs))
    levels = np.linspace(vmin, vmax, _N_PLOT_POINTS)
    errmax = max(np.max(unc_surface), np.max(U_obs_err))
    errlevels = np.linspace(0.0, errmax, _N_PLOT_POINTS)

    deviations = U_obs - fes.internal_energy(V_obs, T_obs)
    mae = SI_prefix(np.mean(np.abs(deviations)), 'eV')
    rmse = SI_prefix(np.sqrt(np.mean(deviations**2)), 'eV')

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12.0, 4.0)
    art = ax[0].tricontourf(T_plot, rho_plot, mean_surface, levels=levels,
                            cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].scatter(T_obs, rho_obs, c=U_obs, cmap='viridis', vmin=vmin,
                  vmax=vmax, ec='white')
    fig.colorbar(art, ax=ax[0], label=r'$E(T,\rho)\;[\text{eV}]$')
    art = ax[1].tricontourf(T_plot, rho_plot, unc_surface, levels=errlevels,
                            cmap='Reds', vmin=0.0, vmax=errmax)
    ax[1].scatter(T_obs, rho_obs, c=U_obs_err, cmap='Reds', vmin=0.0,
                  vmax=errmax, ec='b')
    fig.colorbar(art, ax=ax[1], label=r'$\sigma_{E}(T,\rho)\;[\text{eV}]$')
    for i in range(2):
        ax[i].set_xlim(*_TEMP_RANGE)
        ax[i].set_ylim(*_DENSITY_RANGE)
        ax[i].set_xlabel(r'$T\;[\text{K}]$')
        ax[i].set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
    ax[0].set_title(f'Fitted (MAE = {mae}, RMSE = {rmse})')
    ax[1].set_title(f'Uncertainty')
    fig.suptitle('Total internal energy against temperature and density')
    fig.tight_layout()
    fig.savefig(output_dir / 'internal-energy.png', **_SAVEFIG_PARAMS)

    print(f'Internal energy surface plotted in {elapsed_time(time_start_1)}.')

    #-------------------------------------------------------------------------
    # Plot pressure surface

    time_start_1 = time.time()

    mean_surface = fes.pressure(V_plot, T_plot)
    unc_surface = fes.pressure_unc(V_plot, T_plot)
    vmax = max(np.max(np.abs(mean_surface)), np.max(np.abs(p_obs)))
    norm = AsinhNorm(1.0, -vmax, vmax)
    levels = np.asarray(norm.inverse(np.linspace(0.0, 1.0, _N_PLOT_POINTS)))
    errmax = max(np.max(unc_surface), np.max(p_obs_err))
    errlevels = np.linspace(0.0, errmax, _N_PLOT_POINTS)

    deviations = p_obs - fes.pressure(V_obs, T_obs)
    mae = SI_prefix(np.mean(np.abs(deviations)), 'bar')
    rmse = SI_prefix(np.sqrt(np.mean(deviations**2)), 'bar')

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12.0, 4.0)
    art = ax[0].tricontourf(T_plot, rho_plot, mean_surface, levels=levels,
                            cmap='berlin', norm=norm)
    ax[0].scatter(T_obs, rho_obs, c=p_obs, cmap='berlin', norm=norm,
                  ec='lime')
    fig.colorbar(art, ax=ax[0], label=r'$p(T,\rho)\;[\text{bar}]$')
    art = ax[1].tricontourf(T_plot, rho_plot, unc_surface, levels=errlevels,
                            cmap='Reds', vmin=0.0, vmax=errmax)
    ax[1].scatter(T_obs, rho_obs, c=p_obs_err, cmap='Reds', vmin=0.0,
                  vmax=errmax, ec='b')
    fig.colorbar(art, ax=ax[1], label=r'$\sigma_{p}(T,\rho)\;[\text{bar}]$')
    for i in range(2):
        ax[i].set_xlim(*_TEMP_RANGE)
        ax[i].set_ylim(*_DENSITY_RANGE)
        ax[i].set_xlabel(r'$T\;[\text{K}]$')
        ax[i].set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
    ax[0].set_title(f'Fitted (MAE = {mae}, RMSE = {rmse})')
    ax[1].set_title(f'Uncertainty')
    fig.suptitle('Pressure against temperature and density')
    fig.tight_layout()
    fig.savefig(output_dir / 'pressure.png', **_SAVEFIG_PARAMS)

    print(f'Pressure surface plotted in {elapsed_time(time_start_1)}.')

    #-------------------------------------------------------------------------
    # Plot uncertainty surface

    time_start_1 = time.time()

    xdfdx_unc = (fes.pressure_unc(V_plot, T_plot) * V_plot
                 / (PRESSURE_CONVERSION * K_B * T_plot))
    ydfdy_unc = fes.internal_energy_unc(V_plot, T_plot) / (K_B * T_plot)
    unc_surface = np.sqrt((xdfdx_unc**2) + (ydfdy_unc**2))
    vmax = np.max(unc_surface)
    levels = np.linspace(0.0, vmax, _N_PLOT_POINTS)

    fig, ax = plt.subplots()
    fig.set_size_inches(6.0, 4.0)
    art = ax.tricontourf(T_plot, rho_plot, unc_surface, levels=levels,
                         cmap='Reds', vmin=0.0, vmax=vmax)
    fig.colorbar(art, ax=ax)
    ax.set_xlim(*_TEMP_RANGE)
    ax.set_ylim(*_DENSITY_RANGE)
    ax.set_xlabel(r'$T\;[\text{K}]$')
    ax.set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
    ax.set_title('Combined dimensionless uncertainty')
    fig.savefig(output_dir / 'uncertainty.png', **_SAVEFIG_PARAMS)

    print(f'Uncertainty surface plotted in {elapsed_time(time_start_1)}.')

    #-------------------------------------------------------------------------
    # Plot isochores

    time_start_1 = time.time()

    x = np.linspace(0.0, 1.0, len(_ISOCHORE_DENSITIES))
    colors = np.c_[x, 0.75 - (3.0 * ((x - 0.5)**2)), 1.0 - x]

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(9.0, 7.0)
    for density, color, zorder in zip(_ISOCHORE_DENSITIES, colors, x):
        label = r'$\rho\,=\,' + f'{density:.0f}' + r'\,\text{kg}/\text{m}^3$'
        vol = WATER_MASS_DENSITY_CONVERSION / density
        p_curve = fes.pressure(vol, T_range)
        p_curve_err = 2.0 * fes.pressure_unc(vol, T_range)
        Cv = fes.heat_capacity(vol, T_range)
        Cv_err = 2.0 * fes.heat_capacity_unc(vol, T_range)
        gamma = fes.thermal_pressure_coefficient(vol, T_range)
        gamma_err = 2.0 * fes.thermal_pressure_coefficient_unc(vol, T_range)
        beta = fes.compressibility(vol, T_range)
        beta_err = 2.0 * fes.compressibility_unc(vol, T_range)
        ax[0][0].fill_between(T_range, p_curve - p_curve_err,
                              p_curve + p_curve_err, color=color, ec='none',
                              alpha=0.3, zorder=zorder)
        ax[0][0].plot(T_range, p_curve, '-', color=color, label=label,
                      zorder=(1.0 + zorder))
        ax[0][1].fill_between(T_range, Cv - Cv_err, Cv + Cv_err, color=color,
                              ec='none', alpha=0.3, zorder=zorder)
        ax[0][1].plot(T_range, Cv, '-', color=color, zorder=(1.0 + zorder))
        ax[1][0].fill_between(T_range, gamma - gamma_err, gamma + gamma_err,
                              color=color, ec='none', alpha=0.3,
                              zorder=zorder)
        ax[1][0].plot(T_range, gamma, '-', color=color, zorder=(1.0 + zorder))
        ax[1][1].fill_between(T_range, beta - beta_err, beta + beta_err,
                              color=color, ec='none', alpha=0.3,
                              zorder=zorder)
        ax[1][1].plot(T_range, beta, '-', color=color, zorder=(1.0 + zorder))
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlim(*_TEMP_RANGE)
            ax[i][j].set_xlabel(r'$T\;[\text{K}]$')
    ax[0][0].set_ylim(*_PRESSURE_RANGE)
    ax[0][0].set_ylabel(r'$p\;[\text{bar}]$')
    ax[0][0].set_title('Pressure')
    ax[0][1].set_ylim(*_HEAT_CAPACITY_RANGE)
    ax[0][1].set_ylabel(r'$C_{V}\;[\text{eV}/\text{K}]$')
    ax[0][1].set_title('Isochoric heat capacity')
    ax[1][0].set_ylim(*_THERMAL_PRESSURE_RANGE)
    ax[1][0].set_ylabel(r'$\gamma\;[\text{bar}/\text{K}]$')
    ax[1][0].set_title('Isochoric thermal pressure coefficient')
    ax[1][1].set_ylim(*_COMPRESSIBILITY_RANGE)
    ax[1][1].set_ylabel(r'$\beta\;[\text{bar}^{-1}]$')
    ax[1][1].set_title('Isothermal compressibility')
    fig.suptitle('Isochores')
    fig.legend(loc='outside right center')
    fig.savefig(output_dir / 'isochores.png', **_SAVEFIG_PARAMS)

    print(f'Isochores plotted in {elapsed_time(time_start_1)}.')

    #-------------------------------------------------------------------------
    # Plot isotherms

    time_start_1 = time.time()

    x = np.linspace(0.0, 1.0, len(_ISOTHERM_TEMPS))
    colors = np.c_[x, 0.75 - (3.0 * ((x - 0.5)**2)), 1.0 - x]

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(9.0, 7.0)
    for temp, color, zorder in zip(_ISOTHERM_TEMPS, colors, x):
        label = r'$T\,=\,' + f'{temp:.0f}' + r'\,\text{K}$'
        p_range = fes.pressure(V_range, temp)
        p_min = np.clip(np.min(p_range) + 1.0, *_PRESSURE_RANGE)
        p_max = np.clip(np.max(p_range) - 1.0, *_PRESSURE_RANGE)
        p_range = np.linspace(p_min, p_max, _N_PLOT_POINTS)
        start_idx = np.argmin(np.abs(p_range))
        volumes = np.full(_N_PLOT_POINTS, AMBIENT_VOLUME)
        for j in range(start_idx, _N_PLOT_POINTS):
            volumes[j:] = fes.volume_at_p(
                p_range[j],
                temp,
                init_guess=volumes[j],
            )
            if np.isnan(volumes[j]):
                break
        for j in range(start_idx, 0, -1):
            volumes[:j] = fes.volume_at_p(
                p_range[j-1],
                temp,
                init_guess=volumes[j-1],
            )
            if np.isnan(volumes[j-1]):
                break
        densities = WATER_MASS_DENSITY_CONVERSION / volumes
        gamma = fes.thermal_pressure_coefficient(volumes, T_range)
        gamma_err = 2.0 * fes.thermal_pressure_coefficient_unc(volumes,
                                                               T_range)
        alpha = fes.thermal_expansion_coefficient(volumes, T_range)
        alpha_err = 2.0 * fes.thermal_expansion_coefficient_unc(volumes,
                                                                T_range)
        beta = fes.compressibility(volumes, T_range)
        beta_err = 2.0 * fes.compressibility_unc(volumes, T_range)
        p_unc = fes.pressure_unc(volumes, T_range)
        volumes_err = np.abs(volumes * beta * p_unc)
        densities_err = (WATER_MASS_DENSITY_CONVERSION * volumes_err
                         / (volumes**2))
        ax[0][0].fill_between(p_range, densities - densities_err,
                              densities + densities_err, color=color,
                              ec='none', alpha=0.3, zorder=zorder)
        ax[0][0].plot(p_range, densities, '-', color=color, label=label,
                      zorder=(1.0 + zorder))
        ax[0][1].fill_between(p_range, gamma - gamma_err, gamma + gamma_err,
                              color=color, ec='none', alpha=0.3,
                              zorder=zorder)
        ax[0][1].plot(p_range, gamma, '-', color=color, zorder=(1.0 + zorder))
        ax[1][0].fill_between(p_range, alpha - alpha_err, alpha + alpha_err,
                              color=color, ec='none', alpha=0.3,
                              zorder=zorder)
        ax[1][0].plot(p_range, alpha, '-', color=color, zorder=(1.0 + zorder))
        ax[1][1].fill_between(p_range, beta - beta_err, beta + beta_err,
                              color=color, ec='none', alpha=0.3,
                              zorder=zorder)
        ax[1][1].plot(p_range, beta, '-', color=color, zorder=(1.0 + zorder))
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlim(*_PRESSURE_RANGE)
            ax[i][j].set_xlabel(r'$p\;[\text{bar}]$')
    ax[0][0].set_ylim(*_DENSITY_RANGE)
    ax[0][0].set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
    ax[0][0].set_title('Density')
    ax[0][1].set_ylim(*_THERMAL_PRESSURE_RANGE)
    ax[0][1].set_ylabel(r'$\gamma\;[\text{bar}/\text{K}]$')
    ax[0][1].set_title('Isochoric thermal pressure coefficient')
    ax[1][0].set_ylim(*_THERMAL_EXPANSION_RANGE)
    ax[1][0].set_ylabel(r'$\alpha\;[\text{K}^{-1}]$')
    ax[1][0].set_title('Isobaric thermal expansion coefficient')
    ax[1][1].set_ylim(*_COMPRESSIBILITY_RANGE)
    ax[1][1].set_ylabel(r'$\beta\;[\text{bar}^{-1}]$')
    ax[1][1].set_title('Isothermal compressibility')
    fig.suptitle('Isotherms')
    fig.legend(loc='outside right center')
    fig.savefig(output_dir / 'isotherms.png', **_SAVEFIG_PARAMS)

    print(f'Isotherms plotted in {elapsed_time(time_start_1)}.')

    #-------------------------------------------------------------------------
    # Plot isobars

    time_start_1 = time.time()

    x = np.linspace(0.0, 1.0, len(_ISOBAR_PRESSURES))
    colors = np.c_[x, 0.75 - (3.0 * ((x - 0.5)**2)), 1.0 - x]

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(9.0, 7.0)
    for press, color, zorder in zip(_ISOBAR_PRESSURES, colors, x):
        label = r'$p\,=\,' + f'{press:.0f}' + r'\,\text{bar}$'
        start_idx = np.argmin(np.abs(T_range - AMBIENT_TEMPERATURE))
        init_volume = fes.volume_at_p(press, AMBIENT_TEMPERATURE,
                                      init_guess=AMBIENT_VOLUME)
        volumes = np.full(_N_PLOT_POINTS, init_volume)
        for j in range(start_idx, _N_PLOT_POINTS):
            volumes[j:] = fes.volume_at_p(
                press,
                T_range[j],
                init_guess=volumes[j],
            )
            if np.isnan(volumes[j]):
                break
        for j in range(start_idx, 0, -1):
            volumes[:j] = fes.volume_at_p(
                press,
                T_range[j-1],
                init_guess=volumes[j-1],
            )
            if np.isnan(volumes[j-1]):
                break
        densities = WATER_MASS_DENSITY_CONVERSION / volumes
        Cp = fes.isobaric_heat_capacity(volumes, T_range)
        Cp_err = 2.0 * fes.isobaric_heat_capacity_unc(volumes, T_range)
        alpha = fes.thermal_expansion_coefficient(volumes, T_range)
        alpha_err = 2.0 * fes.thermal_expansion_coefficient_unc(volumes,
                                                                T_range)
        beta = fes.compressibility(volumes, T_range)
        beta_err = 2.0 * fes.compressibility_unc(volumes, T_range)
        p_unc = fes.pressure_unc(volumes, T_range)
        volumes_err = np.abs(volumes * beta * p_unc)
        densities_err = (WATER_MASS_DENSITY_CONVERSION * volumes_err
                         / (volumes**2))
        ax[0][0].fill_between(T_range, densities - densities_err,
                              densities + densities_err, color=color,
                              ec='none', alpha=0.3, zorder=zorder)
        ax[0][0].plot(T_range, densities, '-', color=color, label=label,
                      zorder=(1.0 + zorder))
        ax[0][1].fill_between(T_range, Cp - Cp_err, Cp + Cp_err, color=color,
                              ec='none', alpha=0.3, zorder=zorder)
        ax[0][1].plot(T_range, Cp, '-', color=color, zorder=(1.0 + zorder))
        ax[1][0].fill_between(T_range, alpha - alpha_err, alpha + alpha_err,
                              color=color, ec='none', alpha=0.3,
                              zorder=zorder)
        ax[1][0].plot(T_range, alpha, '-', color=color, zorder=(1.0 + zorder))
        ax[1][1].fill_between(T_range, beta - beta_err, beta + beta_err,
                              color=color, ec='none', alpha=0.3,
                              zorder=zorder)
        ax[1][1].plot(T_range, beta, '-', color=color, zorder=(1.0 + zorder))
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlim(*_TEMP_RANGE)
            ax[i][j].set_xlabel(r'$T\;[\text{K}]$')
    ax[0][0].set_ylim(*_DENSITY_RANGE)
    ax[0][0].set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
    ax[0][0].set_title('Density')
    ax[0][1].set_ylim(*_HEAT_CAPACITY_RANGE)
    ax[0][1].set_ylabel(r'$C_{p}\;[\text{eV}/\text{K}]$')
    ax[0][1].set_title('Isobaric heat capacity')
    ax[1][0].set_ylim(*_THERMAL_EXPANSION_RANGE)
    ax[1][0].set_ylabel(r'$\alpha\;[\text{K}^{-1}]$')
    ax[1][0].set_title('Isobaric thermal expansion coefficient')
    ax[1][1].set_ylim(*_COMPRESSIBILITY_RANGE)
    ax[1][1].set_ylabel(r'$\beta\;[\text{bar}^{-1}]$')
    ax[1][1].set_title('Isothermal compressibility')
    fig.suptitle('Isobars')
    fig.legend(loc='outside right center')
    fig.savefig(output_dir / 'isobars.png', **_SAVEFIG_PARAMS)

    print(f'Isobars plotted in {elapsed_time(time_start_1)}.')

    print(f'Program complete in {elapsed_time(time_start_0)}.')

#=============================================================================
# Run from src

if __name__ == "__main__":
    main()