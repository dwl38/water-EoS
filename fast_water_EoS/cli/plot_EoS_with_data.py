#! /usr/bin/env python

_PROG_DESC_HEADER = '''
==============================================================================
 This program takes in the same input data as `build_equation_of_state`, as
 well as the Equation of State fitting by said program, and regenerates the
 plots but with axes rescaled to the data. Use as:

 >    python plot_EoS_with_data.py <input_file> <EoS_file> [-o <output_dir>]

 where `input_file` is the same as with `build_equation_of_state`, while
 `EoS_file` is the YAML file produced by `build_equation_of_state`.

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
    AMBIENT_VOLUME,
)
from fast_water_EoS.cli.build_equation_of_state import (
    _N_PLOT_POINTS,
    _SAVEFIG_PARAMS,
)

_FILE_PREFIX = 'plot-w-data-'

def main() -> None:

    #-------------------------------------------------------------------------
    # Generate program description and parse input arguments

    prog_desc = ''
    for line in _PROG_DESC_HEADER.splitlines()[2:-1]:
        prog_desc += line + '\n'

    parser = argparse.ArgumentParser(
        prog='plot_EoS_with_data',
        description=prog_desc,
        usage='%(prog)s input_file EoS_file [-o <output_dir>]',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'input_file',
        help='input file to read data from',
    )
    parser.add_argument(
        'EoS_file',
        help='file containing EoS model parameters',
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
    if not Path(args.EoS_file).is_file():
        raise RuntimeError(f'File "{args.EoS_file}" not found.')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #-------------------------------------------------------------------------
    # Read input file and load free energy surface

    time_start_0 = time.time()

    data = read_file(args.input_file)
    V_obs = data[0]
    T_obs = data[1]
    p_obs = data[2]
    p_obs_err = data[3]
    U_obs = data[4]
    U_obs_err = data[5]
    Cv_obs = data[6]
    Cv_obs_err = data[7]
    rho_obs = WATER_MASS_DENSITY_CONVERSION / V_obs

    density_range = [np.min(rho_obs), np.max(rho_obs)]
    temp_range = [np.min(T_obs), np.max(T_obs)]

    fes = FreeEnergySurface()
    fes.load_from_file(args.EoS_file)

    #-------------------------------------------------------------------------
    # Plot internal energy surface

    time_start_1 = time.time()

    rho_range = np.linspace(*density_range, _N_PLOT_POINTS)
    V_range = WATER_MASS_DENSITY_CONVERSION / rho_range
    T_range = np.linspace(*temp_range, _N_PLOT_POINTS)
    V_plot, T_plot = np.meshgrid(V_range, T_range)
    V_plot, T_plot = V_plot.ravel(), T_plot.ravel()
    rho_plot = WATER_MASS_DENSITY_CONVERSION / V_plot

    mean_surface = fes.internal_energy(V_plot, T_plot, lazy=False)
    unc_surface = fes.internal_energy_unc(V_plot, T_plot)
    vmin = min(np.nanmin(mean_surface), np.min(U_obs))
    vmax = max(np.nanmax(mean_surface), np.max(U_obs))
    levels = np.linspace(vmin, vmax, _N_PLOT_POINTS)
    errmax = max(np.nanmax(unc_surface), np.max(U_obs_err))
    errlevels = np.linspace(0.0, errmax, _N_PLOT_POINTS)

    deviations = U_obs - fes.internal_energy(V_obs, T_obs, lazy=False)
    mae = SI_prefix(np.mean(np.abs(deviations)), 'eV')
    rmse = SI_prefix(np.sqrt(np.mean(deviations**2)), 'eV')

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12.0, 4.0)
    mask = np.logical_not(np.isnan(mean_surface))
    art = ax[0].tricontourf(T_plot[mask], rho_plot[mask], mean_surface[mask],
                            levels=levels, cmap='viridis', vmin=vmin,
                            vmax=vmax)
    ax[0].scatter(T_obs, rho_obs, c=U_obs, cmap='viridis', vmin=vmin,
                  vmax=vmax, ec='white')
    fig.colorbar(art, ax=ax[0], label=r'$E(T,\rho)\;[\text{eV}]$')
    mask = np.logical_not(np.isnan(unc_surface))
    art = ax[1].tricontourf(T_plot[mask], rho_plot[mask], unc_surface[mask],
                            levels=errlevels, cmap='Reds', vmin=0.0,
                            vmax=errmax)
    ax[1].scatter(T_obs, rho_obs, c=U_obs_err, cmap='Reds', vmin=0.0,
                  vmax=errmax, ec='b')
    fig.colorbar(art, ax=ax[1], label=r'$\sigma_{E}(T,\rho)\;[\text{eV}]$')
    for i in range(2):
        ax[i].set_xlim(*temp_range)
        ax[i].set_ylim(*density_range)
        ax[i].set_xlabel(r'$T\;[\text{K}]$')
        ax[i].set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
    ax[0].set_title(f'Fitted (MAE = {mae}, RMSE = {rmse})')
    ax[1].set_title(f'Uncertainty')
    fig.suptitle('Total internal energy against temperature and density')
    fig.tight_layout()
    fig.savefig(output_dir / (_FILE_PREFIX + 'internal-energy.png'),
                **_SAVEFIG_PARAMS)
    plt.close(fig)

    print(f'Internal energy surface plotted in {elapsed_time(time_start_1)}.')

    #-------------------------------------------------------------------------
    # Plot pressure surface

    time_start_1 = time.time()

    mean_surface = fes.pressure(V_plot, T_plot, lazy=False)
    unc_surface = fes.pressure_unc(V_plot, T_plot)
    vmax = max(np.nanmax(np.abs(mean_surface)), np.max(np.abs(p_obs)))
    norm = AsinhNorm(1.0, -vmax, vmax)
    levels = np.asarray(norm.inverse(np.linspace(0.0, 1.0, _N_PLOT_POINTS)))
    errmax = max(np.nanmax(unc_surface), np.max(p_obs_err))
    errlevels = np.linspace(0.0, errmax, _N_PLOT_POINTS)

    deviations = p_obs - fes.pressure(V_obs, T_obs, lazy=False)
    mae = SI_prefix(np.mean(np.abs(deviations)), 'bar')
    rmse = SI_prefix(np.sqrt(np.mean(deviations**2)), 'bar')

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12.0, 4.0)
    mask = np.logical_not(np.isnan(mean_surface))
    art = ax[0].tricontourf(T_plot[mask], rho_plot[mask], mean_surface[mask],
                            levels=levels, cmap='berlin', norm=norm)
    ax[0].scatter(T_obs, rho_obs, c=p_obs, cmap='berlin', norm=norm,
                  ec='lime')
    fig.colorbar(art, ax=ax[0], label=r'$p(T,\rho)\;[\text{bar}]$')
    mask = np.logical_not(np.isnan(unc_surface))
    art = ax[1].tricontourf(T_plot[mask], rho_plot[mask], unc_surface[mask],
                            levels=errlevels, cmap='Reds', vmin=0.0,
                            vmax=errmax)
    ax[1].scatter(T_obs, rho_obs, c=p_obs_err, cmap='Reds', vmin=0.0,
                  vmax=errmax, ec='b')
    fig.colorbar(art, ax=ax[1], label=r'$\sigma_{p}(T,\rho)\;[\text{bar}]$')
    for i in range(2):
        ax[i].set_xlim(*temp_range)
        ax[i].set_ylim(*density_range)
        ax[i].set_xlabel(r'$T\;[\text{K}]$')
        ax[i].set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
    ax[0].set_title(f'Fitted (MAE = {mae}, RMSE = {rmse})')
    ax[1].set_title(f'Uncertainty')
    fig.suptitle('Pressure against temperature and density')
    fig.tight_layout()
    fig.savefig(output_dir / (_FILE_PREFIX + 'pressure.png'),
                **_SAVEFIG_PARAMS)
    plt.close(fig)

    print(f'Pressure surface plotted in {elapsed_time(time_start_1)}.')

    #-------------------------------------------------------------------------
    # Plot heat capacity surface

    time_start_1 = time.time()

    if Cv_obs is not None:

        mean_surface = fes.heat_capacity(V_plot, T_plot, lazy=False)
        unc_surface = fes.heat_capacity_unc(V_plot, T_plot)
        vmin = max(min(np.nanmin(mean_surface), np.min(Cv_obs)), 0.0)
        vmax = max(np.nanmax(mean_surface), np.max(Cv_obs))
        levels = np.linspace(vmin, vmax, _N_PLOT_POINTS)
        errmax = max(np.nanmax(unc_surface), np.max(Cv_obs_err))
        errlevels = np.linspace(0.0, errmax, _N_PLOT_POINTS)

        deviations = Cv_obs - fes.heat_capacity(V_obs, T_obs, lazy=False)
        mae = SI_prefix(np.mean(np.abs(deviations)), 'eV/K')
        rmse = SI_prefix(np.sqrt(np.mean(deviations**2)), 'eV/K')

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(12.0, 4.0)
        mask = np.logical_not(np.isnan(mean_surface))
        art = ax[0].tricontourf(T_plot[mask], rho_plot[mask],
                                mean_surface[mask], levels=levels,
                                cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0].scatter(T_obs, rho_obs, c=Cv_obs, cmap='viridis', vmin=vmin,
                      vmax=vmax, ec='white')
        fig.colorbar(art, ax=ax[0],
                     label=r'$C_V(T,\rho)\;[\text{eV}/\text{K}]$')
        mask = np.logical_not(np.isnan(unc_surface))
        art = ax[1].tricontourf(T_plot[mask], rho_plot[mask],
                                unc_surface[mask], levels=errlevels,
                                cmap='Reds', vmin=0.0, vmax=errmax)
        ax[1].scatter(T_obs, rho_obs, c=Cv_obs_err, cmap='Reds', vmin=0.0,
                      vmax=errmax, ec='b')
        fig.colorbar(art, ax=ax[1],
                     label=r'$\sigma_{C_V}(T,\rho)\;[\text{eV}/\text{K}]$')
        for i in range(2):
            ax[i].set_xlim(*temp_range)
            ax[i].set_ylim(*density_range)
            ax[i].set_xlabel(r'$T\;[\text{K}]$')
            ax[i].set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
        ax[0].set_title(f'Fitted (MAE = {mae}, RMSE = {rmse})')
        ax[1].set_title(f'Uncertainty')
        fig.suptitle('Heat capacity against temperature and density')
        fig.tight_layout()
        fig.savefig(output_dir / (_FILE_PREFIX + 'heat-capacity.png'),
                    **_SAVEFIG_PARAMS)
        plt.close(fig)

        print('Heat capacity surface plotted in '
              f'{elapsed_time(time_start_1)}.')

    #-------------------------------------------------------------------------
    # Plot isochores

    time_start_1 = time.time()

    unique_densities = np.sort(np.unique(rho_obs))
    isochore_densities = list()
    for candidate_density in unique_densities:
        if np.sum(rho_obs == candidate_density) > 3:
            isochore_densities.append(candidate_density)

    if len(isochore_densities) > 1:

        x = np.linspace(0.0, 1.0, len(isochore_densities))
        colors = np.c_[x, 0.75 - (3.0 * ((x - 0.5)**2)), 1.0 - x]

        fig, ax = plt.subplots(1, 2, layout='constrained')
        fig.set_size_inches(9.0, 3.5)
        for density, color, zorder in zip(isochore_densities, colors, x):
            label = (r'$\rho\,=\,' + f'{density:.0f}'
                     + r'\,\text{kg}/\text{m}^3$')
            vol = WATER_MASS_DENSITY_CONVERSION / density
            p_curve = fes.pressure(vol, T_range, lazy=False)
            p_curve_err = 2.0 * fes.pressure_unc(vol, T_range)
            Cv = fes.heat_capacity(vol, T_range, lazy=False)
            Cv_err = 2.0 * fes.heat_capacity_unc(vol, T_range)
            mask = (rho_obs == density)
            ax[0].fill_between(T_range, p_curve - p_curve_err,
                               p_curve + p_curve_err, color=color, ec='none',
                               alpha=0.3, zorder=zorder)
            ax[0].plot(T_range, p_curve, '-', color=color, label=label,
                       zorder=(1.0 + zorder))
            ax[0].errorbar(T_obs[mask], p_obs[mask],
                           yerr=(2.0 * p_obs_err[mask]), fmt='o', color=color,
                           mec=color, mfc='none', zorder=(1.1 + zorder))
            ax[1].fill_between(T_range, Cv - Cv_err, Cv + Cv_err, color=color,
                               ec='none', alpha=0.3, zorder=zorder)
            ax[1].plot(T_range, Cv, '-', color=color, zorder=(1.0 + zorder))
            if Cv_obs is not None:
                ax[1].errorbar(T_obs[mask], Cv_obs[mask],
                               yerr=(2.0 * Cv_obs_err[mask]), fmt='o',
                               color=color, mec=color, mfc='none',
                               zorder=(1.1 + zorder))
        for i in range(2):
            ax[i].set_xlim(*temp_range)
            ax[i].set_xlabel(r'$T\;[\text{K}]$')
        ax[0].set_ylabel(r'$p\;[\text{bar}]$')
        ax[0].set_title('Pressure')
        ax[1].set_ylabel(r'$C_{V}\;[\text{eV}/\text{K}]$')
        ax[1].set_title('Isochoric heat capacity')
        fig.suptitle('Isochores')
        fig.legend(loc='outside right center')
        fig.savefig(output_dir / (_FILE_PREFIX + 'isochores.png'),
                    **_SAVEFIG_PARAMS)
        plt.close(fig)

        print(f'Isochores plotted in {elapsed_time(time_start_1)}.')

    #-------------------------------------------------------------------------
    # Plot isotherms

    time_start_1 = time.time()

    unique_temps = np.sort(np.unique(T_obs))
    isotherm_temps = list()
    for candidate_temp in unique_temps:
        if np.sum(T_obs == candidate_temp) > 3:
            isotherm_temps.append(candidate_temp)

    if len(isotherm_temps) > 1:

        x = np.linspace(0.0, 1.0, len(isotherm_temps))
        colors = np.c_[x, 0.75 - (3.0 * ((x - 0.5)**2)), 1.0 - x]

        fig, ax = plt.subplots(1, 2, layout='constrained')
        fig.set_size_inches(9.0, 3.5)
        for temp, color, zorder in zip(isotherm_temps, colors, x):
            label = r'$T\,=\,' + f'{temp:.0f}' + r'\,\text{K}$'
            p_range = fes.pressure(V_range, temp)
            if np.all(np.isnan(p_range)):
                continue
            p_min = np.nanmin(p_range) + 1.0
            p_max = np.nanmax(p_range) - 1.0
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
            beta = fes.compressibility(volumes, T_range, lazy=False)
            p_unc = 2.0 * fes.pressure_unc(volumes, T_range)
            volumes_err = np.abs(volumes * beta * p_unc)
            densities_err = (WATER_MASS_DENSITY_CONVERSION * volumes_err
                             / (volumes**2))
            Cv = fes.heat_capacity(volumes, T_range, lazy=False)
            Cv_err = 2.0 * fes.heat_capacity_unc(volumes, T_range)
            mask = (T_obs == temp)
            ax[0].fill_between(p_range, densities - densities_err,
                               densities + densities_err, color=color,
                               ec='none', alpha=0.3, zorder=zorder)
            ax[0].plot(p_range, densities, '-', color=color, label=label,
                       zorder=(1.0 + zorder))
            ax[0].errorbar(p_obs[mask], rho_obs[mask],
                           xerr=(2.0 * p_obs_err[mask]), fmt='o', color=color,
                           mec=color, mfc='none', zorder=(1.1 + zorder))
            ax[1].fill_between(p_range, Cv - Cv_err, Cv + Cv_err, color=color,
                               ec='none', alpha=0.3, zorder=zorder)
            ax[1].plot(p_range, Cv, '-', color=color, zorder=(1.0 + zorder))
            if Cv_obs is not None:
                ax[1].errorbar(p_obs[mask], Cv_obs[mask],
                               xerr=(2.0 * p_obs_err[mask]),
                               yerr=(2.0 * Cv_obs_err[mask]), fmt='o',
                               color=color, mec=color, mfc='none',
                               zorder=(1.1 + zorder))
        for i in range(2):
            ax[i].set_xlabel(r'$p\;[\text{bar}]$')
        ax[0].set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
        ax[0].set_title('Density')
        ax[1].set_ylabel(r'$C_{V}\;[\text{eV}/\text{K}]$')
        ax[1].set_title('Isochoric heat capacity')
        fig.suptitle('Isotherms')
        fig.legend(loc='outside right center')
        fig.savefig(output_dir / (_FILE_PREFIX + 'isotherms.png'),
                    **_SAVEFIG_PARAMS)
        plt.close(fig)

        print(f'Isotherms plotted in {elapsed_time(time_start_1)}.')

    print(f'Program complete in {elapsed_time(time_start_0)}.')

#=============================================================================
# Run from src

if __name__ == "__main__":
    main()