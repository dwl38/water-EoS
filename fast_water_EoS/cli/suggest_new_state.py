#! /usr/bin/env python

_PROG_DESC_HEADER = '''
==============================================================================
 This program takes in the same input data as `build_equation_of_state`, as
 well as the Equation of State fitting by said program, and calculates the
 location of greatest uncertainty where adding new simulation data would be
 most useful. Use as:

 >    python suggest_new_state.py <input_file> <EoS_file> [-o <output_dir>]

 where `input_file` is the same as with `build_equation_of_state`, while
 `EoS_file` is the YAML file produced by `build_equation_of_state`.

==============================================================================
'''

import time
import argparse
import numpy as np
from pathlib import Path

from fast_water_EoS.io import read_file
from fast_water_EoS.misc import elapsed_time
from fast_water_EoS.free_energy_surface import FreeEnergySurface
from fast_water_EoS.constants import PRESSURE_CONVERSION, K_B

_SCAN_POINTS = 50
_V_TOL = 0.002
_T_TOL = 0.02

def main() -> None:

    #-------------------------------------------------------------------------
    # Generate program description and parse input arguments

    prog_desc = ''
    for line in _PROG_DESC_HEADER.splitlines()[2:-1]:
        prog_desc += line + '\n'

    parser = argparse.ArgumentParser(
        prog='suggest_new_state',
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
    # Read input files

    time_start = time.time()

    data = read_file(args.input_file)
    V_obs, T_obs, *_ = data
    fes = FreeEnergySurface()
    fes.load_from_file(args.EoS_file)

    #-------------------------------------------------------------------------
    # Iteratively find maximum uncertainty

    V_min, V_max = np.min(V_obs), np.max(V_obs)
    T_min, T_max = np.min(T_obs), np.max(T_obs)
    
    while ((V_max - V_min) > _V_TOL) or ((T_max - T_min) > _T_TOL):

        V_range = np.linspace(V_min, V_max, _SCAN_POINTS)
        V_delta = (V_max - V_min) / _SCAN_POINTS
        T_range = np.linspace(T_min, T_max, _SCAN_POINTS)
        T_delta = (T_max - T_min) / _SCAN_POINTS
        V_range, T_range = np.meshgrid(V_range, T_range)
        V_range, T_range = V_range.ravel(), T_range.ravel()

        xdfdx_unc = (fes.pressure_unc(V_range, T_range) * V_range
                     / (PRESSURE_CONVERSION * K_B * T_range))
        ydfdy_unc = (fes.internal_energy_unc(V_range, T_range)
                     / (K_B * T_range))
        unc_surface = np.sqrt((xdfdx_unc**2) + (ydfdy_unc**2))

        idx = np.argmax(unc_surface)
        maximal_unc_V, maximal_unc_T = V_range[idx], T_range[idx]
        V_min = maximal_unc_V - (2.0 * V_delta)
        V_max = maximal_unc_V + (2.0 * V_delta)
        T_min = maximal_unc_T - (2.0 * T_delta)
        T_max = maximal_unc_T + (2.0 * T_delta)

    delta_p = fes.pressure_unc(maximal_unc_V, maximal_unc_T)
    delta_U = fes.internal_energy_unc(maximal_unc_V, maximal_unc_T)
    xdfdx_unc = (delta_p * maximal_unc_V
                 / (PRESSURE_CONVERSION * K_B * maximal_unc_T))
    ydfdy_unc = delta_U / (K_B * maximal_unc_T)
    with open(output_dir / 'uncertainty.txt', 'w') as fp:
        fp.write('Maximal uncertainty at:\n')
        fp.write(f'    V = {maximal_unc_V} Å³\n')
        fp.write(f'    T = {maximal_unc_T} K\n')
        fp.write('with corresponding uncertainties:\n')
        fp.write(f'    δp = {delta_p} bar\n')
        fp.write(f'    δU = {delta_U} eV\n')
        fp.write(f'    δ(x⋅df/dx) = {xdfdx_unc}\n')
        fp.write(f'    δ(y⋅df/dy) = {ydfdy_unc}\n')

    print(f'Program completed in {elapsed_time(time_start)}.')

#=============================================================================
# Run from src

if __name__ == "__main__":
    main()