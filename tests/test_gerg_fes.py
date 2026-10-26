import pytest
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import AsinhNorm
from fast_water_EoS.free_energy_surface import FreeEnergySurface
from fast_water_EoS.constants import WATER_MASS_DENSITY_CONVERSION, AMBIENT_VOLUME, AMBIENT_TEMPERATURE
from fast_water_EoS.misc import elapsed_time


@pytest.fixture
def logging_dir() -> Path:
    output_dir = Path('test_logs/GERG-2008')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng()


def test_gerg(logging_dir: Path, rng: np.random.Generator):

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

    fes = FreeEnergySurface('gerg')
    fes.save_to_file(logging_dir / 'read_write.yaml')
    new_fes = FreeEnergySurface('bspline')
    new_fes.load_from_file(logging_dir / 'read_write.yaml')

    log_file = open(logging_dir / 'log.txt', 'w')
    time_start_0 = time.time()

    test_rho = rng.uniform(*_DENSITY_RANGE, 500)
    test_V = WATER_MASS_DENSITY_CONVERSION / test_rho
    test_T = rng.uniform(*_TEMP_RANGE, 500)
    assert np.allclose(fes.pressure(test_V, test_T), new_fes.pressure(test_V, test_T))
    assert np.allclose(fes.internal_energy(test_V, test_T), new_fes.internal_energy(test_V, test_T))
    assert np.allclose(fes.compressibility(test_V, test_T), new_fes.compressibility(test_V, test_T))
    assert np.allclose(fes.heat_capacity(test_V, test_T), new_fes.heat_capacity(test_V, test_T))
    
    log_file.write(f'Read/write test completed in {elapsed_time(time_start_0)}.\n')
    time_start_1 = time.time()

    rho_range = np.linspace(*_DENSITY_RANGE, _N_PLOT_POINTS)
    V_range = WATER_MASS_DENSITY_CONVERSION / rho_range
    T_range = np.linspace(*_TEMP_RANGE, _N_PLOT_POINTS)
    V_plot, T_plot = np.meshgrid(V_range, T_range)
    V_plot, T_plot = V_plot.ravel(), T_plot.ravel()
    rho_plot = WATER_MASS_DENSITY_CONVERSION / V_plot

    E_plot = fes.internal_energy(V_plot, T_plot)
    E_min, E_max = np.min(E_plot), np.max(E_plot)
    levels = np.linspace(E_min, E_max, _N_PLOT_POINTS)

    fig, ax = plt.subplots()
    fig.set_size_inches(6.0, 4.0)
    art = ax.tricontourf(T_plot, rho_plot, E_plot, levels=levels, cmap='viridis', vmin=E_min, vmax=E_max)
    cbar = fig.colorbar(art, ax=ax)
    cbar.set_label(r'$E(T,\rho)\;[\text{eV}]$')
    ax.set_xlim(*_TEMP_RANGE)
    ax.set_ylim(*_DENSITY_RANGE)
    ax.set_xlabel(r'$T\;[\text{K}]$')
    ax.set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
    ax.set_title('Total internal energy against temperature and density')
    fig.savefig(logging_dir / 'internal-energy.png', **_SAVEFIG_PARAMS)

    log_file.write(f'Plotted internal energies in {elapsed_time(time_start_1)}.\n')
    time_start_1 = time.time()

    p_plot = fes.pressure(V_plot, T_plot)
    p_max = min(np.max(np.abs(p_plot)), 10000.0)
    norm = AsinhNorm(1.0, -p_max, p_max)
    levels = np.asarray(norm.inverse(np.linspace(0.0, 1.0, _N_PLOT_POINTS)))

    fig, ax = plt.subplots()
    fig.set_size_inches(6.0, 4.0)
    art = ax.tricontourf(T_plot, rho_plot, p_plot, levels=levels, cmap='berlin', norm=norm)
    cbar = fig.colorbar(art, ax=ax, extend='both')
    cbar.set_label(r'$p(T,\rho)\;[\text{bar}]$')
    ax.set_xlim(*_TEMP_RANGE)
    ax.set_ylim(*_DENSITY_RANGE)
    ax.set_xlabel(r'$T\;[\text{K}]$')
    ax.set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
    ax.set_title('Pressure against temperature and density')
    fig.savefig(logging_dir / 'pressure.png', **_SAVEFIG_PARAMS)

    log_file.write(f'Plotted pressures in {elapsed_time(time_start_1)}.\n')
    time_start_1 = time.time()

    x = np.linspace(0.0, 1.0, len(_ISOCHORE_DENSITIES))
    colors = np.c_[x, 0.75 - (3.0 * ((x - 0.5)**2)), 1.0 - x]

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(9.0, 7.0)
    for density, color in zip(_ISOCHORE_DENSITIES, colors):
        label = r'$\rho\,=\,' + f'{density:.0f}' + r'\,\text{kg}/\text{m}^3$'
        vol = WATER_MASS_DENSITY_CONVERSION / density
        p_curve = fes.pressure(vol, T_range)
        Cv = fes.heat_capacity(vol, T_range)
        gamma = fes.thermal_pressure_coefficient(vol, T_range)
        beta = fes.compressibility(vol, T_range)
        ax[0][0].plot(T_range, p_curve, '-', color=color, label=label)
        ax[0][1].plot(T_range, Cv, '-', color=color)
        ax[1][0].plot(T_range, gamma, '-', color=color)
        ax[1][1].plot(T_range, beta, '-', color=color)
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
    fig.savefig(logging_dir / 'isochores.png', **_SAVEFIG_PARAMS)

    log_file.write(f'Plotted isochores in {elapsed_time(time_start_1)}.\n')
    time_start_1 = time.time()

    x = np.linspace(0.0, 1.0, len(_ISOTHERM_TEMPS))
    colors = np.c_[x, 0.75 - (3.0 * ((x - 0.5)**2)), 1.0 - x]

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(9.0, 7.0)
    for temp, color in zip(_ISOTHERM_TEMPS, colors):
        label = r'$T\,=\,' + f'{temp:.0f}' + r'\,\text{K}$'
        p_range = fes.pressure(V_range, temp)
        p_min = np.clip(np.min(p_range) + 1.0, *_PRESSURE_RANGE)
        p_max = np.clip(np.max(p_range) - 1.0, *_PRESSURE_RANGE)
        p_range = np.linspace(p_min, p_max, _N_PLOT_POINTS)
        start_idx = np.argmin(np.abs(p_range))
        volumes = np.full(_N_PLOT_POINTS, AMBIENT_VOLUME)
        for j in range(start_idx, _N_PLOT_POINTS):
            volumes[j:] = fes.volume_at_p(p_range[j], temp, init_guess=volumes[j])
            if np.isnan(volumes[j]):
                break
        for j in range(start_idx, 0, -1):
            volumes[:j] = fes.volume_at_p(p_range[j-1], temp, init_guess=volumes[j-1])
            if np.isnan(volumes[j-1]):
                break
        densities = WATER_MASS_DENSITY_CONVERSION / volumes
        gamma = fes.thermal_pressure_coefficient(volumes, T_range)
        alpha = fes.thermal_expansion_coefficient(volumes, T_range)
        beta = fes.compressibility(volumes, T_range)
        ax[0][0].plot(p_range, densities, '-', color=color, label=label)
        ax[0][1].plot(p_range, gamma, '-', color=color)
        ax[1][0].plot(p_range, alpha, '-', color=color)
        ax[1][1].plot(p_range, beta, '-', color=color)
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
    fig.savefig(logging_dir / 'isotherms.png', **_SAVEFIG_PARAMS)

    log_file.write(f'Plotted isotherms in {elapsed_time(time_start_1)}.\n')
    time_start_1 = time.time()

    x = np.linspace(0.0, 1.0, len(_ISOBAR_PRESSURES))
    colors = np.c_[x, 0.75 - (3.0 * ((x - 0.5)**2)), 1.0 - x]

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(9.0, 7.0)
    for press, color in zip(_ISOBAR_PRESSURES, colors):
        label = r'$p\,=\,' + f'{press:.0f}' + r'\,\text{bar}$'
        start_idx = np.argmin(np.abs(T_range - AMBIENT_TEMPERATURE))
        init_volume = fes.volume_at_p(press, AMBIENT_TEMPERATURE, init_guess=AMBIENT_VOLUME)
        volumes = np.full(_N_PLOT_POINTS, init_volume)
        for j in range(start_idx, _N_PLOT_POINTS):
            volumes[j:] = fes.volume_at_p(press, T_range[j], init_guess=volumes[j])
            if np.isnan(volumes[j]):
                break
        for j in range(start_idx, 0, -1):
            volumes[:j] = fes.volume_at_p(press, T_range[j-1], init_guess=volumes[j-1])
            if np.isnan(volumes[j-1]):
                break
        densities = WATER_MASS_DENSITY_CONVERSION / volumes
        Cp = fes.isobaric_heat_capacity(volumes, T_range)
        alpha = fes.thermal_expansion_coefficient(volumes, T_range)
        beta = fes.compressibility(volumes, T_range)
        ax[0][0].plot(T_range, densities, '-', color=color, label=label)
        ax[0][1].plot(T_range, Cp, '-', color=color)
        ax[1][0].plot(T_range, alpha, '-', color=color)
        ax[1][1].plot(T_range, beta, '-', color=color)
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
    fig.savefig(logging_dir / 'isobars.png', **_SAVEFIG_PARAMS)

    log_file.write(f'Plotted isobars in {elapsed_time(time_start_1)}.\n')
    log_file.write(f'All tests completed in {elapsed_time(time_start_0)}.\n')
    log_file.close()


