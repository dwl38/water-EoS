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
    output_dir = Path('test_logs/IAPWS-95')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng()


def test_iapws(logging_dir: Path, rng: np.random.Generator):

    N_PLOT_POINTS = 200
    T_MIN = 221.5
    T_MAX = 383.5
    V_MIN = 24.9628125
    V_MAX = 33.4775625

    ISOCHORE_DENSITIES = [920.0, 960.0, 1000.0, 1040.0, 1080.0, 1120.0, 1160.0]
    ISOCHORE_MIN_TEMP = 210.0
    ISOCHORE_MAX_TEMP = 390.0

    ISOTHERM_TEMPS = [210.0, 240.0, 270.0, 300.0, 330.0, 360.0, 390.0]
    ISOTHERM_MIN_PRESS = -2000.0
    ISOTHERM_MAX_PRESS = 4000.0

    ISOBAR_PRESSURES = [-1200.0, -600.0, 0.0, 1000.0, 2000.0, 3000.0, 4000.0]
    ISOBAR_MIN_TEMP = 210.0
    ISOBAR_MAX_TEMP = 390.0

    DENSITY_PLOT_MIN = 880.0
    DENSITY_PLOT_MAX = 1180.0
    PRESSURE_PLOT_MIN = -2000.0
    PRESSURE_PLOT_MAX = 10000.0
    HEAT_CAPACITY_MAX = 0.0017
    COMPRESSIBILITY_MAX = 0.00015
    THERMAL_PRESSURE_MAX = 35.0
    THERMAL_EXPANSION_MAX = 0.001

    SAVEFIG_PARAMS = {'dpi': 300.0, 'bbox_inches': 'tight', 'pad_inches': 0.05}

    fes = FreeEnergySurface('iapws')
    fes.save_to_file(logging_dir / 'read_write.yaml')
    new_fes = FreeEnergySurface('bspline')
    new_fes.load_from_file(logging_dir / 'read_write.yaml')

    log_file = open(logging_dir / 'log.txt', 'w')
    time_start_0 = time.time()

    test_V = rng.uniform(V_MIN, V_MAX, 500)
    test_T = rng.uniform(T_MIN, T_MAX, 500)
    assert np.allclose(fes.pressure(test_V, test_T), new_fes.pressure(test_V, test_T))
    assert np.allclose(fes.internal_energy(test_V, test_T), new_fes.internal_energy(test_V, test_T))
    assert np.allclose(fes.compressibility(test_V, test_T), new_fes.compressibility(test_V, test_T))
    assert np.allclose(fes.heat_capacity(test_V, test_T), new_fes.heat_capacity(test_V, test_T))
    
    log_file.write(f'Read/write test completed in {elapsed_time(time_start_0)}.\n')
    time_start_1 = time.time()

    T_plot, V_plot = np.meshgrid(np.linspace(T_MIN, T_MAX, N_PLOT_POINTS), np.linspace(V_MIN, V_MAX, N_PLOT_POINTS))
    T_plot, V_plot = T_plot.ravel(), V_plot.ravel()
    F_plot = fes.free_energy(V_plot, T_plot)
    F_min, F_max = np.min(F_plot), np.max(F_plot)

    fig, ax = plt.subplots()
    fig.set_size_inches(6.0, 4.0)
    art = ax.tricontourf(T_plot, V_plot, F_plot, levels=N_PLOT_POINTS, cmap='viridis', vmin=F_min, vmax=F_max)
    cbar = fig.colorbar(art, ax=ax)
    cbar.set_label(r'$F(T,V)\;[\text{eV}]$')
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(V_MIN, V_MAX)
    ax.set_xlabel(r'$T\;[\text{K}]$')
    ax.set_ylabel(r'$V\;[\AA^{3}]$')
    ax.set_title('Free energy against temperature and volume')
    fig.savefig(logging_dir / 'free-energy-surface.png', **SAVEFIG_PARAMS)

    log_file.write(f'Plotted free energies in {elapsed_time(time_start_1)}.\n')
    time_start_1 = time.time()

    E_plot = fes.internal_energy(V_plot, T_plot)
    E_min, E_max = np.min(E_plot), np.max(E_plot)

    fig, ax = plt.subplots()
    fig.set_size_inches(6.0, 4.0)
    art = ax.tricontourf(T_plot, V_plot, E_plot, levels=N_PLOT_POINTS, cmap='viridis', vmin=E_min, vmax=E_max)
    cbar = fig.colorbar(art, ax=ax)
    cbar.set_label(r'$E(T,V)\;[\text{eV}]$')
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(V_MIN, V_MAX)
    ax.set_xlabel(r'$T\;[\text{K}]$')
    ax.set_ylabel(r'$V\;[\AA^{3}]$')
    ax.set_title('Total internal energy against temperature and volume')
    fig.savefig(logging_dir / 'internal-energy.png', **SAVEFIG_PARAMS)

    log_file.write(f'Plotted internal energies in {elapsed_time(time_start_1)}.\n')
    time_start_1 = time.time()

    p_plot = fes.pressure(V_plot, T_plot)
    p_max = min(np.max(np.abs(p_plot)), PRESSURE_PLOT_MAX)
    cmap = plt.colormaps['berlin'].with_extremes(under=[1.0, 1.0, 1.0], over=[1.0, 1.0, 1.0])
    norm = AsinhNorm(1.0, -p_max, p_max)

    fig, ax = plt.subplots()
    fig.set_size_inches(6.0, 4.0)
    art = ax.tricontourf(T_plot, V_plot, p_plot, levels=np.linspace(-p_max, p_max, 200), cmap=cmap, norm=norm)
    cbar = fig.colorbar(art, ax=ax, extend='both')
    cbar.set_label(r'$p(T,V)\;[\text{bar}]$')
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(V_MIN, V_MAX)
    ax.set_xlabel(r'$T\;[\text{K}]$')
    ax.set_ylabel(r'$V\;[\AA^{3}]$')
    ax.set_title('Pressure against temperature and volume')
    fig.savefig(logging_dir / 'pressure.png', **SAVEFIG_PARAMS)

    log_file.write(f'Plotted pressures in {elapsed_time(time_start_1)}.\n')
    time_start_1 = time.time()

    T_range = np.linspace(ISOCHORE_MIN_TEMP, ISOCHORE_MAX_TEMP, N_PLOT_POINTS)
    x = np.linspace(0.0, 1.0, len(ISOCHORE_DENSITIES))
    colors = np.c_[x, 0.75 - (3.0 * ((x - 0.5)**2)), 1.0 - x]

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(9.0, 7.0)
    for i, density in enumerate(ISOCHORE_DENSITIES):
        volume = WATER_MASS_DENSITY_CONVERSION / density
        p_range = fes.pressure(volume, T_range)
        Cv_range = fes.heat_capacity(volume, T_range)
        gamma_range = fes.thermal_pressure_coefficient(volume, T_range)
        kappa_range = fes.compressibility(volume, T_range)
        ax[0][0].plot(T_range, p_range, '-', color=colors[i], label=r'$\rho\,=\,' + f'{density:.0f}' + r'\,\text{kg}/\text{m}^3$')
        ax[0][1].plot(T_range, Cv_range, '-', color=colors[i])
        ax[1][0].plot(T_range, gamma_range, '-', color=colors[i])
        ax[1][1].plot(T_range, kappa_range, '-', color=colors[i])
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlim(ISOCHORE_MIN_TEMP, ISOCHORE_MAX_TEMP)
            ax[i][j].set_xlabel(r'$T\;[\text{K}]$')
    ax[0][0].set_ylim(PRESSURE_PLOT_MIN, PRESSURE_PLOT_MAX)
    ax[0][0].set_ylabel(r'$p\;[\text{bar}]$')
    ax[0][0].set_title('Pressure')
    ax[0][1].set_ylim(0.0, HEAT_CAPACITY_MAX)
    ax[0][1].set_ylabel(r'$C_{V}\;[\text{eV}/\text{K}]$')
    ax[0][1].set_title('Isochoric heat capacity')
    ax[1][0].set_ylim(-THERMAL_PRESSURE_MAX, THERMAL_PRESSURE_MAX)
    ax[1][0].set_ylabel(r'$\gamma\;[\text{bar}/\text{K}]$')
    ax[1][0].set_title('Isochoric thermal pressure coefficient')
    ax[1][1].set_ylim(0.0, COMPRESSIBILITY_MAX)
    ax[1][1].set_ylabel(r'$\beta\;[\text{bar}^{-1}]$')
    ax[1][1].set_title('Isothermal compressibility')
    fig.suptitle('Isochores')
    fig.legend(loc='outside right center')
    fig.savefig(logging_dir / 'isochores.png', **SAVEFIG_PARAMS)

    log_file.write(f'Plotted isochores in {elapsed_time(time_start_1)}.\n')
    time_start_1 = time.time()

    x = np.linspace(0.0, 1.0, len(ISOTHERM_TEMPS))
    colors = np.c_[x, 0.75 - (3.0 * ((x - 0.5)**2)), 1.0 - x]

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(9.0, 7.0)
    for i, temp in enumerate(ISOTHERM_TEMPS):
        p_range = fes.pressure(np.linspace(V_MIN, V_MAX, N_PLOT_POINTS), temp)
        p_min = max(np.min(p_range), ISOTHERM_MIN_PRESS)
        p_max = min(np.max(p_range), ISOTHERM_MAX_PRESS)
        p_range = np.linspace(p_min, p_max, N_PLOT_POINTS)
        start_idx = np.argmin(np.abs(p_range))
        volumes = np.full(N_PLOT_POINTS, AMBIENT_VOLUME)
        for j in range(start_idx, N_PLOT_POINTS):
            volumes[j:] = fes.volume_at_p(p_range[j], temp, init_guess=volumes[j])
            if np.isnan(volumes[j]):
                break
        for j in range(start_idx, 0, -1):
            volumes[:j] = fes.volume_at_p(p_range[j-1], temp, init_guess=volumes[j-1])
            if np.isnan(volumes[j-1]):
                break
        densities = WATER_MASS_DENSITY_CONVERSION / volumes
        gamma_range = fes.thermal_pressure_coefficient(volumes, temp)
        beta_range = fes.thermal_expansion_coefficient(volumes, temp)
        kappa_range = fes.compressibility(volumes, temp)
        ax[0][0].plot(p_range, densities, '-', color=colors[i], label=r'$T\,=\,' + f'{temp:.0f}' + r'\,\text{K}$')
        ax[0][1].plot(p_range, gamma_range, '-', color=colors[i])
        ax[1][0].plot(p_range, beta_range, '-', color=colors[i])
        ax[1][1].plot(p_range, kappa_range, '-', color=colors[i])
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlim(ISOTHERM_MIN_PRESS, ISOTHERM_MAX_PRESS)
            ax[i][j].set_xlabel(r'$p\;[\text{bar}]$')
    ax[0][0].set_ylim(DENSITY_PLOT_MIN, DENSITY_PLOT_MAX)
    ax[0][0].set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
    ax[0][0].set_title('Density')
    ax[0][1].set_ylim(-THERMAL_PRESSURE_MAX, THERMAL_PRESSURE_MAX)
    ax[0][1].set_ylabel(r'$\gamma\;[\text{bar}/\text{K}]$')
    ax[0][1].set_title('Isochoric thermal pressure coefficient')
    ax[1][0].set_ylim(-THERMAL_EXPANSION_MAX, THERMAL_EXPANSION_MAX)
    ax[1][0].set_ylabel(r'$\alpha\;[\text{K}^{-1}]$')
    ax[1][0].set_title('Isobaric thermal expansion coefficient')
    ax[1][1].set_ylim(0.0, COMPRESSIBILITY_MAX)
    ax[1][1].set_ylabel(r'$\beta\;[\text{bar}^{-1}]$')
    ax[1][1].set_title('Isothermal compressibility')
    fig.suptitle('Isotherms')
    fig.legend(loc='outside right center')
    fig.savefig(logging_dir / 'isotherms.png', **SAVEFIG_PARAMS)

    log_file.write(f'Plotted isotherms in {elapsed_time(time_start_1)}.\n')
    time_start_1 = time.time()

    T_range = np.linspace(ISOBAR_MIN_TEMP, ISOBAR_MAX_TEMP, N_PLOT_POINTS)
    x = np.linspace(0.0, 1.0, len(ISOBAR_PRESSURES))
    colors = np.c_[x, 0.75 - (3.0 * ((x - 0.5)**2)), 1.0 - x]

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(9.0, 7.0)
    for i, press in enumerate(ISOBAR_PRESSURES):
        start_idx = np.argmin(np.abs(T_range - AMBIENT_TEMPERATURE))
        init_volume = fes.volume_at_p(press, AMBIENT_TEMPERATURE, init_guess=AMBIENT_VOLUME)
        volumes = np.full(N_PLOT_POINTS, init_volume)
        for j in range(start_idx, N_PLOT_POINTS):
            volumes[j:] = fes.volume_at_p(press, T_range[j], init_guess=volumes[j])
            if np.isnan(volumes[j]):
                break
        for j in range(start_idx, 0, -1):
            volumes[:j] = fes.volume_at_p(press, T_range[j-1], init_guess=volumes[j-1])
            if np.isnan(volumes[j-1]):
                break
        densities = WATER_MASS_DENSITY_CONVERSION / volumes
        kappa_range = fes.compressibility(volumes, T_range)
        gamma_range = fes.thermal_pressure_coefficient(volumes, T_range)
        beta_range = fes.thermal_expansion_coefficient(volumes, T_range)
        Cp_range = fes.isobaric_heat_capacity(volumes, T_range)
        ax[0][0].plot(T_range, densities, '-', color=colors[i], label=r'$p\,=\,' + f'{press:.0f}' + r'\,\text{bar}$')
        ax[0][1].plot(T_range, Cp_range, '-', color=colors[i])
        ax[1][0].plot(T_range, beta_range, '-', color=colors[i])
        ax[1][1].plot(T_range, kappa_range, '-', color=colors[i])
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlim(ISOBAR_MIN_TEMP, ISOBAR_MAX_TEMP)
            ax[i][j].set_xlabel(r'$T\;[\text{K}]$')
    ax[0][0].set_ylim(DENSITY_PLOT_MIN, DENSITY_PLOT_MAX)
    ax[0][0].set_ylabel(r'$\rho\;[\text{kg}/\text{m}^3]$')
    ax[0][0].set_title('Density')
    ax[0][1].set_ylim(0.0, HEAT_CAPACITY_MAX)
    ax[0][1].set_ylabel(r'$C_{p}\;[\text{eV}/\text{K}]$')
    ax[0][1].set_title('Isobaric heat capacity')
    ax[1][0].set_ylim(-THERMAL_EXPANSION_MAX, THERMAL_EXPANSION_MAX)
    ax[1][0].set_ylabel(r'$\alpha\;[\text{K}^{-1}]$')
    ax[1][0].set_title('Isobaric thermal expansion coefficient')
    ax[1][1].set_ylim(0.0, COMPRESSIBILITY_MAX)
    ax[1][1].set_ylabel(r'$\beta\;[\text{bar}^{-1}]$')
    ax[1][1].set_title('Isothermal compressibility')
    fig.suptitle('Isobars')
    fig.legend(loc='outside right center')
    fig.savefig(logging_dir / 'isobars.png', **SAVEFIG_PARAMS)

    log_file.write(f'Plotted isobars in {elapsed_time(time_start_1)}.\n')
    log_file.write(f'All tests completed in {elapsed_time(time_start_0)}.\n')
    log_file.close()


