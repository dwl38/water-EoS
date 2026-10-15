import pytest
import yaml
import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fast_water_EoS.func import FITTING_TAGS
from fast_water_EoS.func.polynomial import Polynomial


@pytest.fixture
def logging_dir() -> Path:
    output_dir = Path('test_logs/func/polynomial')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng()


def _evaluate_and_plot(order, sample_x, sample_y, sample_f, sample_err, plot_x, plot_y, ref_f, ax, cmap, vmin, vmax):
    poly = Polynomial(order=order)
    poly.fit(sample_x, sample_y, sample_f, sample_err, 'f')
    plot_f = poly.f(plot_x, plot_y)
    ax.tricontourf(plot_x, plot_y, plot_f, levels=np.linspace(vmin, vmax, 200), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax.scatter(sample_x, sample_y, c=sample_f, ec='white', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    chi_sq = np.mean(((sample_f - poly.f(sample_x, sample_y)) / sample_err)**2)
    sq_dev = np.mean((ref_f - plot_f)**2)
    return chi_sq, sq_dev


def test_ackley(logging_dir: Path, rng: np.random.Generator):

    plot_file = logging_dir / 'ackley.png'
    plot_file.unlink(missing_ok=True)

    def true_f(x, y):
        part1 = -20.0 * np.exp(-0.14142 * np.sqrt((x**2) + (y**2)))
        part2 = np.exp((np.cos(2.0 * np.pi * x) + np.cos(2.0 * np.pi * y)) / 2.0)
        return part1 - part2 + 15.7

    PLOT_X_MIN, PLOT_X_MAX = -4.0, 4.0
    PLOT_Y_MIN, PLOT_Y_MAX = -4.0, 4.0
    SAMPLE_X_MIN, SAMPLE_X_MAX = -3.5, 3.5
    SAMPLE_Y_MIN, SAMPLE_Y_MAX = -3.5, 3.5
    SAMPLE_SIZE = 150
    ORDERS = [[0, 6], [8, 10]]

    plot_x = np.linspace(PLOT_X_MIN, PLOT_X_MAX, 201)
    plot_y = np.linspace(PLOT_Y_MIN, PLOT_Y_MAX, 201)
    plot_x, plot_y = np.meshgrid(plot_x, plot_y)
    plot_x, plot_y = plot_x.ravel(), plot_y.ravel()
    plot_f = true_f(plot_x, plot_y)
    cmap = plt.colormaps['viridis'].with_extremes(bad=(1.0, 1.0, 1.0), under=(1.0, 1.0, 1.0), over=(1.0, 1.0, 1.0))
    vmin, vmax = plot_f.min(), plot_f.max()

    sample_x = rng.uniform(SAMPLE_X_MIN, SAMPLE_X_MAX, SAMPLE_SIZE)
    sample_y = rng.uniform(SAMPLE_Y_MIN, SAMPLE_Y_MAX, SAMPLE_SIZE)
    sample_err = rng.uniform(0.1, 0.2, SAMPLE_SIZE)
    sample_f = true_f(sample_x, sample_y) + rng.normal(0.0, sample_err, SAMPLE_SIZE)

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(12.0, 10.0)
    art = ax[0][0].tricontourf(plot_x, plot_y, plot_f, levels=np.linspace(vmin, vmax, 200), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax[0][0].set_xlabel('x')
    ax[0][0].set_ylabel('y')
    ax[0][0].set_title('True')
    for i in range(2):
        for j in range(2):
            if i != 0 or j != 0:
                chi_sq, sq_dev = _evaluate_and_plot(ORDERS[i][j], sample_x, sample_y, sample_f, sample_err, plot_x, plot_y, plot_f, ax[i][j], cmap, vmin, vmax)
                ax[i][j].set_title(f'Fitted (order {ORDERS[i][j]} polynomial) [χ² = {chi_sq:g}, |Δf|² = {sq_dev:g}]')
            ax[i][j].set_aspect(1.0)
            ax[i][j].set_xlim(PLOT_X_MIN, PLOT_X_MAX)
            ax[i][j].set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)

    fig.colorbar(art, ax=ax, extend='both')
    fig.savefig(plot_file, dpi=300.0, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def test_branin(logging_dir: Path, rng: np.random.Generator):

    plot_file = logging_dir / 'branin.png'
    plot_file.unlink(missing_ok=True)

    def true_f(x, y):
        b = 5.1 / (4.0 * (np.pi**2))
        part1 = (y - (b * (x**2)) + (5.0 * x / np.pi) - 6.0)**2
        part2 = 10.0 * (1.0 - (0.125 / np.pi)) * np.cos(x)
        return np.log(part1 + part2 + 11.0)

    PLOT_X_MIN, PLOT_X_MAX = -5.0, 10.0
    PLOT_Y_MIN, PLOT_Y_MAX = 0.0, 15.0
    SAMPLE_X_MIN, SAMPLE_X_MAX = -4.5, 9.5
    SAMPLE_Y_MIN, SAMPLE_Y_MAX = 0.5, 14.5
    SAMPLE_SIZE = 150
    ORDERS = [[0, 6], [8, 10]]

    plot_x = np.linspace(PLOT_X_MIN, PLOT_X_MAX, 201)
    plot_y = np.linspace(PLOT_Y_MIN, PLOT_Y_MAX, 201)
    plot_x, plot_y = np.meshgrid(plot_x, plot_y)
    plot_x, plot_y = plot_x.ravel(), plot_y.ravel()
    plot_f = true_f(plot_x, plot_y)
    cmap = plt.colormaps['viridis'].with_extremes(bad=(1.0, 1.0, 1.0), under=(1.0, 1.0, 1.0), over=(1.0, 1.0, 1.0))
    vmin, vmax = plot_f.min(), plot_f.max()

    sample_x = rng.uniform(SAMPLE_X_MIN, SAMPLE_X_MAX, SAMPLE_SIZE)
    sample_y = rng.uniform(SAMPLE_Y_MIN, SAMPLE_Y_MAX, SAMPLE_SIZE)
    sample_err = rng.uniform(0.1, 0.2, SAMPLE_SIZE)
    sample_f = true_f(sample_x, sample_y) + rng.normal(0.0, sample_err, SAMPLE_SIZE)

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(12.0, 10.0)
    art = ax[0][0].tricontourf(plot_x, plot_y, plot_f, levels=np.linspace(vmin, vmax, 200), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax[0][0].set_xlabel('x')
    ax[0][0].set_ylabel('y')
    ax[0][0].set_title('True')
    for i in range(2):
        for j in range(2):
            if i != 0 or j != 0:
                chi_sq, sq_dev = _evaluate_and_plot(ORDERS[i][j], sample_x, sample_y, sample_f, sample_err, plot_x, plot_y, plot_f, ax[i][j], cmap, vmin, vmax)
                ax[i][j].set_title(f'Fitted (order {ORDERS[i][j]} polynomial) [χ² = {chi_sq:g}, |Δf|² = {sq_dev:g}]')
            ax[i][j].set_aspect(1.0)
            ax[i][j].set_xlim(PLOT_X_MIN, PLOT_X_MAX)
            ax[i][j].set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)

    fig.colorbar(art, ax=ax, extend='both')
    fig.savefig(plot_file, dpi=300.0, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def test_himmelblau(logging_dir: Path, rng: np.random.Generator):

    plot_file = logging_dir / 'himmelblau.png'
    plot_file.unlink(missing_ok=True)

    def true_f(x, y):
        part1 = ((x**2) + y - 11.0)**2
        part2 = (x + (y**2) - 7.0)**2
        return np.log(part1 + part2 + 1.0)

    PLOT_X_MIN, PLOT_X_MAX = -5.0, 5.0
    PLOT_Y_MIN, PLOT_Y_MAX = -5.0, 5.0
    SAMPLE_X_MIN, SAMPLE_X_MAX = -4.5, 4.5
    SAMPLE_Y_MIN, SAMPLE_Y_MAX = -4.5, 4.5
    SAMPLE_SIZE = 150
    ORDERS = [[0, 4], [6, 8]]

    plot_x = np.linspace(PLOT_X_MIN, PLOT_X_MAX, 201)
    plot_y = np.linspace(PLOT_Y_MIN, PLOT_Y_MAX, 201)
    plot_x, plot_y = np.meshgrid(plot_x, plot_y)
    plot_x, plot_y = plot_x.ravel(), plot_y.ravel()
    plot_f = true_f(plot_x, plot_y)
    cmap = plt.colormaps['viridis'].with_extremes(bad=(1.0, 1.0, 1.0), under=(1.0, 1.0, 1.0), over=(1.0, 1.0, 1.0))
    vmin, vmax = plot_f.min(), plot_f.max()

    sample_x = rng.uniform(SAMPLE_X_MIN, SAMPLE_X_MAX, SAMPLE_SIZE)
    sample_y = rng.uniform(SAMPLE_Y_MIN, SAMPLE_Y_MAX, SAMPLE_SIZE)
    sample_err = rng.uniform(0.1, 0.2, SAMPLE_SIZE)
    sample_f = true_f(sample_x, sample_y) + rng.normal(0.0, sample_err, SAMPLE_SIZE)

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(12.0, 10.0)
    art = ax[0][0].tricontourf(plot_x, plot_y, plot_f, levels=np.linspace(vmin, vmax, 200), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax[0][0].set_xlabel('x')
    ax[0][0].set_ylabel('y')
    ax[0][0].set_title('True')
    for i in range(2):
        for j in range(2):
            if i != 0 or j != 0:
                chi_sq, sq_dev = _evaluate_and_plot(ORDERS[i][j], sample_x, sample_y, sample_f, sample_err, plot_x, plot_y, plot_f, ax[i][j], cmap, vmin, vmax)
                ax[i][j].set_title(f'Fitted (order {ORDERS[i][j]} polynomial) [χ² = {chi_sq:g}, |Δf|² = {sq_dev:g}]')
            ax[i][j].set_aspect(1.0)
            ax[i][j].set_xlim(PLOT_X_MIN, PLOT_X_MAX)
            ax[i][j].set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)

    fig.colorbar(art, ax=ax, extend='both')
    fig.savefig(plot_file, dpi=300.0, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def test_sphere_grad(logging_dir: Path, rng: np.random.Generator):

    plot_file = logging_dir / 'sphere_grad.png'
    plot_file.unlink(missing_ok=True)

    def true_f(x, y):
        return np.log((x**2) + (y**2) + 1.0)
    
    def true_dfdx(x, y):
        return (2.0 * x) / ((x**2) + (y**2) + 1.0)
    
    def true_dfdy(x, y):
        return (2.0 * y) / ((x**2) + (y**2) + 1.0)

    PLOT_X_MIN, PLOT_X_MAX = -5.0, 5.0
    PLOT_Y_MIN, PLOT_Y_MAX = -5.0, 5.0
    SAMPLE_X_MIN, SAMPLE_X_MAX = -4.5, 4.5
    SAMPLE_Y_MIN, SAMPLE_Y_MAX = -4.5, 4.5
    DELTA = 0.1
    SAMPLE_SIZE = 150
    ORDERS = [[0, 4], [6, 8]]

    plot_x = np.linspace(PLOT_X_MIN, PLOT_X_MAX, 201)
    plot_y = np.linspace(PLOT_Y_MIN, PLOT_Y_MAX, 201)
    plot_x, plot_y = np.meshgrid(plot_x, plot_y)
    plot_x, plot_y = plot_x.ravel(), plot_y.ravel()
    plot_f = true_f(plot_x, plot_y)
    cmap = plt.colormaps['viridis'].with_extremes(bad=(1.0, 1.0, 1.0), under=(1.0, 1.0, 1.0), over=(1.0, 1.0, 1.0))
    vmin, vmax = plot_f.min(), plot_f.max()

    sample_x = rng.uniform(SAMPLE_X_MIN, SAMPLE_X_MAX, SAMPLE_SIZE)
    sample_y = rng.uniform(SAMPLE_Y_MIN, SAMPLE_Y_MAX, SAMPLE_SIZE)
    sample_data_tags = rng.choice(['dfdx', 'dfdy'], SAMPLE_SIZE)
    sample_data = np.empty(SAMPLE_SIZE, dtype=float)
    idx = (sample_data_tags == 'dfdx')
    sample_data[idx] = true_dfdx(sample_x[idx], sample_y[idx])
    idx = (sample_data_tags == 'dfdy')
    sample_data[idx] = true_dfdy(sample_x[idx], sample_y[idx])

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(12.0, 10.0)
    art = ax[0][0].tricontourf(plot_x, plot_y, plot_f, levels=np.linspace(vmin, vmax, 200), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax[0][0].set_title('True')
    for i in range(2):
        for j in range(2):
            if i != 0 or j != 0:
                poly = Polynomial(order=ORDERS[i][j])
                poly.fit(sample_x, sample_y, sample_data, 0.01, sample_data_tags)
                fit_f = poly.f(plot_x, plot_y)
                ax[i][j].tricontourf(plot_x, plot_y, fit_f, levels=np.linspace(vmin, vmax, 200), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
                idx = np.logical_and(sample_data_tags == 'dfdx', sample_data >= 0.0)
                ax[i][j].quiver(sample_x[idx] - DELTA, sample_y[idx], np.full_like(sample_x[idx], 2.0 * DELTA), np.zeros_like(sample_x[idx]), color='white', angles='xy')
                idx = np.logical_and(sample_data_tags == 'dfdx', sample_data < 0.0)
                ax[i][j].quiver(sample_x[idx] + DELTA, sample_y[idx], np.full_like(sample_x[idx], -2.0 * DELTA), np.zeros_like(sample_x[idx]), color='white', angles='xy')
                idx = np.logical_and(sample_data_tags == 'dfdy', sample_data >= 0.0)
                ax[i][j].quiver(sample_x[idx], sample_y[idx] - DELTA, np.zeros_like(sample_x[idx]), np.full_like(sample_x[idx], 2.0 * DELTA), color='white', angles='xy')
                idx = np.logical_and(sample_data_tags == 'dfdy', sample_data < 0.0)
                ax[i][j].quiver(sample_x[idx], sample_y[idx] + DELTA, np.zeros_like(sample_x[idx]), np.full_like(sample_x[idx], -2.0 * DELTA), color='white', angles='xy')
                ax[i][j].set_title(f'Fitted (order {ORDERS[i][j]} polynomial)')
            ax[i][j].set_xlabel('x')
            ax[i][j].set_ylabel('y')
            ax[i][j].set_aspect(1.0)
            ax[i][j].set_xlim(PLOT_X_MIN, PLOT_X_MAX)
            ax[i][j].set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)

    fig.colorbar(art, ax=ax, extend='both')
    fig.savefig(plot_file, dpi=300.0, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def test_mixed_fitting(logging_dir: Path, rng: np.random.Generator):

    plot_file = logging_dir / 'mixed_fitting.png'
    plot_file.unlink(missing_ok=True)

    def true_f(x, y):
        return np.sin(x + y) + ((x - y)**2) - (1.5 * x) + (2.5 * y)
    
    def true_dfdx(x, y):
        return np.cos(x + y) + (2.0 * (x - y)) - 1.5
    
    def true_dfdy(x, y):
        return np.cos(x + y) - (2.0 * (x - y)) + 2.5
    
    def true_ddfdxx(x, y):
        return -np.sin(x + y) + 2.0
    
    def true_ddfdxy(x, y):
        return -np.sin(x + y) - 2.0
    
    def true_ddfdyy(x, y):
        return -np.sin(x + y) + 2.0
    
    GROUND_TRUTH = {'f': true_f, 'dfdx': true_dfdx, 'dfdy': true_dfdy, 'ddfdxx': true_ddfdxx, 'ddfdxy': true_ddfdxy, 'ddfdyy': true_ddfdyy}
    PLOT_X_MIN, PLOT_X_MAX = -2.25, 4.75
    PLOT_Y_MIN, PLOT_Y_MAX = -3.0, 4.0
    SAMPLE_SIZES = [[0, 100], [200, 300]]

    plot_x = np.linspace(PLOT_X_MIN, PLOT_X_MAX, 201)
    plot_y = np.linspace(PLOT_Y_MIN, PLOT_Y_MAX, 201)
    plot_x, plot_y = np.meshgrid(plot_x, plot_y)
    plot_x, plot_y = plot_x.ravel(), plot_y.ravel()
    plot_f = true_f(plot_x, plot_y)
    cmap = plt.colormaps['viridis'].with_extremes(bad=(1.0, 1.0, 1.0), under=(1.0, 1.0, 1.0), over=(1.0, 1.0, 1.0))
    vmin, vmax = plot_f.min(), plot_f.max()

    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.set_size_inches(12.0, 10.0)
    art = ax[0][0].tricontourf(plot_x, plot_y, plot_f, levels=np.linspace(vmin, vmax, 200), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax[0][0].set_title('True')
    for i in range(2):
        for j in range(2):
            if i != 0 or j != 0:
                order = int(np.sqrt((2.0 * SAMPLE_SIZES[i][j]) + 0.25) - 2.0)
                sample_x = rng.uniform(PLOT_X_MIN, PLOT_X_MAX, SAMPLE_SIZES[i][j])
                sample_y = rng.uniform(PLOT_Y_MIN, PLOT_Y_MAX, SAMPLE_SIZES[i][j])
                sample_data_tags = rng.choice(FITTING_TAGS, SAMPLE_SIZES[i][j])
                sample_data = np.empty(SAMPLE_SIZES[i][j], dtype=float)
                for tag in FITTING_TAGS:
                    idx = (sample_data_tags == tag)
                    sample_data[idx] = GROUND_TRUTH[tag](sample_x[idx], sample_y[idx])
                poly = Polynomial(order=order)
                poly.fit(sample_x, sample_y, sample_data, 0.01, sample_data_tags)
                fit_f = poly.f(plot_x, plot_y)
                ax[i][j].tricontourf(plot_x, plot_y, fit_f, levels=np.linspace(vmin, vmax, 200), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
                ax[i][j].scatter(sample_x, sample_y, color='k', ec='white')
                ax[i][j].set_title(f'Fitted on {SAMPLE_SIZES[i][j]} points (order {order} polynomial)')
            ax[i][j].set_xlabel('x')
            ax[i][j].set_ylabel('y')
            ax[i][j].set_aspect(1.0)
            ax[i][j].set_xlim(PLOT_X_MIN, PLOT_X_MAX)
            ax[i][j].set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)

    fig.colorbar(art, ax=ax, extend='both')
    fig.savefig(plot_file, dpi=300.0, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def test_read_write(logging_dir: Path, rng: np.random.Generator):

    plot_file = logging_dir / 'read_write.png'
    save_file = logging_dir / 'read_write.yaml'
    plot_file.unlink(missing_ok=True)
    save_file.unlink(missing_ok=True)

    PLOT_X_MIN, PLOT_X_MAX = -1.0, 1.0
    SAMPLE_SIZE = 200

    plot_x = np.linspace(PLOT_X_MIN, PLOT_X_MAX, 201)
    plot_y = np.linspace(PLOT_X_MIN, PLOT_X_MAX, 201)
    plot_x, plot_y = np.meshgrid(plot_x, plot_y)
    plot_x, plot_y = plot_x.ravel(), plot_y.ravel()

    sample_x = rng.uniform(PLOT_X_MIN, PLOT_X_MAX, SAMPLE_SIZE)
    sample_y = rng.uniform(PLOT_X_MIN, PLOT_X_MAX, SAMPLE_SIZE)
    sample_f = np.clip(rng.normal(0.0, 1.0, SAMPLE_SIZE), a_min=-2.0, a_max=2.0)

    orig_poly = Polynomial(order=6)
    orig_poly.fit(sample_x, sample_y, sample_f, 1.0, 'f')
    with open(save_file, 'w') as fp:
        yaml.safe_dump(orig_poly.get_params(), fp)

    new_poly = Polynomial(order=10)
    with open(save_file, 'r') as fp:
        new_poly.load_params(yaml.safe_load(fp))

    orig_f = orig_poly.f(plot_x, plot_y)
    new_f = new_poly.f(plot_x, plot_y)
    assert np.allclose(orig_f, new_f)

    cmap = plt.colormaps['viridis'].with_extremes(bad=(1.0, 1.0, 1.0), under=(1.0, 1.0, 1.0), over=(1.0, 1.0, 1.0))

    fig, ax = plt.subplots(1, 2, layout='constrained')
    fig.set_size_inches(12.0, 5.0)
    art = ax[0].tricontourf(plot_x, plot_y, orig_f, levels=np.linspace(-2.0, 2.0, 200), cmap=cmap, vmin=-2.0, vmax=2.0, extend='both')
    ax[0].scatter(sample_x, sample_y, c=sample_f, ec='white', cmap=cmap, vmin=-2.0, vmax=2.0)
    ax[0].set_title('Original')
    ax[1].tricontourf(plot_x, plot_y, new_f, levels=np.linspace(-2.0, 2.0, 200), cmap=cmap, vmin=-2.0, vmax=2.0, extend='both')
    ax[1].scatter(sample_x, sample_y, c=sample_f, ec='white', cmap=cmap, vmin=-2.0, vmax=2.0)
    ax[1].set_title('New')
    for i in range(2):
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')
        ax[i].set_aspect(1.0)
        ax[i].set_xlim(PLOT_X_MIN, PLOT_X_MAX)
        ax[i].set_ylim(PLOT_X_MIN, PLOT_X_MAX)

    fig.colorbar(art, ax=ax, extend='both')
    fig.savefig(plot_file, dpi=300.0, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)