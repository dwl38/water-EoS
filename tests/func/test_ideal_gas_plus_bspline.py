import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fast_water_EoS.func import FITTING_TAGS
from fast_water_EoS.func.ideal_gas_plus_bspline import IdealGasPlusBSpline


@pytest.fixture
def logging_dir() -> Path:
    output_dir = Path('test_logs/func/ideal_plus_bspline')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng()


def _make_grid(xlim, ylim, n_grid):
    x, y = np.linspace(*xlim, n_grid), np.linspace(*ylim, n_grid)
    x, y = np.meshgrid(x, y)
    return x.ravel(), y.ravel()


def test_mixed_fitting(logging_dir: Path, rng: np.random.Generator):

    plot_file = logging_dir / 'mixed_fitting.png'
    plot_file.unlink(missing_ok=True)

    def true_f(x, y):
        return np.sin(x + y - 5.75) + ((x - y + 0.75)**2) - (1.5 * x) + (2.5 * y) + np.log(x) + (3.0 * np.log(y))
    
    def true_dfdx(x, y):
        return np.cos(x + y - 5.75) + (2.0 * (x - y + 0.75)) - 1.5 + (1.0 / x)
    
    def true_dfdy(x, y):
        return np.cos(x + y - 5.75) - (2.0 * (x - y + 0.75)) + 2.5 + (3.0 / y)
    
    def true_ddfdxx(x, y):
        return -np.sin(x + y - 5.75) + 2.0 - (1.0 / (x**2))
    
    def true_ddfdxy(x, y):
        return -np.sin(x + y - 5.75) - 2.0
    
    def true_ddfdyy(x, y):
        return -np.sin(x + y - 5.75) + 2.0 - (3.0 / (y**2))
    
    def true_xdfdx(x, y):
        return x * true_dfdx(x, y)
    
    def true_ydfdy(x, y):
        return y * true_dfdy(x, y)
    
    def true_y2ddfdyy(x, y):
        return (y**2) * true_ddfdyy(x, y)
    
    def true_xdfdxmxyddfdxy(x, y):
        return x * (true_dfdx(x, y) - (y * true_ddfdxy(x, y)))
    
    GROUND_TRUTH = {
        'f': true_f,
        'xdfdx': true_xdfdx,
        'ydfdy': true_ydfdy,
        'y2ddfdyy': true_y2ddfdyy,
        'xdfdx-xyddfdxy': true_xdfdxmxyddfdxy,
    }
    XLIM = (0.25, 7.25)
    YLIM = (0.25, 7.25)
    SAMPLE_GRIDSIZE = 8
    N_SAMPLES = SAMPLE_GRIDSIZE**2

    plot_x, plot_y = _make_grid(XLIM, YLIM, 201)
    plot_truth = {key: truth(plot_x, plot_y) for key, truth in GROUND_TRUTH.items()}
    width = {key: np.std(values) for key, values in plot_truth.items()}

    sample_x, sample_y = _make_grid(XLIM, YLIM, SAMPLE_GRIDSIZE)
    sample_data_tags = rng.choice(list(FITTING_TAGS.values()), N_SAMPLES)
    sample_data = np.empty(N_SAMPLES, dtype=float)
    sample_data_err = np.empty(N_SAMPLES, dtype=float)
    for key, tag in FITTING_TAGS.items():
        idx = (sample_data_tags == tag)
        noise = abs(0.01 * width[key])
        sample_data[idx] = GROUND_TRUTH[key](sample_x[idx], sample_y[idx])
        sample_data[idx] += rng.normal(0.0, noise, np.count_nonzero(idx))
        sample_data_err[idx] = noise
    
    model = IdealGasPlusBSpline()
    model.fit(sample_x, sample_y, sample_data, sample_data_err, sample_data_tags)

    funcs_to_plot = [
        (true_f, model.f, 'f'),
        (true_dfdx, model.dfdx, 'df/dx'),
        (true_dfdy, model.dfdy, 'df/dy'),
        (true_ddfdxx, model.ddfdxx, 'd²f/dx²'),
        (true_ddfdxy, model.ddfdxy, 'd²f/dxdy'),
        (true_ddfdyy, model.ddfdyy, 'd²f/dy²'),
    ]
    cmap = plt.colormaps['viridis'].with_extremes(
        bad=(1.0, 1.0, 1.0),
        under=(1.0, 1.0, 1.0),
        over=(1.0, 1.0, 1.0),
    )

    fig, ax = plt.subplots(6, 2, layout='constrained')
    fig.set_size_inches(10.0, 24.0)
    for i, (ref_func, model_func, name) in enumerate(funcs_to_plot):
        ref_values = ref_func(plot_x, plot_y)
        model_values = model_func(plot_x, plot_y)
        sq_dev = np.mean((model_values - ref_values)**2)
        vmin, vmax = ref_values.min(), ref_values.max()
        levels = np.linspace(vmin, vmax, 200)
        ax[i][0].tricontourf(
            plot_x, plot_y, ref_values,
            levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend='both',
        )
        art = ax[i][1].tricontourf(
            plot_x, plot_y, model_values,
            levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend='both',
        )
        ax[i][1].scatter(sample_x, sample_y, color='k', ec='white')
        ax[i][0].set_title('True ' + name)
        ax[i][1].set_title(f'Model {name} [|Δf|² = {sq_dev:g}]')
        fig.colorbar(art, ax=ax[i])
        for j in range(2):
            ax[i][j].set_xlabel('x')
            ax[i][j].set_ylabel('y')
            ax[i][j].set_aspect(1.0)
            ax[i][j].set_xlim(*XLIM)
            ax[i][j].set_ylim(*YLIM)

    fig.savefig(plot_file, dpi=300.0, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def test_read_write(logging_dir: Path, rng: np.random.Generator):

    plot_file = logging_dir / 'read_write.png'
    save_file = logging_dir / 'read_write.npz'
    plot_file.unlink(missing_ok=True)
    save_file.unlink(missing_ok=True)

    PLOT_LIMS = (1.0, 2.0)
    SAMPLE_SIZE = 200
    plot_x, plot_y = _make_grid(PLOT_LIMS, PLOT_LIMS, 201)

    sample_x = rng.uniform(*PLOT_LIMS, SAMPLE_SIZE)
    sample_y = rng.uniform(*PLOT_LIMS, SAMPLE_SIZE)
    sample_f = np.clip(rng.normal(0.0, 1.0, SAMPLE_SIZE), a_min=-2.0, a_max=2.0)

    orig_model = IdealGasPlusBSpline()
    orig_model.fit(sample_x, sample_y, sample_f, 1.0, FITTING_TAGS['f'])
    np.savez_compressed(save_file, **orig_model.get_params())

    new_model = IdealGasPlusBSpline()
    new_model.load_params(np.load(save_file))

    orig_f = orig_model.f(plot_x, plot_y)
    valid_orig = np.isfinite(orig_f)
    new_f = new_model.f(plot_x, plot_y)
    valid_new = np.isfinite(new_f)
    valid_all = valid_orig & valid_new
    assert np.allclose(orig_f[valid_all], new_f[valid_all])

    cmap = plt.colormaps['viridis'].with_extremes(bad=(1.0, 1.0, 1.0), under=(1.0, 1.0, 1.0), over=(1.0, 1.0, 1.0))

    fig, ax = plt.subplots(1, 2, layout='constrained')
    fig.set_size_inches(12.0, 5.0)
    art = ax[0].tricontourf(
        plot_x[valid_orig], plot_y[valid_orig], orig_f[valid_orig],
        levels=np.linspace(-2.0, 2.0, 200), cmap=cmap, vmin=-2.0, vmax=2.0, extend='both',
    )
    ax[0].scatter(sample_x, sample_y, c=sample_f, ec='white', cmap=cmap, vmin=-2.0, vmax=2.0)
    ax[0].set_title('Original')
    ax[1].tricontourf(
        plot_x[valid_new], plot_y[valid_new], new_f[valid_new],
        levels=np.linspace(-2.0, 2.0, 200), cmap=cmap, vmin=-2.0, vmax=2.0, extend='both',
    )
    ax[1].scatter(sample_x, sample_y, c=sample_f, ec='white', cmap=cmap, vmin=-2.0, vmax=2.0)
    ax[1].set_title('New')
    fig.colorbar(art, ax=ax, extend='both')
    for i in range(2):
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')
        ax[i].set_aspect(1.0)
        ax[i].set_xlim(*PLOT_LIMS)
        ax[i].set_ylim(*PLOT_LIMS)

    fig.savefig(plot_file, dpi=300.0, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)