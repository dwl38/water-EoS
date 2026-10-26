import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fast_water_EoS.func.ideal_gas_deltatau import IdealGasDeltaTau


@pytest.fixture
def logging_dir() -> Path:
    output_dir = Path('test_logs/func/fixed')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def test_gerg(logging_dir: Path):

    plot_file = logging_dir / 'ideal_deltatau.png'
    plot_file.unlink(missing_ok=True)

    X_MIN, X_MAX = 0.1, 3.5
    Y_MIN, Y_MAX = 0.6, 2.3

    plot_x, plot_y = np.meshgrid(np.linspace(X_MIN, X_MAX, 200), np.linspace(Y_MIN, Y_MAX, 200))
    plot_x, plot_y = plot_x.ravel(), plot_y.ravel()

    ideal = IdealGasDeltaTau()
    funcs = [[ideal.f, ideal.dfdx, ideal.dfdy], [ideal.ddfdxx, ideal.ddfdxy, ideal.ddfdyy]]
    names = [['f(x, y)', '∂f/∂x', '∂f/∂y'], ['∂²f/∂x²', '∂²f/∂x∂y', '∂²f/∂y²']]
    ranges = [[10.0, 15.0, 15.0], [75.0, 40.0, 20.0]]

    cmap = plt.colormaps['viridis'].with_extremes(bad=(1.0, 1.0, 1.0), under=(1.0, 1.0, 1.0), over=(1.0, 1.0, 1.0))

    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(16.8, 10.0)
    for i in range(2):
        for j in range(3):
            plot_f = funcs[i][j](plot_x, plot_y)
            mean_f = np.median(plot_f)
            vmin = max(mean_f - ranges[i][j], np.min(plot_f))
            vmax = min(mean_f + ranges[i][j], np.max(plot_f))
            if (vmax - vmin) < 0.2:
                vmax += 0.1
                vmin -= 0.1
            levels = np.linspace(vmin, vmax, 200)
            art = ax[i][j].tricontourf(plot_x, plot_y, plot_f, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
            fig.colorbar(art, ax=ax[i][j], extend='both')
            ax[i][j].set_title(names[i][j])
            ax[i][j].set_xlabel('x')
            ax[i][j].set_ylabel('y')
            ax[i][j].set_xlim(X_MIN, X_MAX)
            ax[i][j].set_ylim(Y_MIN, Y_MAX)
    
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300.0, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def test_failures():

    ideal = IdealGasDeltaTau()
    assert len(ideal.get_params()) == 1

    # Warning triggered by attempting to fit fixed functional form
    with pytest.warns(UserWarning):
        ideal.fit([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], 'f')
