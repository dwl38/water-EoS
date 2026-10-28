import sys
import pytest
import numpy as np
import fast_water_EoS.cli.build_equation_of_state
from pathlib import Path
from fast_water_EoS.free_energy_surface import FreeEnergySurface


@pytest.fixture
def data_file() -> Path:
    output_dir = Path('test_logs/build_EoS')
    output_dir.mkdir(parents=True, exist_ok=True)

    V_range = np.linspace(25.0, 33.4, 6)
    T_range = np.linspace(250.0, 370.0, 10)
    V_range, T_range = np.meshgrid(V_range, T_range)
    V_range, T_range = V_range.ravel(), T_range.ravel()

    fes = FreeEnergySurface('gerg')
    p_range = fes.pressure(V_range, T_range)
    U_range = fes.internal_energy(V_range, T_range)

    data_file = output_dir / 'data.txt'
    data_file.unlink(missing_ok=True)
    with open(data_file, 'w') as fp:
        fp.write('# Test file\n')
        fp.write('# V T p p_err U U_err\n')
        for V, T, p, U in zip(V_range, T_range, p_range, U_range):
            fp.write(f'{V} {T} {p} 5.0 {U} 0.005\n')

    return data_file


def test_build_EoS(data_file: Path, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['build_equation_of_state', 'test_logs/build_EoS/data.txt', '-o', 'test_logs/build_EoS'])
        fast_water_EoS.cli.build_equation_of_state.main()


