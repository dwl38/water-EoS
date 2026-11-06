import sys
import shutil
import pytest
import numpy as np
import fast_water_EoS.cli.build_equation_of_state
import fast_water_EoS.cli.suggest_new_state
from pathlib import Path
from fast_water_EoS.free_energy_surface import FreeEnergySurface


@pytest.fixture
def data_file() -> Path:
    output_dir = Path('test_logs/build_EoS')
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
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


@pytest.fixture
def data_file_w_Cv() -> Path:
    output_dir = Path('test_logs/build_EoS_w_Cv')
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    V_range = np.linspace(25.0, 33.4, 6)
    T_range = np.linspace(250.0, 370.0, 10)
    V_range, T_range = np.meshgrid(V_range, T_range)
    V_range, T_range = V_range.ravel(), T_range.ravel()

    fes = FreeEnergySurface('gerg')
    p_range = fes.pressure(V_range, T_range)
    U_range = fes.internal_energy(V_range, T_range)
    Cv_range = fes.heat_capacity(V_range, T_range)

    data_file = output_dir / 'data.txt'
    data_file.unlink(missing_ok=True)
    with open(data_file, 'w') as fp:
        fp.write('# Test file\n')
        fp.write('# V T p p_err U U_err Cv Cv_err\n')
        for V, T, p, U, Cv in zip(V_range, T_range, p_range, U_range, Cv_range):
            fp.write(f'{V} {T} {p} 5.0 {U} 0.005 {Cv} 0.00001\n')

    return data_file


def test_build_EoS(data_file: Path, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['build_equation_of_state', 'test_logs/build_EoS/data.txt', '-o', 'test_logs/build_EoS'])
        fast_water_EoS.cli.build_equation_of_state.main()
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['suggest_new_state', 'test_logs/build_EoS/data.txt', 'test_logs/build_EoS/EoS.yaml', '-o', 'test_logs/build_EoS'])
        fast_water_EoS.cli.suggest_new_state.main()


def test_build_EoS_w_Cv(data_file_w_Cv: Path, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['build_equation_of_state', 'test_logs/build_EoS_w_Cv/data.txt', '-o', 'test_logs/build_EoS_w_Cv'])
        fast_water_EoS.cli.build_equation_of_state.main()
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['suggest_new_state', 'test_logs/build_EoS_w_Cv/data.txt', 'test_logs/build_EoS_w_Cv/EoS.yaml', '-o', 'test_logs/build_EoS_w_Cv'])
        fast_water_EoS.cli.suggest_new_state.main()


def test_EoS_for_spce(monkeypatch):
    output_dir = Path('test_logs/EoS_spce')
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['build_equation_of_state', 'tests/data_spce.dat', '-o', 'test_logs/EoS_spce'])
        fast_water_EoS.cli.build_equation_of_state.main()
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['suggest_new_state', 'tests/data_spce.dat', 'test_logs/EoS_spce/EoS.yaml', '-o', 'test_logs/EoS_spce'])
        fast_water_EoS.cli.suggest_new_state.main()


def test_EoS_for_spcfw(monkeypatch):
    output_dir = Path('test_logs/EoS_spcfw')
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['build_equation_of_state', 'tests/data_spcfw.dat', '-o', 'test_logs/EoS_spcfw'])
        fast_water_EoS.cli.build_equation_of_state.main()
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['suggest_new_state', 'tests/data_spcfw.dat', 'test_logs/EoS_spcfw/EoS.yaml', '-o', 'test_logs/EoS_spcfw'])
        fast_water_EoS.cli.suggest_new_state.main()



def test_EoS_for_ccsdt(monkeypatch):
    output_dir = Path('test_logs/EoS_ccsdt')
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['build_equation_of_state', 'tests/data_ccsdt.dat', '-o', 'test_logs/EoS_ccsdt'])
        fast_water_EoS.cli.build_equation_of_state.main()
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['suggest_new_state', 'tests/data_ccsdt.dat', 'test_logs/EoS_ccsdt/EoS.yaml', '-o', 'test_logs/EoS_ccsdt'])
        fast_water_EoS.cli.suggest_new_state.main()



