# fast-water-EoS
Modelling liquid (and gaseous) water's thermodynamic properties using free energy interpolants. This is achieved by fitting regularized cardinal B-splines to the dimensionless free energy; see `explanation.pdf`.

## How to use

The repository can be installed as a Python package using the following:
```
git clone --depth 1 https://github.com/dwl38/water-EoS.git
pip install ./water-EoS
```

Once installed, the package can be used to build a liquid water Equation of State from the command line:
```
build_equation_of_state <input_file> [-o <output_dir>]
```

where the input file should contain the following thermodynamic data:

- $V$: The volume per particle, in Å³.
- $T$: The temperature, in K.
- $p$: The pressure corresponding to $V$ and $T$, in bar.
- $\delta p$: Uncertainties of $p$, in bar.
- $U$: The internal (total) energy corresponding to $V$ and $T$, in eV.
- $\delta U$: Uncertainties of $U$, in eV.

The input file may also optionally contain the following thermodynamic data:

- $C_V$: The isochoric heat capacity corresponding to $V$ and $T$, in eV/K.
- $\delta C_V$: Uncertainties of $C_V$, in eV/K.

The input file should be provided in one of the following formats:

- YAML or JSON file. File extension ending in ".yaml", ".yml", or ".json". Should represent a dictionary with all of the fields `V`, `T`, `p`, `p_err`, `U`, and `U_err` present, which should all be lists or otherwise iterables of floats of the same length. The fields `Cv` and `Cv_err` may optionally be present.

- Plaintext DAT file. File extension ending in ".dat" or ".txt". The first two lines will be ignored as comments, and all remaining lines will be interpreted as space-separated values, in the following order:

        # Filename, title, other comments, etc.
        #    V       T       p     p_err     U     U_err    Cv    Cv_err
          xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx
          xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx
            ...     ...     ...     ...     ...     ...     ...     ...

  where the final two columns are optional. Note that the data fields are positional, i.e. the first column is always interpreted as volume and the second column as temperature etc. regardless of what the header says!

- Numpy archive. File extension ending in ".npz". Should contain the fields `V`, `T`, `p`, `p_err`, `U`, and `U_err`, which are all expected to be 1D ndarrays of floats of the same shape. The fields `Cv` and `Cv_err` may optionally be present.

The command line script then writes a folder called "EoS" within the working directory (unless otherwise specified by the `-o` argument), which contains plots of the fitted Equation of State, as well as a "EoS.yaml" file. This file can be loaded using the `FreeEnergySurface` object:
```
from fast_water_EoS.free_energy_surface import FreeEnergySurface

fes = FreeEnergySurface()
fes.load_from_file('EoS.yaml')
```

after which thermodynamic properties can be calculated using `FreeEnergySurface`'s methods, e.g. for your own custom analysis/plotting scripts.

## Changelog

```
v1.0.0: Initial version
```
