import yaml
import json
import numpy as np
from pathlib import Path

_FIELDS = ['V', 'T', 'p', 'p_err', 'U', 'U_err']

def read_file(file):
    """Reads a file, presumably containing thermodynamic data from NVT
    simulations.

    Parameters
    ----------
    file : str or pathlib.Path
        The file to read from.

    Returns
    -------
    V : ndarray of floats of shape (N_data,)
        Volumes
    T : ndarray of floats of shape (N_data,)
        Temperatures
    p : ndarray of floats of shape (N_data,)
        Pressures
    p_err : ndarray of floats of shape (N_data,)
        Pressure uncertainties
    U : ndarray of floats of shape (N_data,)
        Internal total energies
    U_err : ndarray of floats of shape (N_data,)
        Internal total energy uncertainties

    Notes
    -----
    File format is expected to be one of the following:
    - **YAML or JSON file.** File extension ending in ".yaml", ".yml", or
      ".json". Should represent a dictionary with all of the fields "V", "T",
      "p", "p_err", "U", and "U_err" present, which should all be lists or
      otherwise iterables of floats of the same length.
    - **Plaintext DAT file.** File extension ending in ".dat" or ".txt". The
      first two lines will be ignored as comments, and all remaining lines
      will be interpreted as space-separated values, in the following order:

    >>> # Filename, title, other comments, etc.
    >>> #   V     T     p   p_err   U   U_err
    >>>   xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx
    >>>   xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx
    >>>    ...   ...   ...   ...   ...   ...

    - **Numpy archive.** File extension ending in ".npz". Should contain the
      fields "V", "T", "p", "p_err", "U", and "U_err", which are all expected
      to be 1D ndarrays of floats of the same shape.

    Any unrecognized file format raises an IOError.
    """
    fp = Path(file)
    if not fp.is_file():
        raise IOError(f'{file} not found!')
    extension = fp.suffix

    if extension.lower() in ('.yaml', '.yml', '.json', '.npz'):

        if extension.lower() in ('.yaml', '.yml'):
            with open(file, 'r') as stream:
                loaded = yaml.safe_load(stream)
        elif extension.lower() == '.json':
            with open(file, 'r') as stream:
                loaded = json.load(stream.read())
        elif extension.lower() == '.npz':
            loaded = np.load(file)

        for field in _FIELDS:
            if field not in loaded:
                raise IOError(f'Field "{field}" missing!')
        sanitized = [np.asarray(loaded[field], dtype=float)
                     for field in _FIELDS]
        broadcasted = np.broadcast_arrays(*sanitized)
        return tuple((d.ravel() for d in broadcasted))
        
    elif extension.lower() in ('.dat', '.txt'):

        full_data = list()
        with open(file, 'r') as stream:
            stream.readline()
            stream.readline()
            for line in stream:
                tokens = line.split()
                row = [float(tokens[i]) for i in range(6)]
                full_data.append(row)
        full_data = np.asarray(full_data, dtype=float)
        return tuple((full_data[:,i] for i in range(6)))

    else:
        raise IOError(f'File format "{extension}" not recognized!')

    
