import yaml
import json
import numpy as np
from pathlib import Path

_MANDATORY_FIELDS = ['V', 'T', 'p', 'p_err', 'U', 'U_err']
_OPTIONAL_FIELDS = ['Cv', 'Cv_err']

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
    Cv : ndarray of floats of shape (N_data,) or None
        Isochoric heat capacities, if present; otherwise None
    Cv_err : ndarray of floats of shape (N_data,) or None
        Isochoric heat capacity uncertainties, if present; otherwise None

    Notes
    -----
    File format is expected to be one of the following:
    
    - **YAML or JSON file.** File extension ending in ".yaml", ".yml", or
      ".json". Should represent a dictionary with all of the fields "V", "T",
      "p", "p_err", "U", and "U_err" present, which should all be lists or
      otherwise iterables of floats of the same length. The fields "Cv" and
      "Cv_err" may optionally be present.

    - **Plaintext DAT file.** File extension ending in ".dat" or ".txt". The
      first two lines will be ignored as comments, and all remaining lines
      will be interpreted as space-separated values, in the following order:
      ```
        # Filename, title, other comments, etc.
        #    V       T       p     p_err     U     U_err    Cv    Cv_err
          xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx
          xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx
            ...     ...     ...     ...     ...     ...     ...     ...
      ```
      where the final two columns are optional. Note that the data fields are
      positional, i.e. the first column is always interpreted as volume and
      the second column as temperature etc. regardless of what the header
      says!

    - **Numpy archive.** File extension ending in ".npz". Should contain the
      fields "V", "T", "p", "p_err", "U", and "U_err", which are all expected
      to be 1D ndarrays of floats of the same shape. The fields "Cv" and
      "Cv_err" may optionally be present.

    Any unrecognized file format raises an IOError.
    """
    fp = Path(file)
    if not fp.is_file():
        raise IOError(f'{file} not found!')
    extension = fp.suffix

    N_mandatory_fields = len(_MANDATORY_FIELDS)
    N_all_fields = len(_MANDATORY_FIELDS) + len(_OPTIONAL_FIELDS)

    if extension.lower() in ('.yaml', '.yml', '.json', '.npz'):

        if extension.lower() in ('.yaml', '.yml'):
            with open(file, 'r') as stream:
                loaded = yaml.safe_load(stream)
        elif extension.lower() == '.json':
            with open(file, 'r') as stream:
                loaded = json.load(stream.read())
        elif extension.lower() == '.npz':
            loaded = np.load(file)

        present_fields = _MANDATORY_FIELDS.copy()
        for field in _MANDATORY_FIELDS:
            if field not in loaded:
                raise IOError(f'Field "{field}" missing!')
        for field in _OPTIONAL_FIELDS:
            if field in loaded:
                present_fields.append(field)
        missing_count = N_all_fields - len(present_fields)
        
        sanitized = [np.asarray(loaded[field], dtype=float)
                     for field in present_fields]
        broadcasted = np.broadcast_arrays(*sanitized)
        present_results = tuple((d.ravel() for d in broadcasted))
        missing_results = tuple((None for _ in range(missing_count)))
        return present_results + missing_results
        
    elif extension.lower() in ('.dat', '.txt'):

        N_fields = None

        full_data = list()
        with open(file, 'r') as stream:
            stream.readline()
            stream.readline()
            for line in stream:
                tokens = line.split()
                if N_fields is None:
                    N_fields = len(tokens)
                    if ((N_fields != N_mandatory_fields)
                        and (N_fields != N_all_fields)):
                        raise IOError('Unexpected number of tokens '
                                      f'({len(tokens)}) found!')
                row = [float(token) for token in tokens]
                full_data.append(row)
        missing_count = N_all_fields - N_fields
        full_data = np.asarray(full_data, dtype=float)
        present_results = tuple((full_data[:,i] for i in range(N_fields)))
        missing_results = tuple((None for _ in range(missing_count)))
        return present_results + missing_results

    else:
        raise IOError(f'File format "{extension}" not recognized!')

    
