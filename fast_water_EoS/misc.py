import time
from typing import Any

def elapsed_time(
        time_start: float,
        time_end: float = None,
    ) -> str:
    """
    Given a start time measured by `time.time()`, this function measures the
    time elapsed since and formats it nicely into a string.

    Parameters
    ----------
    time_start : float
        Time in seconds to report elapsed time from. The elapsed time is
        `time_end` minus `time_start`.
    time_end : float, optional
        Time in seconds to report elapsed time up to. If not specified,
        defaults to the current time at which this function is called.
    
    Returns
    -------
    elapsed_time : str
        The elapsed time, formatted as (hh:mm:ss) for times greater than one
        hour; or as minutes and seconds for times greater than one minute; or
        as seconds for times greater than one second; or as milliseconds for
        times smaller than one second.
    """
    time_taken = (time.time() if time_end is None else time_end) - time_start
    if time_taken > 3600:
        hours = int(time_taken / 3600)
        time_taken -= hours * 3600
        mins = int(time_taken / 60)
        time_taken -= mins * 60
        return f'(hh:mm:ss) {hours:02}:{mins:02}:{round(time_taken):02}'
    if time_taken > 60:
        mins = int(time_taken / 60)
        time_taken -= mins * 60
        return f'{mins}mins{round(time_taken):02}s'
    if time_taken > 1:
        return f'{time_taken:.3f}s'
    return f'{(1000.0 * time_taken):.1f}ms'


def castable(x: Any, t: type) -> bool:
    """Checks if `x` can be safely cast to the desired type."""
    try:
        t(x)
        return isinstance(t, type)
    except:
        return False
    
