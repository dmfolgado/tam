import numpy as np

def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

def tam(path, report='full'):
    """
    Calculates the Time Alignment Measurement (TAM) based on an optimal warping path
    between two time series.
    Reference: Folgado et. al, Time Alignment Measurement for Time Series, 2018.

    :param path: (ndarray)
                A nested array containing the optimal warping path between the
                two sequences.
    :param report: (string)
                A string containing the report mode parameter.
    :return:    In case ``report=instants`` the number of indexes in advance, delay and phase
                will be returned. For ``report=ratios``, the ratio of advance, delay and phase
                will be returned. In case ``report=distance``, only the TAM will be returned.

    """
    # Delay and advance counting
    delay = len(find(np.diff(path[0]) == 0))
    advance = len(find(np.diff(path[1]) == 0))

    # Phase counting
    incumbent = find((np.diff(path[0]) == 1) * (np.diff(path[1]) == 1))
    phase = len(incumbent)

    # Estimated and reference time series duration.
    len_estimation = path[1][-1]
    len_ref = path[0][-1]

    p_advance = advance * 1. / len_ref
    p_delay = delay * 1. / len_estimation
    p_phase = phase * 1. / np.min([len_ref, len_estimation])

    if report == 'instants':
        return np.array([advance, delay, phase])

    if report == 'ratios':
        return np.array([advance, delay, phase])

    if report == 'distance':
        return p_advance + p_delay + (1 - p_phase)

    if report == 'full':
        return np.array([advance, delay, phase, p_advance + p_delay + (1 - p_phase)])
