import numpy as np
import pylab as pl
import scipy.interpolate as it


# Auxiliary functions
def get_mirror(s, ws):
    """
    Performs a signal windowing based on a double inversion from the start and end segments.
    :param s: (array-like)
            the input-signal.
    :param ws: (integer)
            window size.
    :return:
    """

    return np.r_[2 * s[0] - s[ws:0:-1], s, 2 * s[-1] - s[-2:-ws - 2:-1]]


def normalize_signal(s):
    """
    Normalizes a given signal by subtracting the mean and dividing by the standard deviation.
    :param s: (array_like)
            The input signal.
    :return:
            The normalized input signal.
    """
    return (s - np.mean(s)) / np.std(s)


# Processing
def dtw(x, y, dist=lambda a, b: (a - b) ** 2, **kwargs):
    """
    Computes Dynamic Time Warping (DTW) of two time series.
    :param x: (array_like)
            The reference signal.
    :param y: (array_like)
            The estimated signal.
    :param dist: (function)
            The distance used as a local cost measure.

    :param \**kwargs:
    See below:

    * *do_sign_norm* (``bool``) --
      If ``True`` the signals will be normalized before computing the DTW,
      (default: ``False``)

    * *do_dist_norm* (``bool``) --
      If ``True`` the DTW distance will be normalized by dividing the summation of the path dimension.
      (default: ``True``)

    * *window* (``String``) --
      Selects the global window constrains. Available options are ``None`` and ``sakoe-chiba``.
      (default: ``None``)

    * *factor* (``Float``) --
      Selects the global constrain factor.
      (default: ``min(xl, yl) * .50``)


    :return:
           d: (float)
            The DTW distance.
           C: (array_like)
            The local cost matrix.
           ac: (array_like)
            The accumulated cost matrix.
           path (array_like)
            The optimal warping path between the two sequences.
    """
    xl, yl = len(x), len(y)

    do_sign_norm = kwargs.get('normalize', False)
    do_dist_norm = kwargs.get('dist_norm', True)
    window = kwargs.get('window', None)
    factor = kwargs.get('factor', np.min((xl, yl)) * .50)

    if do_sign_norm:
        x, y = normalize_signal(x), normalize_signal(y)

    ac = np.zeros((xl + 1, yl + 1))
    ac[0, 1:] = np.inf
    ac[1:, 0] = np.inf
    tmp_ac = ac[1:, 1:]

    for i in range(xl):
        for j in range(yl):
            # No window selected
            if window is None:
                tmp_ac[i, j] = dist(x[i], y[j])

            # Sakoe-Chiba band
            elif window == 'sakoe-chiba':
                if abs(i - j) < factor:
                    tmp_ac[i, j] = dist(x[i], y[j])
                else:
                    tmp_ac[i, j] = np.inf

            # As last resource, the complete window is calculated
            else:
                tmp_ac[i, j] = dist(x[i], y[j])

    c = tmp_ac.copy()

    for i in range(xl):
        for j in range(yl):
            tmp_ac[i, j] += min([ac[i, j], ac[i, j + 1], ac[i + 1, j]])

    path = _traceback(ac)

    if do_dist_norm:
        d = ac[-1, -1] / np.sum(np.shape(path))
    else:
        d = ac[-1, -1]

    return d, c, ac, path


def dtw_sw(x, y, winlen, alpha=0.5, **kwargs):
    """
    Computes Dynamic Time Warping (DTW) of two time series.
    :param x: (array_like)
            The reference signal.
    :param y: (array_like)
            The estimated signal.
    :param winlen: (int)
            The sliding window length
    :param alpha: (float)
            A factor between 0 and 1 which weights the amplitude and derivative contributions.
            A higher value will favor amplitude and a lower value will favor the first derivative.

    :param \**kwargs:
        See below:

        * *do_sign_norm* (``bool``) --
          If ``True`` the signals will be normalized before computing the DTW,
          (default: ``False``)

        * *do_dist_norm* (``bool``) --
          If ``True`` the DTW distance will be normalized by dividing the summation of the path dimension.
          (default: ``True``)

        * *window* (``String``) --
          Selects the global window constrains. Available options are ``None`` and ``sakoe-chiba``.
          (default: ``None``)

        * *factor* (``Float``) --
          Selects the global constrain factor.
          (default: ``min(xl, yl) * .50``)


    :return:
           d: (float)
            The SW-DTW distance.
           C: (array_like)
            The local cost matrix.
           ac: (array_like)
            The accumulated cost matrix.
           path (array_like)
            The optimal warping path between the two sequences.
    """
    xl, yl = len(x), len(y)

    do_sign_norm = kwargs.get('normalize', False)
    do_dist_norm = kwargs.get('dist_norm', True)
    window = kwargs.get('window', None)
    factor = kwargs.get('factor', np.min((xl, yl)) * .50)

    if do_sign_norm:
        x, y = normalize_signal(x), normalize_signal(y)

    ac = np.zeros((xl + 1, yl + 1))
    ac[0, 1:] = np.inf
    ac[1:, 0] = np.inf
    tmp_ac = ac[1:, 1:]

    nx = get_mirror(x, winlen)
    ny = get_mirror(y, winlen)

    dnx = np.diff(nx)
    dny = np.diff(ny)

    nx = nx[:-1]
    ny = ny[:-1]

    # Workaround to deal with even window sizes
    if winlen % 2 == 0:
        winlen -= 1

    swindow = np.hamming(winlen)
    swindow = swindow / np.sum(swindow)

    for i in range(xl):
        for j in range(yl):
            pad_i, pad_j = i + winlen, j + winlen
            # No window selected
            if window is None:
                tmp_ac[i, j] = sliding_dist(nx[pad_i - (winlen / 2):pad_i + (winlen / 2) + 1],
                                        ny[pad_j - (winlen / 2):pad_j + (winlen / 2) + 1],
                                        dnx[pad_i - (winlen / 2):pad_i + (winlen / 2) + 1],
                                        dny[pad_j - (winlen / 2):pad_j + (winlen / 2) + 1], alpha, swindow)

            # Sakoe-Chiba band
            elif window == 'sakoe-chiba':
                if abs(i - j) < factor:
                    tmp_ac[i, j] = sliding_dist(nx[pad_i - (winlen / 2):pad_i + (winlen / 2) + 1],
                                        ny[pad_j - (winlen / 2):pad_j + (winlen / 2) + 1],
                                        dnx[pad_i - (winlen / 2):pad_i + (winlen / 2) + 1],
                                        dny[pad_j - (winlen / 2):pad_j + (winlen / 2) + 1], alpha, swindow)
                else:
                    tmp_ac[i, j] = np.inf

            # As last resource, the complete window is calculated
            else:
                tmp_ac[i, j] = sliding_dist(nx[pad_i - (winlen / 2):pad_i + (winlen / 2) + 1],
                                        ny[pad_j - (winlen / 2):pad_j + (winlen / 2) + 1],
                                        dnx[pad_i - (winlen / 2):pad_i + (winlen / 2) + 1],
                                        dny[pad_j - (winlen / 2):pad_j + (winlen / 2) + 1], alpha, swindow)

    c = tmp_ac.copy()

    for i in range(xl):
        for j in range(yl):
            tmp_ac[i, j] += min([ac[i, j], ac[i, j + 1], ac[i + 1, j]])

    path = _traceback(ac)

    if do_dist_norm:
        d = ac[-1, -1] / np.sum(np.shape(path))
    else:
        d = ac[-1, -1]

    return d, c, ac, path


def sliding_dist(xw, yw, dxw, dyw, a, win):
    return (1 - a) * np.sqrt(np.sum((((dxw - dyw) * win) ** 2.))) + \
           a * np.sqrt(np.sum((((xw - yw) * win) ** 2.)))


def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def align_sequences(ref, s, path):
    """
    This functions aligns two time-series. The alignment is performed
    for a given reference signal and a vector containing the alignment.
    :param ref: (array-like)
            The reference signal.
    :param s: (array-like)
            The signal to be aligned.
    :param path: (ndarray)
            A rank 2 array containing the optimal warping path between the two signals.
    :return:
    """
    nt = np.linspace(0, len(ref) - 1, len(ref))
    ns = it.interp1d(path[0], s[path[1]])(nt)

    return ns


# Visualization
def plot_alignment(ref_signal, estimated_signal, path, **kwargs):
    """
    This functions plots the resulted alignment of two sequences given the path
    calculated by the Dynamic Time Warping algorithm.

    :param ref_signal: (array-like)
                     The reference sequence.
    :param estimated_signal: (array-like)
                     The estimated sequence.
    :param path: (array-like)
                     A 2D array congaing the path resulted from the algorithm
    :param \**kwargs:
        See below:

        * *offset* (``double``) --
            The offset used to move the reference signal to an upper position for
            visualization purposes.
            (default: ``2``)

        * *linewidths* (``list``) --
            A list containing the linewidth for the reference, estimated and connection
            plots, respectively.
            (default: ``[3, 3, 0.5]``)

        * *step* (``int``) --
            The step for
          (default: ``2``)

        * *colors* (``list``) --
          A list containing the colors for the reference, estimated and connection
          plots, respectively.
          (default: ``[sns.color_palette()[0], sns.color_palette()[1], 'k']``)
    """

    step = kwargs.get('step', 2)
    offset = kwargs.get('offset', 2)
    linewidths = kwargs.get('linewidths', [3, 3, 0.5])
    colors = kwargs.get('colors', [sns.color_palette()[0], sns.color_palette()[1], 'k'])

    copy_ref = np.copy(ref_signal)    # This prevents unexpected changes in the reference signal after the duplicate
    copy_ref += offset * np.max(ref_signal)    # Set an offset for visualization

    # Actual plot occurs here
    pl.plot(copy_ref, color=sns.color_palette()[0], lw=linewidths[0], label='reference')
    pl.plot(estimated_signal, color=sns.color_palette()[1], lw=linewidths[1], label='estimate')
    pl.legend(fontsize=17)

    [pl.plot([[path[0][i]], [path[1][i]]],
             [copy_ref[path[0][i]], estimated_signal[path[1][i]]],
             color=colors[2], lw=linewidths[2]) for i in range(len(path[0]))[::step]]


def plot_costmatrix(matrix, path):
    """
    This functions overlays the optimal warping path and the cost matrices
    :param matrix: (ndarray-like)
                The cost matrix (local cost or accumulated)
    :param path:   (ndarray-like)
                The optimal warping path
    :return: (void)
                Plots the optimal warping path with an overlay of the cost matrix.
    """
    pl.imshow(matrix.T, cmap='viridis', origin='lower', interpolation='None')
    pl.colorbar()
    pl.plot(path[0], path[1], 'w.-')
    pl.xlim((-0.5, matrix.shape[0] - 0.5))
    pl.ylim((-0.5, matrix.shape[1] - 0.5))
