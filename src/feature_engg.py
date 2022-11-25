from math import ceil
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from detecta import detect_peaks
from scipy.signal import butter, lfilter

_label_map = {'standing': 'sedentary', 'sitting': 'sedentary', 'lying': 'sedentary', 'jogging': 'running',
              'walk_slow': 'walking', 'walk_mod': 'walking', 'walk_fast': 'walking', 'upstairs': 'stairs',
              'downstairs': 'stairs'}


def build_time_windows(_df, time_len=10, overlap_ratio=0.5, min_time_len=2,
                       sample_rate='auto', rate_calcref='epoch', sample_rate_multiplier=1000):
    if sample_rate == 'auto':
        sample_rate = (_df.shape[0] * sample_rate_multiplier) / (_df[rate_calcref].max() - _df[rate_calcref].min())

    elif isinstance(sample_rate, (int, float)):
        raise TypeError(
            "Parameter sample_rate expects int or float unless set to 'auto'; received {}".format(sample_rate))

    _twds = []
    idxwidth = ceil(sample_rate * time_len)
    min_idxwidth = ceil(sample_rate * min_time_len)
    increment = ceil(idxwidth * (1 - overlap_ratio))

    n = _df.shape[0]
    i = 0

    while i < n:
        start = i
        end = start + idxwidth

        if (end > n) or (n - end < min_idxwidth):
            end = n - 1
            i = n
        _dftw = _df.iloc[start:end].copy()
        _dftw['class'] = _dftw['activity_class'].map(_label_map)
        _dftw.drop(columns=['activity_class'], inplace=True)
        _twds.append(_dftw)
        i += increment

    return _twds


def plot_window_data(_sample_df):
    x = _sample_df['x']
    y = _sample_df['y']
    z = _sample_df['z']
    m = _sample_df['m']

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, x.size - 1, x.size), x, color='tab:blue', lw=1, label='Accl. X-axis')
    ax.plot(np.linspace(0, y.size - 1, y.size), y, color='tab:blue', linestyle='--', lw=1, label='Accl. Y-axis')
    ax.plot(np.linspace(0, z.size - 1, z.size), z, color='tab:blue', linestyle='-.', lw=1, label='Accl. Z-axis')
    ax.legend(loc='lower right')
    ax.set_ylim(-2.5, 1)

    ax.set_title('Raw acceleration along 3-D coordinates')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Acceleration [g]')

    caption_text = r"""A time-series plot of the acceleration along 3-D coordinates. Time 
is represented in milliseconds along the horizontal axis while 
acceleration [g; 1g = 9.8m/$s^{2}$] is along the veritcal axis."""

    # plt.figtext(0.15, -0.1, caption_text, wrap=False, horizontalalignment='left',
    #             fontsize=9)
    plt.savefig('../figures/sampled_timeseries.png', dpi=300)


def build_time_domain_features(time_windows):
    """Build set of time domain features for each time window. Returns reduced dataframe"""

    def td_sum(x):
        return np.sum(x)

    def td_mean(x):
        return np.mean(x)

    def td_std(x):
        return np.std(x)

    def td_percentile(x, p):
        return np.percentile(x, p)

    def td_iqr(x):
        return np.percentile(x, 75) - np.percentile(x, 25)

    def td_range(x):
        return np.max(x) - np.min(x)

    def td_power(x):
        return np.sum(np.square(x))

    def td_log_power(x, eps=10e-6):
        x = np.add(x, eps)
        return np.sum(np.log(np.square(x)))

    def td_energy(x, y, z):
        ex = np.sqrt(np.sum(np.square(np.subtract(x, np.mean(x)))))
        ey = np.sqrt(np.sum(np.square(np.subtract(y, np.mean(y)))))
        ez = np.sqrt(np.sum(np.square(np.subtract(z, np.mean(z)))))

        e = (1 / (3 * len(x))) * (ex + ey + ez)
        return e

    def td_rms(x):
        return np.sqrt(np.mean(np.square(x)))

    time_domain_func = [td_sum, td_mean, td_std, td_iqr, td_range, td_power, td_log_power, td_rms, td_energy]
    time_domain_col = ['epoch_start', 'epoch_end', 'user', 'class']

    for func in time_domain_func:
        if (func in [td_energy]):
            time_domain_col.append(func.__name__)
            continue
        for axis in ['x', 'y', 'z', 'm']:
            time_domain_col.append(func.__name__ + '_' + axis)

    time_domain_features = []

    for useract_winset in time_windows:
        for twdf in useract_winset:
            twdf_features = []
            twdf_features.extend([twdf['epoch'].min(), twdf['epoch'].max(),
                                  twdf['user'].unique()[0], twdf['class'].unique()[0]])

            for func in time_domain_func:
                if (func in [td_energy]):
                    twdf_features.extend([func(twdf['x'].values, twdf['y'].values, twdf['z'].values)])
                    continue
                twdf_features.extend([func(twdf['x'].values), func(twdf['y'].values),
                                      func(twdf['z'].values), func(twdf['m'].values)])

            time_domain_features.append(twdf_features)

    time_domain_df = pd.DataFrame(time_domain_features, columns=time_domain_col)
    return time_domain_df


def build_freq_domain_features(time_windows):
    """Build set of frequency domain features for each time window. Returns reduced dataframe"""

    def fd_dominant_freq(x, t):
        """Returns the dominant frequency of x in Hz."""
        n = x.size
        dt = t / n
        df = 1 / t

        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(n) * n * df

        if n % 2 == 0:
            eff_size = int(n / 2)
        else:
            eff_size = int((n + 1) / 2)

        idx = np.argmax(np.abs(X[1:eff_size]))
        freq = freqs[1:eff_size][idx]
        return freq

    def fd_spectral_entropy(x):
        """Returns the spectral entropy of a signal"""
        n = x.size
        p = np.square(x) / n
        pi = p / np.sum(p) + 1e-15
        H = -np.sum(pi * np.log2(pi)) / (np.log2(n))
        return H

    def fd_step_counter(m):
        """Estimates number of steps based on number of peaks in the mag vector"""

        def _filter_signal(x, samp_rate=100, high=2, order=5):
            nyq = 0.5 * samp_rate
            high = high / nyq
            b, a = butter(order, high)
            y = lfilter(b, a, x)
            return y

        y = _filter_signal(m, samp_rate=100, high=2)
        peaks = detect_peaks(y, mph=1.2)
        steps = len(peaks)
        return steps

    freq_domain_func = [fd_dominant_freq, fd_step_counter, fd_spectral_entropy]
    freq_domain_col = ['epoch_start', 'epoch_end', 'user', 'class']

    for func in freq_domain_func:
        for axis in ['x', 'y', 'z', 'm']:
            freq_domain_col.append(func.__name__ + '_' + axis)

    freq_domain_features = []

    for useract_winset in time_windows:
        for twdf in useract_winset:
            twdf_features = []
            twdf_features.extend([twdf['epoch'].min(), twdf['epoch'].max(),
                                  twdf['user'].unique()[0], twdf['class'].unique()[0]])

            signal_secs = (twdf['epoch'].max() - twdf['epoch'].min()) / 1000

            for func in freq_domain_func:
                if func in [fd_dominant_freq]:
                    twdf_features.extend([func(twdf['x'].values, signal_secs), func(twdf['y'].values, signal_secs),
                                          func(twdf['z'].values, signal_secs), func(twdf['m'].values, signal_secs)])
                else:
                    twdf_features.extend([func(twdf['x'].values), func(twdf['y'].values),
                                          func(twdf['z'].values), func(twdf['m'].values)])

            freq_domain_features.append(twdf_features)

    freq_domain_df = pd.DataFrame(freq_domain_features, columns=freq_domain_col)
    return freq_domain_df


if __name__ == '__main__':

    with open('../data/wrist_dump.pkl', 'rb') as f:
        person_data = pickle.load(f)

    time_windows = []
    for person, activity_data in person_data.items():
        for activity, activity_df in activity_data.items():
            time_windows.append(build_time_windows(activity_df))

    plot_window_data(time_windows[12][0])

    time_domain_df = build_time_domain_features(time_windows)
    freq_domain_df = build_freq_domain_features(time_windows)

    time_freq_domain_df = pd.merge(left=time_domain_df, right=freq_domain_df, left_index=True, right_index=True,
                                   how='outer', suffixes=('', '_drop'))
    time_freq_domain_df.drop(columns=[col_name for col_name in time_freq_domain_df.columns if '_drop' in col_name],
                             inplace=True)

    time_domain_df.to_csv('../data/time_domain_windows.csv', index=False)
    freq_domain_df.to_csv('../data/freq_domain_windows.csv', index=False)
    time_freq_domain_df.to_csv('../data/time_freq_domain_windows.csv', index=False)
