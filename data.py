import numpy as np
from typing import Tuple
import pandas as pd
import numpy as np
from typing import Tuple
import yfinance as yf

def sample_non_overlapping_intervals(
    n: int,
    k: int,
    Lmin: int,
    Lmax: int,
    rng: np.random.Generator,
    occupied: np.ndarray | None = None
) -> Tuple[list[tuple[int, int]], np.ndarray]:
    """
    Sample k non-overlapping closed-open intervals [start, end) on a 1D timeline.

    Each interval length L is drawn uniformly from [Lmin, Lmax], and intervals are
    placed uniformly at random subject to a binary occupancy mask.

    Parameters
    ----------
    n : int
        Total number of discrete time steps on the line (length of the mask).
    k : int
        Number of intervals to place.
    Lmin : int
        Minimum interval length (inclusive).
    Lmax : int
        Maximum interval length (inclusive).
    rng : np.random.Generator
        Numpy random generator used for reproducibility.
    occupied : np.ndarray | None
        Optional boolean mask of shape (n,) marking already-occupied positions.
        If None, a fresh all-False mask is created.

    Returns
    -------
    list[tuple[int, int]], np.ndarray
        A list of k intervals as (start, end) pairs sorted by start,
        and the updated boolean occupancy mask of shape (n,).

    Raises
    ------
    ValueError
        If requested lengths are incompatible with n.
    RuntimeError
        If k intervals cannot be placed without overlap within a fixed attempt budget.
    """
    if occupied is None:
        occupied = np.zeros(n, dtype=bool)

    intervals: list[tuple[int, int]] = []
    attempts, max_attempts = 0, 50_000

    while len(intervals) < k and attempts < max_attempts:
        length = int(rng.integers(Lmin, Lmax + 1))
        if length >= n:
            raise ValueError("Lmin/Lmax too large compared to n.")
        start = int(rng.integers(0, n - length))
        end = start + length
        if not occupied[start:end].any():
            intervals.append((start, end))
            occupied[start:end] = True
        attempts += 1

    if len(intervals) < k:
        raise RuntimeError(
            "Could not place all intervals without overlap. Reduce k, "
            "reduce Lmin/Lmax, or increase n."
        )

    intervals.sort(key=lambda t: t[0])
    return intervals, occupied


def build_global_masks(n, m_global, p_global, Lmin, Lmax, rng):
    """
    Build global crisis/growth masks shared by all series on a length-n timeline.

    Global masks represent system-wide regimes (e.g., macro shocks) that affect all series.

    Parameters
    ----------
    n : int
        Number of time steps.
    m_global : int
        Number of global crisis intervals to place.
    p_global : int
        Number of global growth intervals to place.
    Lmin : int
        Minimum interval length (inclusive).
    Lmax : int
        Maximum interval length (inclusive).
    rng : np.random.Generator
        Random generator for reproducibility.

    Returns
    -------
    list[tuple[int, int]], list[tuple[int, int]], np.ndarray, np.ndarray
        The list of crisis intervals, the list of growth intervals,
        the boolean crisis mask of shape (n,), and the boolean growth mask of shape (n,).

    Raises
    ------
    AssertionError
        If any global crisis and growth segments overlap.
    """
    occupied_mask = np.zeros(n, dtype=bool)

    crisis_windows_global, occupied_mask = sample_non_overlapping_intervals(
        n, m_global, Lmin, Lmax, rng, occupied_mask
    )
    growth_windows_global, occupied_mask = sample_non_overlapping_intervals(
        n, p_global, Lmin, Lmax, rng, occupied_mask
    )

    crisis_mask_global = np.zeros(n, dtype=bool)
    growth_mask_global = np.zeros(n, dtype=bool)

    for s, e in crisis_windows_global:
        crisis_mask_global[s:e] = True
    for s, e in growth_windows_global:
        growth_mask_global[s:e] = True

    assert not np.any(crisis_mask_global & growth_mask_global), "Global crisis/growth overlap detected."
    return crisis_windows_global, growth_windows_global, crisis_mask_global, growth_mask_global


def build_personal_masks(n_curves, n, m_personal, p_personal, Lmin, Lmax, rng):
    """
    Build per-series crisis/growth masks on a shared length-n timeline.

    Personal masks capture idiosyncratic regimes unique to each series.

    Parameters
    ----------
    n_curves : int
        Number of series to simulate.
    n : int
        Number of time steps per series.
    m_personal : int
        Number of personal crisis intervals per series.
    p_personal : int
        Number of personal growth intervals per series.
    Lmin : int
        Minimum interval length (inclusive).
    Lmax : int
        Maximum interval length (inclusive).
    rng : np.random.Generator
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray, np.ndarray
        Personal crisis mask of shape (n_curves, n) and personal growth mask of shape (n_curves, n).

    Raises
    ------
    AssertionError
        If any personal crisis and growth segments overlap within a series.
    """
    crisis_mask_personal = np.zeros((n_curves, n), dtype=bool)
    growth_mask_personal = np.zeros((n_curves, n), dtype=bool)

    for idx in range(n_curves):
        occupancy = np.zeros(n, dtype=bool)
        crisis_windows, occupancy = sample_non_overlapping_intervals(n, m_personal, Lmin, Lmax, rng, occupancy)
        growth_windows, occupancy = sample_non_overlapping_intervals(n, p_personal, Lmin, Lmax, rng, occupancy)

        for s, e in crisis_windows:
            crisis_mask_personal[idx, s:e] = True
        for s, e in growth_windows:
            growth_mask_personal[idx, s:e] = True

    assert not np.any(crisis_mask_personal & growth_mask_personal), "Personal crisis/growth overlap detected."
    return crisis_mask_personal, growth_mask_personal


def simulate_curves(
    n,
    n_curves,
    lookback,
    rng,
    m_global,
    p_global,
    m_personal,
    p_personal,
    Lmin,
    Lmax,
    u_limit,
    d_limit,
    omega,
    l_noise_factor,
    u_noise_factor,
    A=None
):
    """
    Simulate n_curves time series of length n with global/personal regime shocks and VAR-like dynamics.

    Dynamics combine:
      1) Idiosyncratic innovations,
      2) A decaying multi-lag coupling kernel across series (akin to a VAR with lookback lags),
      3) Additive regime effects from global and personal crisis/growth masks,
      4) Optional bounding of the state.

    Parameters
    ----------
    n : int
        Number of time steps per series.
    n_curves : int
        Number of series to simulate.
    lookback : int
        Number of past increments used in the coupling kernel.
    rng : np.random.Generator
        Random generator for reproducibility.
    m_global : int
        Count of global crisis intervals.
    p_global : int
        Count of global growth intervals.
    m_personal : int
        Count of personal crisis intervals per series.
    p_personal : int
        Count of personal growth intervals per series.
    Lmin : int
        Minimum interval length (inclusive) for all intervals.
    Lmax : int
        Maximum interval length (inclusive) for all intervals.
    u_limit : float
        Upper bound applied to the evolving state (saturation).
    d_limit : float
        Lower bound applied to the evolving state (saturation).
    omega : float
        Geometric decay factor for lag weights in the coupling kernel.
    l_noise_factor : float
        Lower bound for per-series noise scaling.
    u_noise_factor : float
        Upper bound for per-series noise scaling.
    A : np.ndarray | None
        Optional pre-specified coupling kernels of shape (n_curves, n_curves, lookback).
        If None, kernels are sampled and decayed by omega, with a self-weight on the last lag.

    Returns
    -------
    np.ndarray, list[tuple[int, int]], list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray
        simulations : array of shape (n_curves, n)
            Simulated series.
        crises_global : list[(start, end)]
            Global crisis intervals.
        growths_global : list[(start, end)]
            Global growth intervals.
        mask_crisis_personal : (n_curves, n) boolean
            Personal crisis mask.
        mask_growth_personal : (n_curves, n) boolean
            Personal growth mask.
        mask_crisis_global : (n,) boolean
            Global crisis mask.
        mask_growth_global : (n,) boolean
            Global growth mask.

    Notes
    -----
    The coupling kernel operates on recent increments of the state, similar in spirit to
    a moving-average view of a VAR where past differences are linearly combined with
    decaying weights. See the VAR/MA discussion for stationarity and impulse responses. :contentReference[oaicite:1]{index=1}
    """
    self_weight = 0.6 + 0.2 * rng.uniform(0, 1, n_curves)
    noise_scale = rng.uniform(l_noise_factor, u_noise_factor, n_curves)
    alpha_crisis_global = rng.uniform(0.1, 0.8, n_curves)
    alpha_growth_global = rng.uniform(0.1, 0.8, n_curves)
    alpha_crisis_personal = rng.uniform(0.1, 0.8, n_curves)
    alpha_growth_personal = rng.uniform(0.1, 0.8, n_curves)
    trend_scale = rng.integers(5000, 10000, n_curves)

    if A is None:
        lag_kernels = rng.standard_normal((n_curves, n_curves, lookback))
        decay = omega ** (np.arange(lookback)[::-1] + 1)
        lag_kernels *= decay[None, None, :]
        rows, cols = np.diag_indices(n_curves)
        lag_kernels[rows, cols, -1] = self_weight
    else:
        lag_kernels = A

    innovation = rng.standard_normal((n_curves, n))

    crises_global, growths_global, mask_crisis_global, mask_growth_global = build_global_masks(
        n, m_global, p_global, Lmin, Lmax, rng
    )
    mask_crisis_personal, mask_growth_personal = build_personal_masks(
        n_curves, n, m_personal, p_personal, Lmin, Lmax, rng
    )

    state = np.zeros((n_curves, n))
    state[:, 0] = innovation[:, 0]
    state[:, 1] = state[:, 0] + innovation[:, 1]

    for t in range(1, n - 1):
        start_idx = max(t - lookback, 0)
        effective_lags = min(t, lookback)

        diffs_lookback = state[:, start_idx + 1 : t + 1] - state[:, start_idx : t]
        kernel_slice = lag_kernels[:, :, (lookback - effective_lags) :]

        coupled = np.einsum('abi,bi->a', kernel_slice, diffs_lookback)
        core = noise_scale * innovation[:, t + 1] + coupled

        state[:, t + 1] = state[:, t] + core

        state[:, t + 1] += mask_growth_global[t] * alpha_growth_global
        state[:, t + 1] -= mask_crisis_global[t] * alpha_crisis_global
        state[:, t + 1] += alpha_growth_personal * mask_growth_personal[:, t]
        state[:, t + 1] -= alpha_crisis_personal * mask_crisis_personal[:, t]

        state[state[:, t + 1] > u_limit] = u_limit
        state[state[:, t + 1] < d_limit] = d_limit

    x_axis = np.linspace(0.0, 1.0, n)
    x_axis = np.tile(x_axis, (n_curves, 1))

    simulations = trend_scale[:, None] * x_axis + state

    return (
        simulations,
        crises_global,
        growths_global,
        mask_crisis_personal,
        mask_growth_personal,
        mask_crisis_global,
        mask_growth_global,
    )
    
# def fetch_french_stocks(n_comp=50, period="6mo", interval="1d"):
#     """
#     Download adjusted close prices for CAC 40 components plus the CAC index.

#     Parameters
#     ----------
#     n_comp : int, default 50
#         Number of component tickers to include from the predefined list.
#     period : str, default "6mo"
#         yfinance period string (e.g., "1y", "6mo", "5d").
#     interval : str, default "1d"
#         yfinance interval string (e.g., "1d", "1h", "5m").

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame of closing prices with columns being tickers and rows being timestamps.

#     Notes
#     -----
#     Falls back to the single-level "Close" column layout if yfinance returns it that way.
#     """
#     cac_ticker = "^FCHI"
#     components = [
#         "AIR.PA","AI.PA","SU.PA","OR.PA","BNP.PA","MC.PA","SAN.PA","KER.PA","TTE.PA","EL.PA",
#         "DG.PA","GLE.PA","ENGI.PA","VIE.PA","ACA.PA","SGO.PA","RI.PA","HO.PA","VIV.PA","CAP.PA",
#         "STM.PA","EN.PA","CA.PA","ALO.PA","FR.PA","SAF.PA","DSY.PA","LR.PA","PUB.PA","SOLB.PA",
#         "ACA.PA","ALO.PA","BN.PA","BOUY.PA","EDF.PA","EXHO.PA","RMS.PA","ATO.PA","POM.PA","ML.PA",
#         "BVI.PA","FNAC.PA","RNO.PA","UG.PA","IL.PA","ORP.PA","CNP.PA","ACC.PA","ALU.PA","SW.PA"
#     ]
#     tickers = components[:n_comp] + [cac_ticker]
#     data = yf.download(tickers, period=period, interval=interval, group_by="ticker", progress=False)
#     closes = {}
#     for t in tickers:
#         if (t, "Close") in getattr(data, "columns", []):
#             closes[t] = data[(t, "Close")]
#         else:
#             closes[t] = data["Close"]
#     df_closes = pd.DataFrame(closes)
#     return df_closes


def fetch_french_stocks_interleaved(n_comp=50, period="6mo", interval="1d"):
    cac_ticker = "^FCHI"
    components = [
        "AIR.PA","AI.PA","SU.PA","OR.PA","BNP.PA","MC.PA","SAN.PA","KER.PA","TTE.PA","EL.PA",
        "DG.PA","GLE.PA","ENGI.PA","VIE.PA","ACA.PA","SGO.PA","RI.PA","HO.PA","VIV.PA","CAP.PA",
        "STM.PA","EN.PA","CA.PA","ALO.PA","FR.PA","SAF.PA","DSY.PA","LR.PA","PUB.PA","SOLB.PA",
        "ACA.PA","ALO.PA","BN.PA","BOUY.PA","EDF.PA","EXHO.PA","RMS.PA","ATO.PA","POM.PA","ML.PA",
        "BVI.PA","FNAC.PA","RNO.PA","UG.PA","IL.PA","ORP.PA","CNP.PA","ACC.PA","ALU.PA","SW.PA"
    ]
    seen=set(); components=[t for t in components if not (t in seen or seen.add(t))]
    tickers = components[:n_comp] + [cac_ticker]
    seen=set(); tickers=[t for t in tickers if not (t in seen or seen.add(t))]

    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,   
        progress=False,
        threads=True,
    )

    if data.empty:
        raise RuntimeError("Download returned empty DataFrame.")

    if isinstance(data.columns, pd.MultiIndex):
        
        fields = set(data.columns)
        kept = [t for t in tickers if (t, "Open") in fields and (t, "Close") in fields]
        if not kept:
            raise RuntimeError("No ticker has both Open and Close.")
        opens_df  = data.loc[:, (kept, "Open")]
        closes_df = data.loc[:, (kept, "Close")]
        
        opens_df.columns = opens_df.columns.get_level_values(0)
        closes_df.columns = closes_df.columns.get_level_values(0)
    else:
       
        if "Open" not in data.columns or "Close" not in data.columns:
            raise RuntimeError("Flat layout but missing Open/Close.")
        kept = tickers[:1]  
        opens_df  = data[["Open"]].rename(columns={"Open": kept[0]})
        closes_df = data[["Close"]].rename(columns={"Close": kept[0]})

  
    idx = opens_df.index.intersection(closes_df.index)
    opens_df  = opens_df.reindex(idx)
    closes_df = closes_df.reindex(idx)

    n_dates = len(idx)
    n_cols  = len(kept)

    interleaved_values = np.empty((2 * n_dates, n_cols), dtype=float)
    interleaved_values[0::2] = opens_df.values
    interleaved_values[1::2] = closes_df.values

    interleaved_index = []
    for ts in idx:
        interleaved_index.extend([ts, ts])

    df_final = pd.DataFrame(
        interleaved_values,
        columns=kept,
        index=pd.Index(interleaved_index, name="Date")
    )
    return df_final, kept

def fill_nan_with_neighbors(df):
    """
    Fill NaN values by linear interpolation between their nearest known neighbors.
    Leaves leading/trailing NaNs unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of df with internal NaNs linearly interpolated.
    """
    return df.interpolate(method="linear", axis=0, limit_direction="both")


def drop_nan_cols(df, threshold):
    """
    Drop columns with too many NaNs based on a minimum non-NaN count.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float
        Fractional tolerance for NaNs converted to a minimum non-NaN count as
        floor((1 - threshold) * len(df)) for pandas' `thresh` parameter.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns dropped if they fail the non-NaN count threshold.

    Notes
    -----
    This interprets `threshold` as a fraction of allowed NaNs; higher values allow more NaNs.
    """
    min_non_nan = int((1 - threshold) * len(df))
    return df.dropna(axis=1, thresh=min_non_nan)

def train_test_split(X, percent, step):
    """
    Downsample, difference, split, and rescale a multivariate series into train/test sets.

    Parameters
    ----------
    X : np.ndarray, shape (k, T)
        Input multivariate series (k variables by T time steps).
    percent : float
        Fraction of the differenced samples to allocate to the training set (in (0, 1]).
    step : int
        Downsampling stride along the time axis; X is sliced as X[:, ::step].

    Returns
    -------
    train : np.ndarray, shape (k, floor(percent * (T' - 1)))
        Row-wise min–max scaled to [-1, 1] after differencing and splitting.
    test : np.ndarray, shape (k, (T' - 1) - floor(percent * (T' - 1)))
        Row-wise min–max scaled to [-1, 1] after differencing and splitting.

    Notes
    -----
    NaNs in the downsampled series are set to 0. Differencing is first-order along time.
    Scaling is applied independently per row on each split (train and test separately).
    """
    X_ds = X[:, ::step]
    X_ds[np.isnan(X_ds)] = 0.0
    X_diff = np.diff(X_ds, axis=1)

    n_split = int(percent * X_diff.shape[1])
    train = X_diff[:, :n_split]
    test = X_diff[:, n_split:]

    train = 2 * ((train - train.min(axis=1)[:, None]) /
                 (train.max(axis=1)[:, None] - train.min(axis=1)[:, None])) - 1
    test = 2 * ((test - test.min(axis=1)[:, None]) /
                (test.max(axis=1)[:, None] - test.min(axis=1)[:, None])) - 1
    return train, test


def simulate_groups(n_groups, rng, n=10000, lookback=20, n_curves=50, omega=0.01):
    """
    Generate multiple labeled groups of simulated coupled time series.

    Parameters
    ----------
    n_groups : int
        Number of distinct groups (labels 0..n_groups-1) to simulate.
    rng : np.random.Generator
        Random generator used by the underlying simulator.
    n : int, default 10000
        Number of time steps per simulation.
    lookback : int, default 20
        Maximum lag order for the coupling kernel in the simulator.
    n_curves : int, default 50
        Number of series per group simulation.
    omega : float, default 0.01
        Exponential decay factor for lag weights in the simulator.

    Returns
    -------
    sims : np.ndarray, shape (n_groups * n_curves, n)
        Concatenated simulations from all groups (stacked along the first axis).
    labels : np.ndarray, shape (n_groups * n_curves,)
        Integer group label for each simulated series, aligned with `sims`.

    Notes
    -----
    Uses `simulate_curves` to sample each group's dynamics with fixed regime parameters.
    """
    sims_list = []
    group_labels = []
    for g in range(n_groups):
        sims_g, _, _, _, _, _, _ = simulate_curves(
            n=n,
            n_curves=n_curves,
            lookback=lookback,
            rng=rng,
            m_global=3,
            p_global=2,
            m_personal=2,
            p_personal=2,
            Lmin=200,
            Lmax=1000,
            u_limit=100000,
            d_limit=-50000,
            omega=omega,
            l_noise_factor=5,
            u_noise_factor=20
        )
        sims_list.append(sims_g)
        group_labels.extend([g] * sims_g.shape[0])
    return np.concatenate(sims_list, axis=0), np.array(group_labels)


def random_index_tuples(n, k):
    """
    Sample k random index pairs for an n×n matrix.

    Parameters
    ----------
    n : int
        Matrix dimension.
    k : int
        Number of index pairs to generate.

    Returns
    -------
    list[tuple[int, int]]
        List of (row, col) index tuples, each in [0, n-1].
    """
    rows = np.random.randint(0, n, size=k)
    cols = np.random.randint(0, n, size=k)
    return list(zip(rows, cols))