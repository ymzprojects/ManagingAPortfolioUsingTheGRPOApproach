import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.style.use('dark_background')
def plot_curves_grid(x, y, xlabels, ylabels, titles, ncols=3, figsize_scale=3):
    """
    Plot multiple (x, y) curves in a grid layout.

    Parameters
    ----------
    x : np.ndarray, shape (n_curves, T)
        X-coordinates for each curve.
    y : np.ndarray, shape (n_curves, T)
        Y-coordinates for each curve.
    xlabels : list[str]
        X-axis labels for each subplot.
    ylabels : list[str]
        Y-axis labels for each subplot.
    titles : list[str]
        Titles for each subplot.
    ncols : int, default 3
        Number of columns in the figure grid.
    figsize_scale : float, default 3
        Size multiplier for the overall figure.

    Returns
    -------
    None
        Displays a grid of line plots.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n_curves = x.shape[0]
    ncols = max(1, min(ncols, n_curves))
    nrows = int(np.ceil(n_curves / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_scale * ncols, figsize_scale * nrows),
        squeeze=False
    )
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx < n_curves:
            ax.plot(x[idx], y[idx])
            ax.set_title(titles[idx])
            ax.set_xlabel(xlabels[idx])
            ax.set_ylabel(ylabels[idx])
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_hist_grid(sets, bins=30, ncols=3, figsize_scale=3.2, title="", labels=None,
                   a=None, b=None):
    """
    Plot a grid of histograms, optionally restricting values to the range [a, b].

    Parameters
    ----------
    sets : np.ndarray, shape (n_sets, T)
        Each row is a dataset to be plotted as one histogram.
    bins : int or array-like, default 30
        Number of bins or explicit bin edges.
    ncols : int, default 3
        Number of columns in the histogram grid.
    figsize_scale : float, default 3.2
        Scaling factor for figure size.
    title : str, default ""
        Global title for the entire figure.
    labels : list[str] or None, default None
        Per-histogram titles; defaults to "Set i" if None.
    a, b : float or None, optional
        Lower and upper bounds for data display. Only values within [a, b]
        will be included in each histogram.

    Returns
    -------
    None
        Displays histograms with a shared title.
    """
    sets = np.asarray(sets)
    n_sets, _ = sets.shape
    ncols = max(1, min(ncols, n_sets))
    nrows = int(np.ceil(n_sets / ncols))


    if a is not None or b is not None:
        clipped_sets = []
        for data in sets:
            data_clipped = data.copy()
            if a is not None:
                data_clipped = data_clipped[data_clipped >= a]
            if b is not None:
                data_clipped = data_clipped[data_clipped <= b]
            clipped_sets.append(data_clipped)
        sets = clipped_sets
    else:
        sets = [s for s in sets]

    all_data = np.concatenate([s[~np.isnan(s)] for s in sets])
    vmin = np.nanmin(all_data)
    vmax = np.nanmax(all_data)
    bin_edges = (
        np.linspace(vmin, vmax, bins + 1)
        if isinstance(bins, int)
        else np.asarray(bins)
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_scale * ncols, figsize_scale * nrows),
        squeeze=False
    )

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx < len(sets):
            data = sets[idx]
            if data.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ax.hist(data[~np.isnan(data)], bins=bin_edges, color="steelblue", alpha=0.7)
            if labels is None:
                ax.set_title(f"Set {idx}")
            else:
                ax.set_title(labels[idx])
            ax.set_xlim(vmin, vmax)
        else:
            ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_groups_fast(sims, labels, alpha=0.6, lw=0.7):
    """
    Quickly plot time series grouped by label, each group in a different color.

    Parameters
    ----------
    sims : np.ndarray, shape (N, T)
        Collection of curves to be plotted, one per row.
    labels : array-like, shape (N,)
        Group label for each curve.
    alpha : float, default 0.6
        Opacity for the line plots.
    lw : float, default 0.7
        Line width for the curves.

    Returns
    -------
    None
        Displays a figure where each group is color-coded.
    """
    labels = np.asarray(labels)
    groups = np.unique(labels)
    x = np.arange(sims.shape[1])
    cmap = plt.get_cmap('tab20' if len(groups) <= 20 else 'hsv', len(groups))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for idx, g in enumerate(groups):
        curves_group = sims[labels == g]
        ax.plot(x, curves_group.T, color=cmap(idx), alpha=alpha, linewidth=lw)

    legend_handles = [
        plt.Line2D([0], [0], color=cmap(i), lw=2) for i in range(len(groups))
    ]
    ax.legend(legend_handles, [f'g{g}' for g in groups], ncol=3, loc='best')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Curves by group')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_actual_and_prediction(y_true, y_pred, title="Actual (white) with Prediction overlay", w=1.0):
    """
    Compare actual and predicted time series via bar overlay and sign-based coloring.

    Parameters
    ----------
    y_true : array-like
        Actual values for the series.
    y_pred : array-like
        Corresponding predictions to compare.
    title : str, default "Actual (white) with Prediction overlay"
        Plot title.
    w : float, default 1.0
        Bar width for the plot.

    Returns
    -------
    None
        Displays a bar-based comparison with color-coded sign agreement.
    """
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    mask_valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask_valid], y_pred[mask_valid]
    x = np.arange(len(y_true))

    same_sign = np.sign(y_true) == np.sign(y_pred)
    colors = np.where(same_sign, "#1f77b4", "#d62728")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhline(0, linewidth=1)

    ax.bar(x, y_true, width=w, color="white", zorder=2)
    ax.bar(x, y_pred, width=w, color=colors, alpha=0.6, zorder=3)

    lim = 1.1 * np.nanmax(
        np.abs(np.concatenate([y_true, y_pred]))
    ) if len(y_true) else 1.0
    if not np.isfinite(lim) or lim == 0:
        lim = 1.0

    ax.set_ylim(-lim, lim)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Variation")
    ax.set_title(title)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="white"),
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
        plt.Rectangle((0, 0), 1, 1, color="#d62728")
    ]
    ax.legend(
        legend_handles,
        ["Actual", "Pred (same dir)", "Pred (opp. dir)"],
        ncol=3,
        frameon=False
    )
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    
def simple_scatter(x, y, xlabel="", ylabel="", title=""):
    """
    Create a basic scatter plot of two arrays.

    Parameters
    ----------
    x : array-like
        Values for the horizontal axis.
    y : array-like
        Values for the vertical axis.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    title : str, optional
        Figure title.

    Returns
    -------
    None
        Displays the scatter plot.
    """
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_score_grid(scores, curve_labels=None, p_labels=None, title=None):
    """
    Display a heatmap of scores by curve and lag order.

    Parameters
    ----------
    scores : array-like, shape (p, n_curves)
        Matrix of values in [0,1] or similar range. Rows correspond to lag indices,
        columns correspond to curves. The transpose is plotted so curves appear vertically.
    curve_labels : list[str], optional
        Labels for each curve. If None, defaults to "curve{i}".
    p_labels : list[str], optional
        Labels for lag indices. If None, defaults to "p{j}".
    title : str, optional
        Title for the figure.

    Returns
    -------
    None
        Displays a color-coded grid of scores.
    """
    scores = np.asarray(scores, float)
    p, n_curves = scores.shape
    M = scores.T

    cmap = LinearSegmentedColormap.from_list("red_blue", [(1, 0, 0), (0, 0, 1)])

    fig, ax = plt.subplots(figsize=(max(6, p * 0.5), max(4, n_curves * 0.4)))
    im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto", origin="upper")

    ax.set_xticks(np.arange(p))
    ax.set_yticks(np.arange(n_curves))
    if p_labels is not None:
        ax.set_xticklabels(p_labels)
    else:
        ax.set_xticklabels([f"p{j}" for j in range(p)])
    if curve_labels is not None:
        ax.set_yticklabels(curve_labels)
    else:
        ax.set_yticklabels([f"curve{i}" for i in range(n_curves)])

    ax.set_xlabel("lags (p)")
    ax.set_ylabel("curves")
    if title:
        ax.set_title(title)

    ax.set_xticks(np.arange(-0.5, p, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_curves, 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("score (0=rouge, 1=bleu)")

    plt.tight_layout()
    plt.show()


def plot_signals(
    C,
    crises_global,
    growths_global,
    mask_crisis_personal,
    mask_growth_personal
):
    """
    Plot multiple signals with shaded global regimes and highlighted personal ones.

    Parameters
    ----------
    C : np.ndarray, shape (n_curves, n)
        Matrix of signals, one per row.
    crises_global : list of (start, end)
        Intervals marking global crisis periods.
    growths_global : list of (start, end)
        Intervals marking global growth periods.
    mask_crisis_personal : np.ndarray, shape (n_curves, n)
        Boolean mask for personal crises per curve.
    mask_growth_personal : np.ndarray, shape (n_curves, n)
        Boolean mask for personal growth phases per curve.

    Returns
    -------
    None
        Displays the signals with overlays indicating regimes.
    """
    n_curves, n = C.shape
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    first_crisis, first_growth = True, True
    for s, e in crises_global:
        ax.axvspan(x[s], x[e - 1], color="red", alpha=0.12,
                   label="Global Crisis" if first_crisis else None)
        first_crisis = False
    for s, e in growths_global:
        ax.axvspan(x[s], x[e - 1], color="green", alpha=0.10,
                   label="Global Growth" if first_growth else None)
        first_growth = False

    for k in range(n_curves):
        ax.plot(x, C[k], color="white", linewidth=0.9, alpha=0.7,
                label="Signals (white)" if k == 0 else None)

    for k in range(n_curves):
        y_crisis = np.where(mask_crisis_personal[k], C[k], np.nan)
        ax.plot(x, y_crisis, color="red", linewidth=1.8, alpha=0.95,
                label="Personal Crisis" if k == 0 else None, zorder=3)

        y_growth = np.where(mask_growth_personal[k], C[k], np.nan)
        ax.plot(x, y_growth, color="green", linewidth=1.8, alpha=0.95,
                label="Personal Growth" if k == 0 else None, zorder=3)

    ax.set_xlabel("Time (x)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{n_curves} Signals — white base; red personal crises; green personal growth")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    plt.tight_layout()
    plt.show()

    
def plot_real_vs_predictions(
    real_curve,
    pred_curves,
    x=None,
    H=None,
    labels=None,
    title="Real vs Predictions",
    figsize=(10, 5),
    grid_alpha=0.3,
    real_style=dict(linewidth=2, label="Real"),
    pred_style=dict(linewidth=2, linestyle="--"),
    show=True,
    save_path=None,
):
    """
    Plot a single real time series against one or multiple prediction series.

    Parameters
    ----------
    real_curve : array-like, shape (T,) or (T,1)
        Ground-truth series to display.
    pred_curves : array-like or dict
        Prediction series. Accepted formats:
          - dict: {name -> series}
          - 1D array: a single prediction
          - 2D array: each row is one prediction series
          - iterable of arrays: list/tuple of prediction series
    x : array-like or None, optional
        Optional x-axis values; if None, uses np.arange(T).
    H : int or None, optional
        Optional horizon limiting the plotted length to min(T, H).
    labels : list[str] or None, optional
        Labels for prediction series when `pred_curves` is not a dict.
        Must match the number of series.
    title : str, default "Real vs Predictions"
        Title of the figure.
    figsize : tuple, default (10, 5)
        Matplotlib figure size.
    grid_alpha : float, default 0.3
        Grid line transparency.
    real_style : dict, optional
        Matplotlib style kwargs for the real series (e.g., linewidth, color, label).
        If 'color' is omitted or None, Matplotlib’s default color is used.
    pred_style : dict, optional
        Matplotlib style kwargs shared by all prediction series
        (e.g., linewidth, linestyle, color).
    show : bool, default True
        If True, calls plt.show() at the end.
    save_path : str or None, optional
        If provided, saves the figure to this path (dpi=150, tight bbox).

    Returns
    -------
    None
        Displays (and optionally saves) the comparison plot.

    Notes
    -----
    The function harmonizes the horizon across real and predicted series,
    truncating to the smallest common length (and optionally to H).
    """
    real = np.asarray(real_curve, dtype=float).ravel()

    pred_list = []
    if isinstance(pred_curves, dict):
        for k, v in pred_curves.items():
            pred_list.append((str(k), np.asarray(v, dtype=float).ravel()))
    else:
        arr = np.asarray(pred_curves, dtype=float)
        if arr.ndim == 1:
            series = [arr]
        elif arr.ndim == 2:
            n_rows, n_cols = arr.shape
            series = [arr[i, :] for i in range(n_rows)]
        else:
            try:
                series = [np.asarray(v, dtype=float).ravel() for v in pred_curves]
            except Exception:
                raise ValueError("Unsupported pred_curves structure.")
        if labels is None:
            labels = [f"Pred {i+1}" for i in range(len(series))]
        elif len(labels) != len(series):
            raise ValueError("`labels` length must match number of prediction series.")
        pred_list = list(zip(labels, series))

    if len(pred_list) == 0:
        raise ValueError("No prediction series found in `pred_curves`.")

    lengths = [len(real)] + [len(p) for _, p in pred_list]
    T = min(lengths)
    if H is not None:
        T = min(T, int(H))
    if T <= 0:
        raise ValueError("Computed horizon is non-positive. Check your inputs.")

    real = real[:T]
    pred_list = [(lab, p[:T]) for lab, p in pred_list]

    if x is None:
        x_vals = np.arange(T)
    else:
        x_vals = np.asarray(x)[:T]
        if len(x_vals) != T:
            raise ValueError("`x` must have at least T elements.")

    plt.figure(figsize=figsize)
    plt.plot(x_vals, real, **dict(color=None, **real_style))
    for lab, p in pred_list:
        kw = pred_style.copy()
        if "color" in kw and kw["color"] is None:
            kw.pop("color")
        plt.plot(x_vals, p, label=lab, **kw)

    plt.xlabel("Index")
    plt.ylabel("Level")
    plt.title(title)
    plt.grid(True, alpha=grid_alpha)

    handles, leg_labels = plt.gca().get_legend_handles_labels()
    if not any(l == real_style.get("label", "Real") for l in leg_labels):
        handles = [plt.Line2D([], [], **real_style)] + handles
        leg_labels = [real_style.get("label", "Real")] + leg_labels
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_real_pred_grid(
    reals,
    preds,
    titles=None,
    figsize=(12, 6),
    grid_alpha=0.3,
    real_style=dict(linewidth=2, label="Real"),
    pred_style=dict(linewidth=2, linestyle="--"),
    save_path=None
):
    """
    Plot multiple real vs predicted series in a grid of subplots.

    Parameters
    ----------
    reals : np.ndarray, shape (k, n)
        Matrix of ground-truth series; one series per row.
    preds : np.ndarray, shape (k, n)
        Matrix of predicted series; must match reals.shape exactly.
    titles : list[str] or None, optional
        Optional per-subplot titles. If None, defaults to "Series i".
    figsize : tuple, default (12, 6)
        Overall figure size. The grid layout is determined automatically.
    grid_alpha : float, default 0.3
        Grid line transparency for each subplot.
    real_style : dict, optional
        Matplotlib style kwargs for real series lines.
    pred_style : dict, optional
        Matplotlib style kwargs for predicted series lines.
    save_path : str or None, optional
        If provided, saves the figure to this path (default bbox).

    Returns
    -------
    None
        Displays (and optionally saves) the grid of comparisons.

    Notes
    -----
    The grid uses a square-like layout with ncols = ceil(sqrt(k)) and
    nrows = ceil(k / ncols). Any extra axes are turned off.
    """
    reals = np.asarray(reals, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if reals.shape != preds.shape:
        raise ValueError("Shapes of reals and preds must match (k,n).")

    k, n = reals.shape
    ncols = int(np.ceil(np.sqrt(k)))
    nrows = int(np.ceil(k / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i in range(k):
        ax = axes[i // ncols, i % ncols]
        x_vals = np.arange(n)
        ax.plot(x_vals, reals[i], **dict(color=None, **real_style))
        ax.plot(x_vals, preds[i], **pred_style)
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
        else:
            ax.set_title(f"Series {i+1}")
        ax.grid(True, alpha=grid_alpha)

    for j in range(k, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
