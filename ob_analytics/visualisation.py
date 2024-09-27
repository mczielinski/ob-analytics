import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ob_analytics.auxiliary import reverse_matrix
from ob_analytics.depth import filter_depth

sns.set_theme(
    style="darkgrid", context="notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}
)


def plot_time_series(
    timestamp,
    series,
    start_time=None,
    end_time=None,
    title="time series",
    y_label="series",
):
    """
    Plots a time series.

    Parameters:
    - timestamp: pd.Series
        Series of timestamps.
    - series: pd.Series
        Series of values to plot.
    - start_time: str, optional
        Start time for the plot.
    - end_time: str, optional
        End time for the plot.
    - title: str
        Title of the plot.
    - y_label: str
        Label for the y-axis.

    Returns:
    - None
    """
    if len(timestamp) != len(series):
        raise ValueError("Length of timestamp and series must be the same.")

    # Create a dataframe from the provided series
    df = pd.DataFrame({"ts": timestamp, "val": series})

    if not start_time:
        start_time = df["ts"].min()
    if not end_time:
        end_time = df["ts"].max()

    # Filter the dataframe based on the provided start and end times
    df = df[(df["ts"] >= start_time) & (df["ts"] <= end_time)]

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="ts", y="val", drawstyle="steps-post")

    plt.title(title)
    plt.xlabel("time")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_trades(trades, start_time=None, end_time=None):
    """
    Plots the trades data as a step plot.

    Parameters:
    - trades: pd.DataFrame
        DataFrame containing the trades data with columns 'timestamp' and 'price'.
    - start_time: datetime, optional
        Start time for the plot.
    - end_time: datetime, optional
        End time for the plot.

    Returns:
    - None
    """
    if not start_time:
        start_time = trades["timestamp"].min()
    if not end_time:
        end_time = trades["timestamp"].max()

    filtered_trades = trades[
        (trades["timestamp"] >= start_time) & (trades["timestamp"] <= end_time)
    ]

    # Calculate price breaks for y-axis
    price_range = filtered_trades["price"].max() - filtered_trades["price"].min()
    price_by = 10 ** round(np.log10(price_range) - 1)
    y_breaks = np.arange(
        round(min(filtered_trades["price"]) / price_by) * price_by,
        round(max(filtered_trades["price"]) / price_by) * price_by,
        step=price_by,
    )

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=filtered_trades, x="timestamp", y="price", drawstyle="steps-post")

    plt.xlabel("Time")
    plt.ylabel("Limit Price")
    plt.yticks(y_breaks)
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def plot_price_levels(
    depth,
    spread=None,
    trades=None,
    show_mp=True,
    show_all_depth=False,
    col_bias=0.1,
    start_time=None,
    end_time=None,
    price_from=None,
    price_to=None,
    volume_from=None,
    volume_to=None,
    volume_scale=1,
    price_by=None,
):

    if start_time is None:
        start_time = depth["timestamp"].min()
    if end_time is None:
        end_time = depth["timestamp"].max()

    # Scale the volume
    depth["volume"] *= volume_scale

    # Filter spread based on start and end time and set price_from, price_to
    if spread is not None:
        spread = spread[
            (spread["timestamp"] >= start_time) & (spread["timestamp"] <= end_time)
        ]
        if price_from is None:
            price_from = 0.995 * spread["best.bid.price"].min()
        if price_to is None:
            price_to = 1.005 * spread["best.ask.price"].max()

    # Filter trades based on start and end time and set price_from, price_to
    if trades is not None:
        trades = trades[
            (trades["timestamp"] >= start_time) & (trades["timestamp"] <= end_time)
        ]
        if price_from is None:
            price_from = 0.995 * trades["price"].min()
        else:
            trades = trades[trades["price"] >= price_from]

        if price_to is None:
            price_to = 1.005 * trades["price"].max()
        else:
            trades = trades[trades["price"] <= price_to]

    # Filter depth by price and volume
    if price_from is not None:
        depth = depth[depth["price"] >= price_from]
    if price_to is not None:
        depth = depth[depth["price"] <= price_to]
    if volume_from is not None:
        depth = depth[(depth["volume"] >= volume_from) | (depth["volume"] == 0)]
    if volume_to is not None:
        depth = depth[depth["volume"] <= volume_to]

    # Filter depth by the time window (using a placeholder function for filterDepth)
    depth_filtered = filter_depth(depth, start_time, end_time)

    # Remove price levels with no update during time window if requested
    if not show_all_depth:
        # Group by price and check for unchanged timestamps
        groups = depth_filtered.groupby("price")
        unchanged_prices = []
        for name, group in groups:
            timestamps = group["timestamp"]
            if (
                len(timestamps) == 2
                and timestamps.iloc[0] == start_time
                and timestamps.iloc[1] == end_time
            ):
                unchanged_prices.append(name)
        depth_filtered = depth_filtered[~depth_filtered["price"].isin(unchanged_prices)]

    depth_filtered.loc[depth_filtered["volume"] == 0, "volume"] = np.nan

    # Call the plotPriceLevelsFaster function (assuming it's defined elsewhere)
    plot_price_levels_faster(
        depth_filtered, spread, trades, show_mp, col_bias, price_by
    )


def plot_price_levels_faster(
    depth, spread=None, trades=None, show_mp=True, col_bias=0.1, price_by=None
):

    # Filter trades based on their direction
    if trades is not None:
        buys = trades[trades["direction"] == "buy"]
        sells = trades[trades["direction"] == "sell"]
    else:
        buys = None
        sells = None

    # Color palette generation
    if col_bias <= 0:
        col_bias = 1
        log_10 = True
    else:
        log_10 = False

    col_pal = sns.color_palette("RdBu_r", n_colors=len(depth["volume"].unique()))

    if price_by is None:
        price_by = 10 ** round(
            np.log10(depth["price"].max() - depth["price"].min()) - 1
        )

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Heatmap-style plot using seaborn's lineplot
    if log_10:
        sns.lineplot(
            data=depth, x="timestamp", y="price", hue="volume", palette=col_pal, ax=ax
        )
    else:
        sns.lineplot(
            data=depth, x="timestamp", y="price", hue="volume", palette=col_pal, ax=ax
        )

    # Plot midprice or spread
    if spread is not None:
        if show_mp:
            midprice = (spread["best.bid.price"] + spread["best.ask.price"]) / 2
            ax.plot(spread["timestamp"], midprice, color="#ffffff", linewidth=1.1)
        else:
            ax.plot(
                spread["timestamp"],
                spread["best.ask.price"],
                color="#ff0000",
                linewidth=1.5,
            )
            ax.plot(
                spread["timestamp"],
                spread["best.bid.price"],
                color="#00ff00",
                linewidth=1.5,
            )

    # Plot trades
    if trades is not None:
        ax.scatter(
            sells["timestamp"],
            sells["price"],
            color="#ff0000",
            s=50,
            marker="o",
            edgecolors="white",
        )
        ax.scatter(
            buys["timestamp"],
            buys["price"],
            color="#00ff00",
            s=50,
            marker="o",
            edgecolors="white",
        )

    # Set labels
    ax.set_xlabel("Time")
    ax.set_ylabel("Limit Price")

    # Theme configuration
    sns.set_style("darkgrid")
    plt.legend(loc="upper left")

    plt.savefig("foo.png")


def plot_event_map(
    events,
    start_time=None,
    end_time=None,
    price_from=None,
    price_to=None,
    volume_from=None,
    volume_to=None,
    volume_scale=1,
):

    # Set defaults if not provided
    if start_time is None:
        start_time = events["timestamp"].min()
    if end_time is None:
        end_time = events["timestamp"].max()

    # Filter for the given time window and event types
    events = events[
        (events["timestamp"] >= start_time)
        & (events["timestamp"] <= end_time)
        & ((events["type"] == "flashed-limit") | (events["type"] == "resting-limit"))
    ].copy()  # Create a copy to avoid SettingWithCopyWarning

    # Scale the volume
    events["volume"] *= volume_scale

    # Further filtering based on volume if specified
    if volume_from is not None:
        events = events[events["volume"] >= volume_from]
    if volume_to is not None:
        events = events[events["volume"] <= volume_to]

    # If price range is not specified, set it to contain 99% of data
    if price_from is None:
        price_from = events["price"].quantile(0.01)
    if price_to is None:
        price_to = events["price"].quantile(0.99)

    # Filter based on price range
    events = events[(events["price"] >= price_from) & (events["price"] <= price_to)]

    # Separate created and deleted events
    created = events[events["action"] == "created"]
    deleted = events[events["action"] == "deleted"]

    # Set up color palette for direction
    col_pal = {"bid": "#0000ff", "ask": "#ff0000"}

    # Define the price range for breaks
    price_by = 10 ** round(np.log10(events["price"].max() - events["price"].min()) - 1)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot created events with a filled marker
    sns.scatterplot(
        data=created,
        x="timestamp",
        y="price",
        size="volume",
        sizes=(20, 200),  # Adjust bubble size range
        color="#333333",  # Color for the created events
        ax=ax,
        legend=False,
        marker="o",
    )

    # Plot deleted events with an open marker
    sns.scatterplot(
        data=deleted,
        x="timestamp",
        y="price",
        size="volume",
        sizes=(20, 200),  # Adjust bubble size range
        color="#333333",  # Color for the deleted events
        ax=ax,
        legend=False,
        marker="o",
        edgecolor="black",
        alpha=0.5,
    )

    # Overlay the events with color based on the direction (ask/bid)
    sns.scatterplot(
        data=events,
        x="timestamp",
        y="price",
        hue="direction",
        size=0.1,  # Small size for direction overlay
        palette=col_pal,
        ax=ax,
        legend=False,
    )

    # Set axis labels and customize plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Limit Price")
    ax.set_yticks(
        np.arange(
            round(events["price"].min() / price_by) * price_by,
            round(events["price"].max() / price_by) * price_by,
            price_by,
        )
    )

    sns.set_style("darkgrid")
    plt.show()


def plot_volume_map(
    events,
    action="deleted",
    event_type=["flashed-limit"],
    start_time=None,
    end_time=None,
    price_from=None,
    price_to=None,
    volume_from=None,
    volume_to=None,
    volume_scale=1,
    log_scale=False,
):

    assert action in ["deleted", "created"], "Invalid action provided"

    if start_time is None:
        start_time = events["timestamp"].min()
    if end_time is None:
        end_time = events["timestamp"].max()

    # Scale the volume
    events["volume"] *= volume_scale

    # Filter the events based on the provided criteria
    mask = (
        (events["action"] == action)
        & events["type"].isin(event_type)
        & (events["timestamp"] >= start_time)
        & (events["timestamp"] <= end_time)
    )
    events = events[mask]

    # Further filtering based on volume and price
    if price_from:
        events = events[events["price"] >= price_from]
    if price_to:
        events = events[events["price"] <= price_to]
    if volume_from is None:
        volume_from = events["volume"].quantile(0.0001)
    events = events[events["volume"] >= volume_from]
    if volume_to is None:
        volume_to = events["volume"].quantile(0.9999)
    events = events[events["volume"] <= volume_to]

    # Set up the color palette
    col_pal = {"bid": "#0000ff", "ask": "#ff0000"}

    # Plotting
    plt.figure(figsize=(10, 6))
    if log_scale:
        plt.yscale("log")
    sns.scatterplot(
        data=events,
        x="timestamp",
        y="volume",
        hue="direction",
        palette=col_pal,
        size=0.5,
        marker="o",
    )
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.title("Volume Map of Flashed Limit Orders")
    sns.set_style("darkgrid")
    plt.show()


def plot_current_depth(
    order_book, volume_scale=1, show_quantiles=True, show_volume=True
):
    # Ensure bids are sorted descending and asks ascending
    # bids = order_book['bids'].sort_values(by='price', ascending=False).reset_index(drop=True)
    # asks = order_book['asks'].sort_values(by='price', ascending=True).reset_index(drop=True)
    bids = reverse_matrix(order_book["bids"])
    asks = reverse_matrix(order_book["asks"])

    # Combine both sides into a single series
    x = np.concatenate(
        [
            bids["price"].values,
            [bids["price"].values[-1]],
            [asks["price"].values[0]],
            asks["price"].values,
        ]
    )
    y1 = (
        np.concatenate([bids["liquidity"].values, [0], [0], asks["liquidity"].values])
        * volume_scale
    )
    y2 = (
        np.concatenate([bids["volume"].values, [0], [0], asks["volume"].values])
        * volume_scale
    )
    side = ["bid"] * (len(bids) + 1) + ["ask"] * (len(asks) + 1)

    # Create depth DataFrame
    depth = pd.DataFrame({"price": x, "liquidity": y1, "volume": y2, "side": side})

    # Set up the plot
    # plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot volume using ax.bar with auto-scaling bar width
    if show_volume:
        # Compute bar width based on the resolution of the price data
        unique_prices = np.sort(np.unique(depth["price"]))
        price_diffs = np.diff(unique_prices)
        # Filter out zero or negative differences
        price_diffs = price_diffs[price_diffs > 0]
        if len(price_diffs) > 0:
            resolution = np.min(price_diffs)
            bar_width = resolution * 5  # 500% of the minimum non-zero price difference
        else:
            # If all prices are the same, set a default bar width
            bar_width = 1  # Adjust as necessary

        ax.bar(
            depth["price"],
            depth["volume"],
            width=bar_width,
            color="#555555",
            alpha=0.3,
            align="center",
            edgecolor=None,
        )

    # Plot liquidity (cumulative volume)
    col_pal = {"ask": "#ff0000", "bid": "#0000ff"}
    for side_value in ["bid", "ask"]:
        side_data = depth[depth["side"] == side_value]
        ax.step(
            side_data["price"],
            side_data["liquidity"],
            where="pre",  # Changed 'post' to 'pre'
            color=col_pal[side_value],
            label=side_value,
            linewidth=2,
        )

    # Highlight highest 1% volume with vertical lines
    if show_quantiles:
        bid_quantile = bids["volume"].quantile(0.99)
        bid_quantiles = bids.loc[bids["volume"] >= bid_quantile, "price"]
        ask_quantile = asks["volume"].quantile(0.99)
        ask_quantiles = asks.loc[asks["volume"] >= ask_quantile, "price"]
        for x_value in bid_quantiles:
            ax.axvline(x=x_value, color="#222222", linestyle="--")
        for x_value in ask_quantiles:
            ax.axvline(x=x_value, color="#222222", linestyle="--")

    # Set x-axis ticks
    xmin = round(bids["price"].min())
    xmax = round(asks["price"].max())
    xticks = np.arange(xmin, xmax + 1, 1)
    ax.set_xticks(xticks)

    # Set labels and title
    timestamp = pd.to_datetime(order_book["timestamp"], unit="s", utc=True)
    ax.set_title(timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"))
    ax.set_xlabel("Price")
    ax.set_ylabel("Liquidity")

    # Customize plot appearance
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_volume_percentiles(
    depth_summary,
    start_time=None,
    end_time=None,
    volume_scale=1,
    perc_line=True,
    side_line=True,
):
    from datetime import timedelta

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    if start_time is None:
        start_time = depth_summary["timestamp"].iloc[0]
    if end_time is None:
        end_time = depth_summary["timestamp"].iloc[-1]

    # Generate bid and ask column names
    bid_names = [f"bid.vol{i}bps" for i in range(25, 501, 25)]
    ask_names = [f"ask.vol{i}bps" for i in range(25, 501, 25)]

    td = (end_time - start_time).total_seconds()
    td = round(td)

    # Determine frequency based on time difference
    frequency = "mins" if td > 900 else "secs"

    # Subset the data based on start and end times
    delta = timedelta(seconds=60 if frequency == "mins" else 1)
    mask = (depth_summary["timestamp"] >= (start_time - delta)) & (
        depth_summary["timestamp"] <= end_time
    )
    ob_percentiles = depth_summary.loc[mask, ["timestamp"] + bid_names + ask_names]

    # Remove duplicates, keeping the last occurrence
    ob_percentiles = ob_percentiles.drop_duplicates(subset="timestamp", keep="last")

    # Set timestamp as index
    ob_percentiles.set_index("timestamp", inplace=True)

    # Truncate timestamps to frequency
    if frequency == "mins":
        intervals = ob_percentiles.index.floor("T")
    else:
        intervals = ob_percentiles.index.floor("S")

    # Aggregate data by intervals
    aggregated = ob_percentiles.groupby(intervals).mean()

    # Adjust timestamps and reset index
    aggregated.index = aggregated.index + delta
    aggregated.reset_index(inplace=True)
    aggregated.rename(columns={"index": "timestamp"}, inplace=True)
    ob_percentiles = aggregated

    # Update bid and ask names with zero padding
    bid_names = [f"bid.vol{int(i):03d}bps" for i in range(25, 501, 25)]
    ask_names = [f"ask.vol{int(i):03d}bps" for i in range(25, 501, 25)]
    ob_percentiles.columns = ["timestamp"] + bid_names + ask_names

    # Calculate max ask and bid volumes
    max_ask = ob_percentiles[ask_names].sum(axis=1).max()
    max_bid = ob_percentiles[bid_names].sum(axis=1).max()

    # Melt the data frames for asks and bids
    melted_asks = ob_percentiles.melt(
        id_vars="timestamp",
        value_vars=ask_names,
        var_name="percentile",
        value_name="liquidity",
    )
    melted_asks["percentile"] = pd.Categorical(
        melted_asks["percentile"], categories=ask_names[::-1], ordered=True
    )
    melted_asks["liquidity"] *= volume_scale

    melted_bids = ob_percentiles.melt(
        id_vars="timestamp",
        value_vars=bid_names,
        var_name="percentile",
        value_name="liquidity",
    )
    melted_bids["percentile"] = pd.Categorical(
        melted_bids["percentile"], categories=bid_names[::-1], ordered=True
    )
    melted_bids["liquidity"] *= volume_scale

    # Define color palette
    colors_list = [
        "#f92b20",
        "#fe701b",
        "#facd1f",
        "#d6fd1c",
        "#65fe1b",
        "#1bfe42",
        "#1cfdb4",
        "#1fb9fa",
        "#1e71fb",
        "#261cfd",
    ]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors_list, N=20)
    col_pal = [cmap(i / 19) for i in range(20)]
    col_pal *= 2  # Duplicate for bids and asks

    # Define breaks and legend names
    breaks = [f"ask.vol{int(i):03d}bps" for i in range(500, 49, -50)] + [
        f"bid.vol{int(i):03d}bps" for i in range(50, 501, 50)
    ]
    legend_names = [f"+{int(i):03d}bps" for i in range(500, 49, -50)] + [
        f"-{int(i):03d}bps" for i in range(50, 501, 50)
    ]

    pl = 0.1 if perc_line else 0  # Line size

    # Pivot data to wide format
    asks_pivot = melted_asks.pivot(
        index="timestamp", columns="percentile", values="liquidity"
    )
    bids_pivot = melted_bids.pivot(
        index="timestamp", columns="percentile", values="liquidity"
    )

    # Sort columns according to the ordered categories
    asks_pivot = asks_pivot[ask_names[::-1]]  # Reverse to match R code stacking order
    bids_pivot = bids_pivot[bid_names[::-1]]

    # Compute cumulative sums for stacking
    asks_cumsum = asks_pivot.cumsum(axis=1)
    bids_cumsum = bids_pivot.cumsum(axis=1)

    # Multiply bids by -1 for plotting negative values
    bids_cumsum_neg = -bids_cumsum

    # Prepare colors
    asks_cols = asks_cumsum.columns.tolist()
    bids_cols = bids_cumsum.columns.tolist()
    all_cols = asks_cols + bids_cols
    colors_dict = dict(zip(all_cols, col_pal))

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot asks
    prev = np.zeros(len(asks_cumsum))
    x = asks_cumsum.index
    for percentile in asks_cols:
        current = asks_cumsum[percentile].values
        ax.fill_between(
            x,
            prev,
            current,
            facecolor=colors_dict[percentile],
            edgecolor="black" if perc_line else None,
            linewidth=pl,
        )
        prev = current

    # Plot bids
    prev = np.zeros(len(bids_cumsum_neg))
    x = bids_cumsum_neg.index
    for percentile in bids_cols:
        current = bids_cumsum_neg[percentile].values
        ax.fill_between(
            x,
            prev,
            current,
            facecolor=colors_dict[percentile],
            edgecolor="black" if perc_line else None,
            linewidth=pl,
        )
        prev = current

    # Add horizontal line at y=0
    if side_line:
        ax.axhline(y=0, color="#000000", linewidth=0.1)

    # Set y limits
    y_range = volume_scale * max(max_ask, max_bid)
    ax.set_ylim(-y_range, y_range)

    # Set x label
    ax.set_xlabel("time")

    # Format x-axis dates
    fig.autofmt_xdate()

    # Create legend
    legend_elements = []
    for col, label in zip(all_cols, legend_names):
        patch = Patch(
            facecolor=colors_dict[col],
            edgecolor="black" if perc_line else None,
            label=label,
        )
        legend_elements.append(patch)

    ax.legend(handles=legend_elements, title="depth         \n", loc="best", ncol=2)

    plt.tight_layout()
    plt.show()


def plot_events_histogram(
    events, start_time=None, end_time=None, val="volume", bw=None
):
    """
    Plot a histogram given event data.

    Convenience function for plotting event price and volume histograms.
    Will plot ask/bid bars side by side.

    Parameters:
    - events: pandas DataFrame containing event data.
    - start_time: Include event data >= this time.
    - end_time: Include event data <= this time.
    - val: 'volume' or 'price'.
    - bw: Bin width (e.g., for price, 0.5 = 50 cent buckets).

    Returns:
    - None
    """
    assert val in ["volume", "price"], "val must be 'volume' or 'price'"

    # Set default start_time and end_time if not provided
    if start_time is None:
        start_time = events["timestamp"].min()
    if end_time is None:
        end_time = events["timestamp"].max()

    # Filter events between start_time and end_time
    events_filtered = events[
        (events["timestamp"] >= start_time) & (events["timestamp"] <= end_time)
    ]

    # Set up the plot
    plt.figure(figsize=(12, 7))

    # Plot the histogram
    sns.histplot(
        data=events_filtered,
        x=val,
        hue="direction",
        multiple="dodge",  # Side by side bars
        binwidth=bw,
        palette={"bid": "#0000ff", "ask": "#ff0000"},
        edgecolor="white",
        linewidth=0.5,
    )

    # Set labels and title
    plt.title(f"Events {val} distribution")
    plt.xlabel(val.capitalize())
    plt.ylabel("Count")

    # Show the plot
    plt.tight_layout()
    plt.show()
