import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ob_analytics.depth import filter_depth


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

    if start_time is None:
        start_time = events["timestamp"].min()
    if end_time is None:
        end_time = events["timestamp"].max()

    # Filter events based on given criteria
    events = events[
        (events["timestamp"] >= start_time)
        & (events["timestamp"] <= end_time)
        & ((events["type"] == "flashed-limit") | (events["type"] == "resting-limit"))
    ]

    events["volume"] *= volume_scale

    # Further filtering based on provided arguments
    if volume_from:
        events = events[events["volume"] >= volume_from]
    if volume_to:
        events = events[events["volume"] <= volume_to]
    if not price_from:
        price_from = events["price"].quantile(0.01)
    if not price_to:
        price_to = events["price"].quantile(0.99)
    events = events[(events["price"] >= price_from) & (events["price"] <= price_to)]

    # Split the events based on action
    created = events[events["action"] == "created"]
    deleted = events[events["action"] == "deleted"]

    # Set up color palette
    col_pal = {"bid": "#0000ff", "ask": "#ff0000"}

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=created,
        x="timestamp",
        y="price",
        hue="direction",
        size="volume",
        palette=col_pal,
        ax=ax,
        legend=None,
    )
    sns.scatterplot(
        data=deleted,
        x="timestamp",
        y="price",
        hue="direction",
        size="volume",
        palette=col_pal,
        ax=ax,
        legend=None,
        marker="o",
        edgecolor=None,
        alpha=0.5,
    )

    # Set labels and other aesthetics
    ax.set_xlabel("Time")
    ax.set_ylabel("Limit Price")
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
    # Extract bid and ask data
    bids = order_book["bids"][::-1].reset_index(drop=True)
    asks = order_book["asks"][::-1].reset_index(drop=True)

    # Construct cumulative liquidity data
    bids["liquidity"] = bids["volume"].cumsum() * volume_scale
    asks["liquidity"] = asks["volume"].cumsum() * volume_scale

    # Combine bid and ask data for plotting
    combined_price = (
        bids["price"].tolist()
        + [bids["price"].iloc[-1], asks["price"].iloc[0]]
        + asks["price"].tolist()
    )
    combined_liquidity = (
        bids["liquidity"].tolist() + [0, 0] + asks["liquidity"].tolist()
    )
    combined_volume = bids["volume"].tolist() + [0, 0] + asks["volume"].tolist()
    side = ["bid"] * (len(bids) + 1) + ["ask"] * (len(asks) + 1)

    df = pd.DataFrame(
        {
            "price": combined_price,
            "liquidity": combined_liquidity,
            "volume": combined_volume,
            "side": side,
        }
    )

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df[df["side"] == "ask"],
        x="price",
        y="liquidity",
        color="red",
        drawstyle="steps-post",
    )
    sns.lineplot(
        data=df[df["side"] == "bid"],
        x="price",
        y="liquidity",
        color="blue",
        drawstyle="steps-pre",
    )

    if show_volume:
        sns.barplot(
            data=df,
            x="price",
            y="volume",
            hue="side",
            dodge=False,
            palette=["blue", "red"],
            alpha=0.5,
        )

    if show_quantiles:
        bid_quantiles = bids[bids["volume"] >= bids["volume"].quantile(0.99)]["price"]
        ask_quantiles = asks[asks["volume"] >= asks["volume"].quantile(0.99)]["price"]
        for val in bid_quantiles:
            plt.axvline(val, color="grey", linestyle="--")
        for val in ask_quantiles:
            plt.axvline(val, color="grey", linestyle="--")

    plt.title(pd.to_datetime(order_book["timestamp"], unit="s"))
    plt.ylabel("Liquidity")
    plt.xlabel("Price")
    sns.set_style("darkgrid")
    plt.show()


def plot_volume_percentiles(
    depth_summary,
    start_time=None,
    end_time=None,
    volume_scale=1,
    perc_line=True,
    side_line=True,
):

    if start_time is None:
        start_time = depth_summary["timestamp"].min()
    if end_time is None:
        end_time = depth_summary["timestamp"].max()

    # Subset the data frame based on start and end times
    depth_summary = depth_summary[
        (depth_summary["timestamp"] >= start_time)
        & (depth_summary["timestamp"] <= end_time)
    ]

    # Define bid and ask column names
    bid_names = [f"bid.vol{bps}bps" for bps in range(25, 525, 25)]
    ask_names = [f"ask.vol{bps}bps" for bps in range(25, 525, 25)]

    # Rescale the volume data
    depth_summary[bid_names + ask_names] *= volume_scale

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot ask side
    ax = depth_summary.plot(
        x="timestamp", y=ask_names, kind="area", stacked=True, colormap="autumn_r"
    )
    if perc_line:
        for col in ask_names:
            depth_summary.plot(
                x="timestamp", y=col, ax=ax, color="black", linewidth=0.1
            )

    # Plot bid side
    depth_summary[bid_names] = -depth_summary[
        bid_names
    ]  # invert bid side for visualization
    depth_summary.plot(
        x="timestamp", y=bid_names, kind="area", stacked=True, ax=ax, colormap="winter"
    )
    if perc_line:
        for col in bid_names:
            depth_summary.plot(
                x="timestamp", y=col, ax=ax, color="black", linewidth=0.1
            )

    # Add side separator line
    if side_line:
        ax.axhline(0, color="black", linewidth=0.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Depth Liquidity")
    plt.title("Order Book Liquidity Through Time")
    plt.show()


def plot_events_histogram(
    events, start_time=None, end_time=None, val="volume", bw=None
):

    if start_time is None:
        start_time = events["timestamp"].min()
    if end_time is None:
        end_time = events["timestamp"].max()

    assert val in [
        "volume",
        "price",
    ], "Invalid value for 'val'. Choose 'volume' or 'price'."

    # Filter the events dataframe based on start and end times
    events = events[
        (events["timestamp"] >= start_time) & (events["timestamp"] <= end_time)
    ]

    # Set up the palette for bid and ask events
    palette = {"bid": "#0000ff", "ask": "#ff0000"}

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=events, x=val, hue="direction", bins=bw, palette=palette, multiple="dodge"
    )

    plt.title(f"Events {val} Distribution")
    plt.show()
