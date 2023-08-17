
import pandas as pd
import numpy as np

def price_level_volume(events):
    
    def directional_price_level_volume(dir_events):
        dir_events = dir_events.copy()
        condition = (
    (dir_events['action'] == 'created')
    | (
        (dir_events['action'] == 'changed')
        & (dir_events['fill'] == 0)
    )
) & (dir_events['type'] != 'pacman') & (dir_events['type'] != 'market')
        added_volume = dir_events[condition][['event.id', 'id', 'timestamp', 'exchange.timestamp', 'price', 'volume', 'direction', 'action']]
        
        condition = (
    (dir_events['action'] == 'deleted')
    & (dir_events['volume'] > 0)
    & (dir_events['type'] != 'pacman')
    & (dir_events['type'] != 'market')
)
        cancelled_volume = dir_events[condition][['event.id', 'id', 'timestamp', 'exchange.timestamp', 'price', 'volume', 'direction', 'action']]
        cancelled_volume['volume'] = -cancelled_volume['volume']
        cancelled_volume = cancelled_volume[cancelled_volume['id'].isin(added_volume['id'])]

        condition = (
    (dir_events['fill'] > 0)
    & (dir_events['type'] != 'pacman')
    & (dir_events['type'] != 'market')
)
        filled_volume = dir_events[condition][['event.id', 'id', 'timestamp', 'exchange.timestamp', 'price', 'fill', 'direction', 'action']]
        filled_volume['fill'] = -filled_volume['fill']
        filled_volume = filled_volume[filled_volume['id'].isin(added_volume['id'])]
        filled_volume.columns = ['event.id', 'id', 'timestamp', 'exchange.timestamp', 'price', 'volume', 'direction', 'action']

        volume_deltas = pd.concat([added_volume, cancelled_volume, filled_volume])
        volume_deltas = volume_deltas.sort_values(['price', 'timestamp'])
        
        cum_volume = volume_deltas.groupby('price')['volume'].cumsum()
        cum_volume = np.where(cum_volume < 0, 0, cum_volume)
        
        return pd.DataFrame({
            'timestamp': volume_deltas['timestamp'],
            'price': volume_deltas['price'],
            'volume': cum_volume,
            'side': volume_deltas['direction']
        })

    bids = events[events['direction'] == 'bid']
    depth_bid = directional_price_level_volume(bids)
    asks = events[events['direction'] == 'ask']
    depth_ask = directional_price_level_volume(asks)
    depth_data = pd.concat([depth_bid, depth_ask])
    
    return depth_data.sort_values('timestamp')

def interval_sum_breaks(volume_range, breaks):
    sums = []
    prev_idx = 0
    for b in breaks:
        sums.append(volume_range[prev_idx:b].sum())
        prev_idx = b
    return sums

def filter_depth(d, from_time, to_time):
    # 1. Get all active price levels before the start of the range.
    pre = d[d['timestamp'] <= from_time]
    pre = pre.sort_values(by=['price', 'timestamp'])

    # Last update for each price level <= from_time. This becomes the starting point
    # for all updates within the range.
    pre = pre.loc[~pre['price'].duplicated(keep='last') & (pre['volume'] > 0)]

    # Clamp range (reset timestamp to from_time if price level active before start of range).
    if not pre.empty:
        pre['timestamp'] = pre['timestamp'].apply(lambda r: max(from_time, r))

    # 2. Add all volume changes within the range.
    mid = d[(d['timestamp'] > from_time) & (d['timestamp'] < to_time)]
    combined_range = pd.concat([pre, mid])

    # 3. At the end of the range, set all price level volume to 0.
    open_ends = combined_range.loc[~combined_range['price'].duplicated(keep='last') & (combined_range['volume'] > 0)].copy()
    open_ends['timestamp'] = to_time
    open_ends['volume'] = 0

    # Combine pre, mid, and open_ends. Ensure it's in order.
    combined_range = pd.concat([combined_range, open_ends])
    combined_range = combined_range.sort_values(by=['price', 'timestamp'])

    return combined_range
    
def depth_metrics(depth, bps=25, bins=20):
    def pct_names(name):
        return [f"{name}{i}bps" for i in range(bps, bps * bins + 1, bps)]
    
    ordered_depth = depth.sort_values(by="timestamp")
    ordered_depth["price"] = (100 * ordered_depth["price"]).values.round().astype(int)
    depth_matrix = np.column_stack([ordered_depth["price"], ordered_depth["volume"], 
                                   np.where(ordered_depth["side"] == "bid", 0, 1)])

    metrics_columns = ["best.bid.price", "best.bid.vol"] + pct_names("bid.vol") + ["best.ask.price", "best.ask.vol"] + pct_names("ask.vol")
    metrics = pd.DataFrame(0, columns=metrics_columns, index=ordered_depth.index)

    asks_state = np.zeros(1000000, dtype=int)
    asks_state[-1] = 1
    bids_state = np.zeros(1000000, dtype=int)
    bids_state[0] = 1

    best_ask = ordered_depth.loc[ordered_depth["side"] == "ask", "price"].max()
    best_bid = ordered_depth.loc[ordered_depth["side"] == "bid", "price"].min()
    best_ask_vol = 0
    best_bid_vol = 0

    for i in range(len(ordered_depth)):
        depth_row = depth_matrix[i]
        price, volume, side = depth_row

        if side > 0:
            if price > best_bid:
                asks_state[int(price)] = volume
                if volume > 0:
                    if price < best_ask:
                        best_ask = price
                        best_ask_vol = volume
                    elif price == best_ask:
                        best_ask_vol = volume
                else:
                    if price == best_ask:
                        best_ask = np.where(asks_state > 0)[0][0]
                        best_ask_vol = asks_state[best_ask]

                price_range = np.arange(int(best_ask), int((1 + bps * bins * 0.0001) * best_ask) + 1)
                volume_range = asks_state[price_range]

                breaks = np.ceil(np.cumsum([len(price_range) / bins] * bins)).astype(int)

                metrics.at[ordered_depth.index[i], "best.ask.price"] = best_ask
                metrics.at[ordered_depth.index[i], "best.ask.vol"] = best_ask_vol
                metrics.loc[ordered_depth.index[i], pct_names("ask.vol")] = interval_sum_breaks(volume_range, breaks)

                if i > 0:
                    metrics.iloc[i, :bins + 2] = metrics.iloc[i - 1, :bins + 2]
            else:
                if i > 0:
                    metrics.iloc[i] = metrics.iloc[i - 1]

        else:
            if price < best_ask:
                bids_state[int(price)] = volume
                if volume > 0:
                    if price > best_bid:
                        best_bid = price
                    elif price == best_bid:
                        best_bid_vol = volume
                else:
                    if price == best_bid:
                        best_bid = np.where(bids_state > 0)[0][-1]
                        best_bid_vol = bids_state[best_bid]

                price_range = np.arange(int(best_bid), int((1 - bps * bins * 0.0001) * best_bid) - 1, -1)

                volume_range = bids_state[price_range]

                breaks = np.ceil(np.cumsum([len(price_range) / bins] * bins)).astype(int)

                metrics.at[ordered_depth.index[i], "best.bid.price"] = best_bid
                metrics.at[ordered_depth.index[i], "best.bid.vol"] = best_bid_vol
                metrics.loc[ordered_depth.index[i], pct_names("bid.vol")] = interval_sum_breaks(volume_range, breaks)

                if i > 0:
                    metrics.iloc[i, bins + 2:] = metrics.iloc[i - 1, bins + 2:]
            else:
                if i > 0:
                    metrics.iloc[i] = metrics.iloc[i - 1]

    res = pd.concat([ordered_depth["timestamp"], metrics], axis=1)
    res[["best.bid.price", "best.ask.price"]] = (0.01 * res[["best.bid.price", "best.ask.price"]]).round(2)
    
    return res

def get_spread(depth_summary):
    spread = depth_summary[['timestamp', 'best.bid.price', 'best.bid.vol', 'best.ask.price', 'best.ask.vol']]
    changes = (spread['best.bid.price'].diff() != 0) | (spread['best.bid.vol'].diff() != 0) | (spread['best.ask.price'].diff() != 0) | (spread['best.ask.vol'].diff() != 0)
    return spread[changes].copy()
