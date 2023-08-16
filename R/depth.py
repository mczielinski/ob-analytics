
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


def depth_metrics(depth, bps=25, bins=20):
    
    def pct_names(name):
        return [f"{name}{i}bps" for i in range(bps, bps*bins+1, bps)]
    
    def interval_sum_breaks(volume_range, breaks):
        sums = []
        prev_idx = 0
        for b in breaks:
            sums.append(volume_range[prev_idx:b].sum())
            prev_idx = b
        return sums
    
    ordered_depth = depth.sort_values('timestamp')
    ordered_depth['price'] = (100 * ordered_depth['price']).astype(int)
    
    col_names = ['best.bid.price', 'best.bid.vol'] + pct_names('bid.vol') + ['best.ask.price', 'best.ask.vol'] + pct_names('ask.vol')
    metrics = pd.DataFrame(0, index=range(len(ordered_depth)), columns=col_names)
    
    asks_state = np.zeros(1000000, dtype=int)
    asks_state[-1] = 1  # trick
    bids_state = np.zeros(1000000, dtype=int)
    bids_state[0] = 1  # trick
    best_ask = max(ordered_depth[ordered_depth['side'] == 'ask']['price'])
    best_bid = min(ordered_depth[ordered_depth['side'] == 'bid']['price'])
    best_ask_vol = 0
    best_bid_vol = 0
    
    for idx, row in ordered_depth.iterrows():
        price, volume, side = row['price'], row['volume'], row['side']
        
        if side == 'ask':
            if price > best_bid:
                asks_state[price] = volume
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
                
                price_range = np.arange(best_ask, int(round((1 + bps * bins * 0.0001) * best_ask)) + 1)
                volume_range = asks_state[price_range]
                breaks = np.ceil(np.cumsum(np.full(bins, len(price_range)/bins))).astype(int)
                
                metrics.at[idx, 'best.ask.price'] = best_ask
                metrics.at[idx, 'best.ask.vol'] = best_ask_vol
                metrics.at[idx, (bins+5):(2*(2+bins))] = interval_sum_breaks(volume_range, breaks)
                if idx > 0:
                    metrics.iloc[idx, 0:(2+bins)] = metrics.iloc[idx-1, 0:(2+bins)]
            
            else:
                if idx > 0:
                    metrics.iloc[idx, :] = metrics.iloc[idx-1, :]
        
        else:
            if price < best_ask:
                bids_state[price] = volume
                if volume > 0:
                    if price > best_bid:
                        best_bid = price
                        best_bid_vol = volume
                    elif price == best_bid:
                        best_bid_vol = volume
                else:
                    if price == best_bid:
                        best_bid = np.where(bids_state > 0)[0][-1]
                        best_bid_vol = bids_state[best_bid]
                
                price_range = np.arange(best_bid, int(round((1 - bps * bins * 0.0001) * best_bid)) + 1)
                volume_range = bids_state[price_range]
                breaks = np.ceil(np.cumsum(np.full(bins, len(price_range)/bins))).astype(int)
                
                metrics.at[idx, 'best.bid.price'] = best_bid
                metrics.at[idx, 'best.bid.vol'] = best_bid_vol
                metrics.at[idx, 3:(2+bins)] = interval_sum_breaks(volume_range, breaks)
                if idx > 0:
                    metrics.iloc[idx, (bins+3):(2*(2+bins))] = metrics.iloc[idx-1, (bins+3):(2*(2+bins))]
            
            else:
                if idx > 0:
                    metrics.iloc[idx, :] = metrics.iloc[idx-1, :]
    
    res = pd.concat([ordered_depth['timestamp'], 0.01 * metrics[['best.bid.price', 'best.ask.price']].round(2), metrics.drop(columns=['best.bid.price', 'best.ask.price'])], axis=1)
    return res

def get_spread(depth_summary):
    spread = depth_summary[['timestamp', 'best.bid.price', 'best.bid.vol', 'best.ask.price', 'best.ask.vol']]
    changes = (spread['best.bid.price'].diff() != 0) | (spread['best.bid.vol'].diff() != 0) | (spread['best.ask.price'].diff() != 0) | (spread['best.ask.vol'].diff() != 0)
    return spread[changes].copy()
