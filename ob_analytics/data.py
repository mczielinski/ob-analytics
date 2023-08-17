import pandas as pd
import numpy as np
from ob_analytics.event_processing import load_event_data
from ob_analytics.matching_engine import event_match
from ob_analytics.trades import match_trades
from ob_analytics.order_types import set_order_types
from ob_analytics.depth import price_level_volume
from ob_analytics.depth import depth_metrics
from ob_analytics.event_processing import order_aggressiveness

def process_data(csv_file, price_digits=2, volume_digits=8):
    def get_zombie_ids(events, trades):
        cancelled = events[events['action'] == "deleted"]['id']
        zombies = events[~events['id'].isin(cancelled)]
        bid_zombies = zombies[zombies['direction'] == "bid"]
        ask_zombies = zombies[zombies['direction'] == "ask"]
        
        bid_zombie_ids = bid_zombies[bid_zombies['id'].apply(
            lambda id_val: (trades[(trades['direction'] == "sell") & 
                                  (trades['timestamp'] >= bid_zombies[bid_zombies['id'] == id_val].iloc[-1]['timestamp']) & 
                                  (trades['price'] < bid_zombies[bid_zombies['id'] == id_val].iloc[-1]['price'])]['price'] != 0).any())]['id'].unique()
        
        ask_zombie_ids = ask_zombies[ask_zombies['id'].apply(
            lambda id_val: (trades[(trades['direction'] == "buy") & 
                                  (trades['timestamp'] >= ask_zombies[ask_zombies['id'] == id_val].iloc[-1]['timestamp']) & 
                                  (trades['price'] > ask_zombies[ask_zombies['id'] == id_val].iloc[-1]['price'])]['price'] != 0).any())]['id'].unique()
        
        return list(bid_zombie_ids) + list(ask_zombie_ids)

    events = load_event_data(csv_file)
    events = event_match(events)
    trades = match_trades(events)
    events = set_order_types(events, trades)
    zombie_ids = get_zombie_ids(events, trades)
    events = events[~events['id'].isin(zombie_ids)]

    depth = price_level_volume(events)
    depth_summary = depth_metrics(depth)
    events = order_aggressiveness(events, depth_summary)
    
    offset = min(events['timestamp']) + pd.Timedelta(minutes=60)
    return {
        'events': events,
        'trades': trades,
        'depth': depth,
        'depth_summary': depth_summary[depth_summary['timestamp'] >= offset]
    }

def load_data(bin_file, **kwargs):
    return pd.read_pickle(bin_file)

def save_data(lob_data, bin_file, **kwargs):
    lob_data.to_pickle(bin_file)
