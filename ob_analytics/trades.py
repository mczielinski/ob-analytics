
import pandas as pd
import numpy as np

def match_trades(events: pd.DataFrame) -> pd.DataFrame:
    """
    Match trades with their respective makers and takers, and handle price jumps.
    
    Parameters
    ----------
    events : pd.DataFrame
        Events dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with matched trades.
    """
    events=events.reset_index(drop=True)
    
    matching_bids = events[(events['direction'] == 'bid') & (~events['matching.event'].isna())]
    matching_bids = matching_bids.sort_values(by='event.id')
    
    matching_asks = events[(events['direction'] == 'ask') & (~events['matching.event'].isna())]
    matching_asks = matching_asks.sort_values(by='matching.event')
    
    assert set(matching_bids['event.id']) == set(matching_asks['matching.event']), "Mismatch in bid and ask events."
    
    bid_exchange_ts = matching_bids['exchange.timestamp']
    ask_exchange_ts = matching_asks['exchange.timestamp']

    bid_maker = (bid_exchange_ts.values < ask_exchange_ts.values) | ((bid_exchange_ts.values == ask_exchange_ts.values) & 
                                                     (matching_bids['id'].values < matching_asks['id'].values))
    
    bid_local_ts = matching_bids['timestamp']
    ask_local_ts = matching_asks['timestamp']
    timestamp = np.where(bid_local_ts.values <= ask_local_ts.values, bid_local_ts.values, ask_local_ts.values)
    
    price = np.where(bid_maker, matching_bids['price'], matching_asks['price'])
    volume = matching_bids['fill']
    
    direction = np.where(bid_maker, 'sell', 'buy')
    
    maker_event_id = np.where(bid_maker, matching_bids['event.id'], matching_asks['event.id'])
    taker_event_id = np.where(bid_maker, matching_asks['event.id'], matching_bids['event.id'])
    
    maker = events.loc[events['event.id'].isin(maker_event_id), 'id'].values
    taker = events.loc[events['event.id'].isin(taker_event_id), 'id'].values

    print(timestamp.shape)
    print(price.shape)
    print(volume.shape)
    print(direction.shape)
    print(maker_event_id.shape)
    print(taker_event_id.shape)
    print(maker.shape)
    print(taker.shape)


    combined = pd.DataFrame({
        'timestamp': timestamp,
        'price': price,
        'volume': volume,
        'direction': direction,
        'maker_event_id': maker_event_id,
        'taker_event_id': taker_event_id,
        'maker': maker,
        'taker': taker
    })
    
    combined = combined.sort_values(by='timestamp')
    
    jumps = np.where(np.abs(np.diff(combined['price'])) > 10)[0]
    
    # Swapping makers and takers based on price jumps
    for jump in jumps:
        combined.iloc[jump, combined.columns.get_loc('maker')], combined.iloc[jump, combined.columns.get_loc('taker')] =         combined.iloc[jump, combined.columns.get_loc('taker')], combined.iloc[jump, combined.columns.get_loc('maker')]
        
        combined.iloc[jump, combined.columns.get_loc('maker_event_id')], combined.iloc[jump, combined.columns.get_loc('taker_event_id')] =         combined.iloc[jump, combined.columns.get_loc('taker_event_id')], combined.iloc[jump, combined.columns.get_loc('maker_event_id')]
        
        combined.iloc[jump, combined.columns.get_loc('direction')] = 'buy' if combined.iloc[jump, combined.columns.get_loc('direction')] == 'sell' else 'sell'
        
    return combined

def trade_impacts(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trade impacts.

    Parameters
    ----------
    trades : pd.DataFrame
        Trades dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with trade impacts.
    """

    trades["value"] = trades["price"] * trades["volume"]
    vwap = trades.groupby("taker")["value"].sum() / trades.groupby("taker")["volume"].sum()
    hits = trades.groupby("taker").size()
    vol = trades.groupby("taker")["volume"].sum()
    price_impact = trades.groupby("taker")["price"].last() - vwap

    impacts = pd.DataFrame({
        "vwap": vwap,
        "hits": hits,
        "vol": vol,
        "price_impact": price_impact
    })

    return impacts
