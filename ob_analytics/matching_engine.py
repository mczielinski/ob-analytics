import pandas as pd
import numpy as np
from ob_analytics.needleman_wunsch import create_similarity_matrix
from ob_analytics.needleman_wunsch import align_sequences

def event_match(events: pd.DataFrame, cut_off_ms: int = 5000) -> pd.DataFrame:
  """
  Match market orders (takers) to limit orders (makers).

  Args:
    events: A pandas DataFrame of order book events.
    cut_off_ms: The time window in milliseconds for considering candidate matches.

  Returns:
    The events DataFrame with a 'matching.event' column indicating matched events. 
  """  
  def matcher() -> np.ndarray:

    cut_off_td = np.timedelta64(cut_off_ms*1000000, 'ns')
    res = []
    cols = ['original_number','event.id', 'fill', 'timestamp']

    bid_fills = events[(events['direction'] == 'bid') & (events['fill'] != 0)][cols]
    ask_fills = events[(events['direction'] == 'ask') & (events['fill'] != 0)][cols]

    def fill_id(src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
      id_fills = src[src['fill'].isin(dst['fill'])]
      return id_fills.sort_values(by=['fill', 'timestamp'], ascending=[False, True])

    id_bid_fills = fill_id(bid_fills, ask_fills)
    id_ask_fills = fill_id(ask_fills, bid_fills)

    for volume in id_bid_fills['fill'].unique():
        #print('--------------------------------------------------------------------')
        #print('volume:', volume)
        #print('')
        bids = id_bid_fills[id_bid_fills['fill'] == volume]
        asks = id_ask_fills[id_ask_fills['fill'] == volume]
        #print('bids ts:', bids['timestamp'].values)
        #print('asks ts:', asks['timestamp'].values)
        #print('')
        #print('bids event id: ', bids['event.id'].values)
        #print('asks event id: ', asks['event.id'].values)
        #print('')
        #print('bids orig: ', bids['original_number'].values)
        #print('asks orig: ', asks['original_number'].values)
        #print('')
        #print('bids fill: ', bids['fill'].values)
        #print('asks fill: ', asks['fill'].values)
        #print('')
        #print('bids reshape:', bids['timestamp'].values.reshape(-1, 1))
        #print('subtraction: ', (bids['timestamp'].values.reshape(-1, 1) - asks['timestamp'].values))
        #print('subtraction dtype: ', (bids['timestamp'].values.reshape(-1, 1) - asks['timestamp'].values).dtype)
        
        # Calculate distance matrix in milliseconds
        distance_matrix_ms = (bids['timestamp'].values.reshape(-1, 1) - asks['timestamp'].values).astype('timedelta64[ns]').astype('int64')
        #print('distance matrix: ', distance_matrix_ms)
        
        # Handle single ask case
        if len(asks) == 1:
            distance_matrix_ms = distance_matrix_ms.reshape(-1, 1)
            #print('reshaped (handled single ask case)')
        
        # Find the closest ask indices for each bid
        closest_ask_indices = np.argmin(np.abs(distance_matrix_ms), axis=1)
        #print(np.abs(distance_matrix_ms))
        #print('closest_ask_indices: ', closest_ask_indices)
    
        # Retrieve event ids of the closest asks
        ask_event_ids = asks['event.id'].values[closest_ask_indices]
        #print('closest_ask_indices orig no: ', ask_event_ids)
    
        # Create a mask for the valid matches within cutoff
        mask = np.abs(distance_matrix_ms[np.arange(len(bids)), closest_ask_indices]) <= cut_off_td
        #print('mask: ', np.abs(distance_matrix_ms[np.arange(len(bids)), closest_ask_indices]))
        #print('mask dtype: ', np.abs(distance_matrix_ms[np.arange(len(bids)), closest_ask_indices]).dtype)
        #print('cutoff: ', cut_off_td)
        #print('cutoff dtype: ', cut_off_td.dtype)
        #print('mask bool:', mask)
        
        # Apply mask to get the final ask event ids
        ask_event_ids = np.where(mask, ask_event_ids, np.nan)
        #print('ask event ids:', ask_event_ids)
    
        if not any(pd.isna(ask_event_ids)) and len(ask_event_ids) == len(set(ask_event_ids)):
            #print('first logic: ', any(pd.isna(ask_event_ids)))
            #print('second logic: ', len(ask_event_ids) == len(set(ask_event_ids)))
            matches = np.column_stack((bids['event.id'], ask_event_ids))
            res.extend(matches)
            #print('matches', matches)
        else:
            #print('sim matrix time!')
            similarity_matrix = create_similarity_matrix(bids['timestamp'], asks['timestamp'], cut_off_td)
            #print('generated matrix: ', similarity_matrix)
            aligned_indices = align_sequences(similarity_matrix)
            #print('aligned_indices: ', aligned_indices)
            matched_indices = aligned_indices[np.abs((bids['timestamp'].values[aligned_indices[:, 0]] - 
                                                      asks['timestamp'].values[aligned_indices[:, 1]])) <= cut_off_td]
            #print('matched_indices : ', matched_indices )
            #print(bids['event.id'].values[matched_indices[:, 0]],asks['event.id'].values[matched_indices[:, 1]])
            #print(np.column_stack((bids['event.id'].values[matched_indices[:, 0]], 
            #                            asks['event.id'].values[matched_indices[:, 1]])))
            res.extend(np.column_stack((bids['event.id'].values[matched_indices[:, 0]], 
                                        asks['event.id'].values[matched_indices[:, 1]])))

    return np.array(res, dtype=int)

  matched = matcher()

  #pd.DataFrame(matched).to_csv('matched.csv')
  events['matching.event']=np.nan
  # matched[:, 0] = bid event ids, matched[:, 1] = ask event ids
  matched_bids = pd.DataFrame(matched, columns=['event.id', 'matching.event'])
  matched_asks = pd.DataFrame(matched, columns=['matching.event','event.id'])
  events = pd.merge(events,matched_bids, on='event.id', how='left')
  events = pd.merge(events, matched_asks, on='event.id', how='left')

  # Combine columns into a single 'matching.event' column
  events['matching.event'] = events['matching.event_y'].fillna(events['matching.event_x']).fillna(events['matching.event'])

  # Drop the unnecessary columns
  events = events.drop(columns=['matching.event_x', 'matching.event_y'])
  return events