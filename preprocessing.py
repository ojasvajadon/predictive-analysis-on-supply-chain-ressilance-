import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def preprocess_and_merge():
    print("Loading datasets...")
    try:
        shipments = pd.read_csv("shipments.csv")
        weather = pd.read_csv("weather_logs.csv")
        try:
            news = pd.read_csv("news_features.csv")
        except FileNotFoundError:
            print("news_features.csv not found. Falling back to news_alerts.csv with mock scores.")
            news = pd.read_csv("news_alerts.csv")
            # Generate mock nlp_risk_score if NLP hasn't run yet
            if 'nlp_risk_score' not in news.columns:
                 news['nlp_risk_score'] = np.random.uniform(0.1, 0.9, size=len(news))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Convert dates to datetime objects for easy comparison
    shipments['dispatch_date'] = pd.to_datetime(shipments['dispatch_date'])
    weather['date'] = pd.to_datetime(weather['date'])
    news['date'] = pd.to_datetime(news['date'])

    print("Merging features...")
    
    # We will build features for each shipment
    # Features: Origin, Destination, Item, Transit_Days (planned), 
    # Max Weather Severity at Origin (around dispatch), 
    # Max Weather Severity at Destination (around ETA)
    # Max NLP Risk Score at Origin (around dispatch)
    
    shipments['planned_eta'] = pd.to_datetime(shipments['planned_eta'])
    shipments['transit_days'] = (shipments['planned_eta'] - shipments['dispatch_date']).dt.days
    
    merged_data = []
    
    for idx, row in shipments.iterrows():
        origin = row['origin']
        dest = row['destination']
        dispatch = row['dispatch_date']
        eta = row['planned_eta']
        
        # Weather at origin (window: -2 to +2 days of dispatch)
        w_origin = weather[(weather['location'] == origin) & 
                           (weather['date'] >= dispatch - timedelta(days=2)) & 
                           (weather['date'] <= dispatch + timedelta(days=2))]
        origin_weather_severity = w_origin['weather_severity'].max() if not w_origin.empty else 0
        
        # Weather at destination (window: -2 to +2 days of ETA)
        w_dest = weather[(weather['location'] == dest) & 
                         (weather['date'] >= eta - timedelta(days=2)) & 
                         (weather['date'] <= eta + timedelta(days=2))]
        dest_weather_severity = w_dest['weather_severity'].max() if not w_dest.empty else 0
        
        # News/NLP risk at origin (window: -5 to +2 days of dispatch)
        n_origin = news[(news['location'] == origin) & 
                        (news['date'] >= dispatch - timedelta(days=5)) & 
                        (news['date'] <= dispatch + timedelta(days=2))]
        origin_nlp_risk = n_origin['nlp_risk_score'].max() if not n_origin.empty else 0.1 # Base risk
        
        # Combine all features
        feature_dict = {
            'shipment_id': row['shipment_id'],
            'origin': origin,
            'destination': dest,
            'item': row['item'],
            'transit_days': row['transit_days'],
            'origin_weather_severity': origin_weather_severity,
            'dest_weather_severity': dest_weather_severity,
            'origin_nlp_risk': origin_nlp_risk,
            'is_delayed': row['is_delayed'],
            'delay_days': row['delay_days']
        }
        merged_data.append(feature_dict)
        
    final_df = pd.DataFrame(merged_data)
    
    # Fill any NaNs with 0
    final_df.fillna(0, inplace=True)
    
    # One-hot encoding for categorical variables (origin, destination, item)
    print("Encoding categorical variables...")
    final_df = pd.get_dummies(final_df, columns=['origin', 'destination', 'item'], drop_first=False)
    
    final_df.to_csv("training_data.csv", index=False)
    print(f"Preprocessing complete. Saved {len(final_df)} rows to training_data.csv")

if __name__ == "__main__":
    preprocess_and_merge()
