import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_dates(start_date, num_days):
    return [start_date + timedelta(days=i) for i in range(num_days)]

def generate_synthetic_data(num_shipments=1000):
    print("Generating synthetic data...")
    
    locations = ["Mumbai Port", "Shanghai Port", "Los Angeles Port", "Rotterdam Port", "Dubai Port", "Singapore Port"]
    items = ["Electronics", "Auto Parts", "Textiles", "Pharmaceuticals", "Machinery"]
    
    start_date = datetime(2023, 1, 1)
    
    # 1. Generate ERP Data (Shipments)
    shipments = []
    for i in range(num_shipments):
        origin = random.choice(locations)
        destination = random.choice([loc for loc in locations if loc != origin])
        item = random.choice(items)
        
        # Dispatch date
        dispatch_date = start_date + timedelta(days=random.randint(0, 365))
        
        # Planned transit time (e.g., 10 to 30 days)
        transit_days = random.randint(10, 30)
        planned_eta = dispatch_date + timedelta(days=transit_days)
        
        # Base delay probability
        base_delay_prob = 0.1
        
        # Introduce some patterns: Shanghai -> LA electronics often delayed
        if origin == "Shanghai Port" and destination == "Los Angeles Port":
            base_delay_prob += 0.2
        if item == "Electronics":
            base_delay_prob += 0.1
            
        is_delayed = random.random() < base_delay_prob
        
        delay_days = 0
        if is_delayed:
            # Delay duration
            delay_days = int(np.random.gamma(shape=2.0, scale=3.0)) # Skewed distribution
            if delay_days == 0: delay_days = 1
            
        actual_arrival = planned_eta + timedelta(days=delay_days)
        
        shipments.append({
            "shipment_id": f"SHP{10000+i}",
            "origin": origin,
            "destination": destination,
            "item": item,
            "dispatch_date": dispatch_date.strftime("%Y-%m-%d"),
            "planned_eta": planned_eta.strftime("%Y-%m-%d"),
            "actual_arrival": actual_arrival.strftime("%Y-%m-%d"),
            "delay_days": delay_days,
            "is_delayed": 1 if delay_days > 0 else 0
        })
        
    df_shipments = pd.DataFrame(shipments)
    df_shipments.to_csv("shipments.csv", index=False)
    print(f"Created shipments.csv with {len(df_shipments)} records.")

    # 2. Generate Weather Logs
    weather_conditions = ["Clear", "Rain", "Heavy Rain", "Storm", "Cyclone", "Fog"]
    weather_severities = {"Clear": 0, "Rain": 1, "Heavy Rain": 3, "Storm": 7, "Cyclone": 10, "Fog": 4}
    
    weather_logs = []
    dates = generate_dates(start_date, 400) # Cover whole year + transit
    
    for date in dates:
        for loc in locations:
            # Default to clear
            cond = "Clear"
            
            # Random chance of bad weather
            if random.random() < 0.15:
                cond = random.choice(["Rain", "Heavy Rain", "Fog"])
            # Rare extreme events
            elif random.random() < 0.02:
                cond = random.choice(["Storm", "Cyclone"])
                
            weather_logs.append({
                "date": date.strftime("%Y-%m-%d"),
                "location": loc,
                "weather_condition": cond,
                "weather_severity": weather_severities[cond]
            })
            
    df_weather = pd.DataFrame(weather_logs)
    df_weather.to_csv("weather_logs.csv", index=False)
    print(f"Created weather_logs.csv with {len(df_weather)} records.")

    # 3. Generate News Alerts (Text Data)
    news_alerts = []
    # Generate some alerts corresponding to bad weather or random events
    bad_weather_events = df_weather[df_weather['weather_severity'] >= 7]
    
    for _, row in bad_weather_events.iterrows():
        # 50% chance a severe weather event makes the news
        if random.random() < 0.5:
            snippets = [
                f"{row['weather_condition']} halts operations at {row['location']}.",
                f"Severe {row['weather_condition'].lower()} approaching {row['location']}, expect delays.",
                f"Port of {row['location']} under lockdown due to {row['weather_condition']}."
            ]
            news_alerts.append({
                "date": row['date'],
                "location": row['location'],
                "text_snippet": random.choice(snippets),
                "source": "NewsAPI Mock"
            })
            
    # Add random port strikes/congestion news
    for _ in range(50):
        date = random.choice(dates).strftime("%Y-%m-%d")
        loc = random.choice(locations)
        snippets = [
            f"Workers strike at {loc} causing massive backlog.",
            f"Customs system failure at {loc} delays processing by 48 hours.",
            f"High congestion reported at {loc}, vessels waiting at anchorage."
        ]
        news_alerts.append({
            "date": date,
            "location": loc,
            "text_snippet": random.choice(snippets),
            "source": "Port Authority Mock"
        })
        
    df_news = pd.DataFrame(news_alerts)
    df_news.sort_values(by="date", inplace=True)
    df_news.to_csv("news_alerts.csv", index=False)
    print(f"Created news_alerts.csv with {len(df_news)} records.")
    
    print("Data generation complete!")

if __name__ == "__main__":
    generate_synthetic_data(1500)
