import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

def process_nlp_data(input_file="news_alerts.csv", output_file="news_features.csv"):
    print("Loading NLP pipeline...")
    # Use a small pre-trained model for sentiment/risk
    # Using default sentiment model (distilbert-base-uncased-finetuned-sst-2-english)
    device = 0 if torch.cuda.is_available() else -1
    sentiment_analyzer = pipeline("sentiment-analysis", device=device)
    
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    if df.empty:
        print("No data to process.")
        return

    print("Extracting NLP risk scores...")
    risk_scores = []
    
    for text in tqdm(df['text_snippet'].tolist()):
        try:
            # Predict sentiment
            result = sentiment_analyzer(text)[0]
            # If label is NEGATIVE, risk is higher. If POSITIVE, risk is lower.
            if result['label'] == 'NEGATIVE':
                score = result['score'] # High score means very negative (high risk)
            else:
                score = 1.0 - result['score'] # High positive score means low risk
                
            risk_scores.append(score)
        except Exception as e:
            print(f"Error processing text: {text}. Error: {e}")
            risk_scores.append(0.5) # Default neutral risk
            
    df['nlp_risk_score'] = risk_scores
    
    # Save the processed features
    df.to_csv(output_file, index=False)
    print(f"NLP processing complete. Saved to {output_file}")

if __name__ == "__main__":
    process_nlp_data()
