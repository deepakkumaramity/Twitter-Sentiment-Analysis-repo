"""
Twitter Sentiment Analysis - Deepak Kumar
"""

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv("data/tweets.csv")

# Sentiment analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["tweet"].apply(get_sentiment)

# Save results
df.to_csv("data/tweets_with_sentiment.csv", index=False)

# Plot sentiment distribution with watermark
plt.figure(figsize=(6,4))
df["Sentiment"].value_counts().plot(kind="bar", color=["green","red","blue"])
plt.title("Sentiment Distribution - Deepak Kumar")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.text(0.5, -0.2, 'Deepak Kumar', fontsize=12, color='gray',
         ha='center', va='center', alpha=0.5, transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig("images/sentiment_distribution.png")
plt.close()

# WordCloud with watermark
all_text = " ".join(df["tweet"])
wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud - Deepak Kumar")
plt.text(0.5, -0.1, 'Deepak Kumar', fontsize=14, color='gray',
         ha='center', va='center', alpha=0.5, transform=plt.gca().transAxes)
plt.savefig("images/wordcloud.png")
plt.close()

print("Analysis complete! Check images and data folder.")
