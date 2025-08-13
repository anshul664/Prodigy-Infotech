import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('twitter_training.csv')

# Display basic information about the dataset
print("\n=== DATASET OVERVIEW ===")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Basic statistics
print("\n=== BASIC STATISTICS ===")
print(df.describe())

# Analyze sentiment distribution
print("\n=== SENTIMENT ANALYSIS ===")
sentiment_counts = df.iloc[:, 2].value_counts()
print("Sentiment distribution:")
print(sentiment_counts)

# Analyze topics/brands
print("\n=== TOPICS/BRANDS ANALYSIS ===")
topic_counts = df.iloc[:, 1].value_counts()
print("Top 10 topics/brands:")
print(topic_counts.head(10))

# Create visualizations - 2 plots per figure for better readability

# Figure 1: Sentiment Distribution and Top Topics
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig1.suptitle('Twitter Sentiment Analysis - Part 1', fontsize=16, fontweight='bold')

# 1. Sentiment Distribution Pie Chart
ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'])
ax1.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
ax1.set_title('Overall Sentiment Distribution', fontweight='bold')

# 2. Top Topics/Brands Bar Chart
top_topics = topic_counts.head(10)
ax2.barh(range(len(top_topics)), top_topics.values)
ax2.set_xlim(2000, 2500)  # Adjust x-axis limits for better visibility
ax2.set_xticks([100*i for i in range(20, 26)])  # Set x-ticks for clarity
ax2.set_yticks(range(len(top_topics)))
ax2.set_yticklabels(top_topics.index)
ax2.set_title('Top 10 Topics/Brands', fontweight='bold')
ax2.set_xlabel('Number of Tweets')

plt.tight_layout()
plt.savefig('sentiment_analysis_part1.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 2: Sentiment by Topic Heatmap and Distribution
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Twitter Sentiment Analysis - Part 2', fontsize=16, fontweight='bold')

# 3. Sentiment by Topic Heatmap
sentiment_topic = pd.crosstab(df.iloc[:, 1], df.iloc[:, 2])
top_topics_for_heatmap = sentiment_topic.loc[topic_counts.head(8).index]
sns.heatmap(top_topics_for_heatmap, annot=True, fmt='d', cmap='YlOrRd', ax=ax3)
ax3.set_title('Sentiment Distribution by Topic', fontweight='bold')

# 4. Sentiment Distribution by Topic (Stacked Bar)
sentiment_topic_percent = top_topics_for_heatmap.div(top_topics_for_heatmap.sum(axis=1), axis=0) * 100
sentiment_topic_percent.plot(kind='bar', stacked=True, ax=ax4, colormap='Set3')
ax4.set_title('Sentiment Distribution by Topic (%)', fontweight='bold')
ax4.set_xlabel('Topics')
ax4.set_ylabel('Percentage')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
ax4.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.tight_layout()
plt.savefig('sentiment_analysis_part2.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 3: Tweet Length Analysis
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 6))
fig3.suptitle('Twitter Sentiment Analysis - Part 3', fontsize=16, fontweight='bold')

# 5. Tweet Length Analysis
df['tweet_length'] = df.iloc[:, 3].astype(str).str.len()
ax5.hist(df['tweet_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax5.set_title('Distribution of Tweet Lengths', fontweight='bold')
ax5.set_xlabel('Tweet Length (characters)')
ax5.set_ylabel('Frequency')

# 6. Average Tweet Length by Sentiment
avg_length_by_sentiment = df.groupby(df.iloc[:, 2])['tweet_length'].mean()
ax6.bar(avg_length_by_sentiment.index, avg_length_by_sentiment.values, color=['green', 'red', 'orange', 'blue'])
ax6.set_title('Average Tweet Length by Sentiment', fontweight='bold')
ax6.set_xlabel('Sentiment')
ax6.set_ylabel('Average Length (characters)')

plt.tight_layout()
plt.savefig('sentiment_analysis_part3.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional Analysis
print("\n=== DETAILED ANALYSIS ===")

# Sentiment analysis by topic
print("\nSentiment Analysis by Top Topics:")
for topic in topic_counts.head(5).index:
    topic_data = df[df.iloc[:, 1] == topic]
    sentiment_dist = topic_data.iloc[:, 2].value_counts()
    print(f"\n{topic}:")
    for sentiment, count in sentiment_dist.items():
        percentage = (count / len(topic_data)) * 100
        print(f"  {sentiment}: {count} tweets ({percentage:.1f}%)")

# Word frequency analysis for each sentiment
print("\n=== WORD FREQUENCY ANALYSIS ===")
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['cleaned_text'] = df.iloc[:, 3].apply(clean_text)

for sentiment in sentiment_counts.index:
    sentiment_tweets = df[df.iloc[:, 2] == sentiment]['cleaned_text']
    all_words = ' '.join(sentiment_tweets).split()
    word_freq = Counter(all_words)
    
    print(f"\nTop 10 words in {sentiment} tweets:")
    for word, freq in word_freq.most_common(10):
        if len(word) > 2:  # Filter out short words
            print(f"  {word}: {freq}")

# Note: WordCloud visualization removed due to import issues
print("\nWordCloud visualization skipped - focus on other analysis results")

print("\n=== SUMMARY INSIGHTS ===")
print(f"Total tweets analyzed: {len(df)}")
print(f"Number of unique topics/brands: {df.iloc[:, 1].nunique()}")
print(f"Most common sentiment: {sentiment_counts.index[0]} ({sentiment_counts.iloc[0]} tweets)")
print(f"Most discussed topic: {topic_counts.index[0]} ({topic_counts.iloc[0]} tweets)")
print(f"Average tweet length: {df['tweet_length'].mean():.1f} characters")

# Sentiment polarity analysis
positive_topics = df[df.iloc[:, 2] == 'Positive'].iloc[:, 1].value_counts().head(5)
negative_topics = df[df.iloc[:, 2] == 'Negative'].iloc[:, 1].value_counts().head(5)

print(f"\nTop 5 topics with positive sentiment:")
for topic, count in positive_topics.items():
    print(f"  {topic}: {count} tweets")

print(f"\nTop 5 topics with negative sentiment:")
for topic, count in negative_topics.items():
    print(f"  {topic}: {count} tweets")

print("\nAnalysis complete! Check the generated visualizations for detailed insights.")

