# File: src/sentiment_analysis.py

from typing import List
import pandas as pd
from transformers import pipeline
from src.utils import get_logger

logger = get_logger()

class SentimentAnalyzer:
    """
    Analyzes sentiment from text data using pre-trained models.
    """

    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        Initializes the SentimentAnalyzer.

        Args:
            model_name (str): Name of the pre-trained sentiment analysis model.
        """
        self.pipeline = pipeline("sentiment-analysis", model=model_name)
        logger.info(f"Initialized SentimentAnalyzer with model: {model_name}")

    def analyze_sentiment(self, texts: List[str]) -> pd.Series:
        """
        Analyzes sentiment for a list of texts.

        Args:
            texts (List[str]): List of text strings.

        Returns:
            pd.Series: Sentiment scores.
        """
        results = self.pipeline(texts)
        sentiments = [result['score'] if result['label'] in ['POSITIVE', '5 stars'] else -result['score'] for result in results]
        return pd.Series(sentiments, index=range(len(sentiments)))

    def add_sentiment_to_df(self, df: pd.DataFrame, text_column: str = "news_headline", sentiment_column: str = "sentiment") -> pd.DataFrame:
        """
        Adds sentiment scores to the DataFrame based on a text column.

        Args:
            df (pd.DataFrame): Original DataFrame.
            text_column (str): Column containing text data.
            sentiment_column (str): Name of the new sentiment column.

        Returns:
            pd.DataFrame: DataFrame with added sentiment scores.
        """
        if text_column not in df.columns:
            logger.warning(f"Text column '{text_column}' not found in DataFrame.")
            df[sentiment_column] = 0
            return df

        texts = df[text_column].tolist()
        df[sentiment_column] = self.analyze_sentiment(texts)
        logger.info(f"Added sentiment scores to DataFrame as '{sentiment_column}'.")
        return df

# Example usage
# sentiment_analyzer = SentimentAnalyzer()
# df = sentiment_analyzer.add_sentiment_to_df(df, text_column="news_headline")
