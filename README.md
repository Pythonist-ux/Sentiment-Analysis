# Sentiment Analysis with Python

## Introduction

Sentiment Analysis is a Python-based approach for extracting sentiment from text data, as exemplified in the provided code. This process involves several steps, including data loading, preprocessing, splitting into training and testing sets, text vectorization using TF-IDF, training a Multinomial Naive Bayes machine learning model, sentiment prediction, accuracy calculation, and the generation of a detailed classification report. Additionally, the code identifies influential words for each sentiment class and analyzes the word frequency distribution in the text data. The project concludes by showcasing a custom tweet prediction, demonstrating the real-world utility of the sentiment analysis model. This process is versatile and applicable to tasks like social media sentiment tracking and customer feedback analysis, providing actionable insights from unstructured text data.

## Working

1. **Data Preparation:**
   The code starts by loading a dataset containing text data and corresponding sentiment labels. It ensures data quality by removing rows with missing values.

2. **Feature and Target Definition:**
   The text data is divided into features (X) and sentiment labels (y).

3. **Data Splitting:**
   The dataset is split into a training set and a testing set to train and evaluate the sentiment analysis model.

4. **Text Vectorization:**
   TF-IDF vectorization is applied to convert text data into numerical features. This quantifies the importance of words within documents and across the dataset.

5. **Model Training:**
   A Multinomial Naive Bayes classifier is trained using the TF-IDF-transformed training data. The classifier learns to associate text features with sentiment labels.

6. **Sentiment Prediction:**
   The trained model is used to predict sentiment labels for the text data in the testing set.

7. **Model Evaluation:**
   The code calculates the accuracy of sentiment predictions to assess the model's performance. It generates a detailed classification report providing precision, recall, F1-score, and support for each sentiment class.

8. **Top Words for Sentiment Classes:**
   The code identifies the most influential words for each sentiment class, helping understand which words strongly correlate with particular sentiments.

9. **Word Frequency Distribution:**
   The overall word frequency distribution in the text data is analyzed, and the top 20 most frequently occurring words are displayed.

10. **Custom Tweet Prediction:**
    The code showcases the practical application of the model by predicting the sentiment of a user-provided tweet using the trained model.

    ![image](https://github.com/Pythonist-ux/Sentiment-Analysis/assets/83156291/6beac5e7-c5f3-49e1-a1ed-2055a38dc53c)
    
    ![image](https://github.com/Pythonist-ux/Sentiment-Analysis/assets/83156291/4b334dab-c81e-433e-93b1-8b4574147710)



## Usage

In a real-world scenario, the provided sentiment analysis code can be applied to analyze sentiments in a dataset containing text data. Let's break down how you might use this code:
1. Data Preparation:

    What it does: Loads a dataset (Twitter_Data.csv) containing text data and corresponding sentiment labels. Cleans the data by removing rows with missing values.

    In real life: You would replace Twitter_Data.csv with your own dataset, ensuring that it has columns for text data (clean_text) and corresponding sentiment labels (category).

2. Model Training:

    What it does: Splits the dataset into training and testing sets, vectorizes the text data using TF-IDF, and trains a Multinomial Naive Bayes classifier.

    In real life: This step is crucial for building a model that can predict sentiments accurately. You would train the model on a sufficiently large and representative dataset.

3. Sentiment Prediction:

    What it does: Uses the trained model to predict sentiment labels for the text data in the testing set.

    In real life: After training the model, you can apply it to new, unseen data to predict sentiments. This is useful for making predictions on fresh social media data, customer reviews, or any other textual data.

4. Model Evaluation:

    What it does: Calculates the accuracy of sentiment predictions and generates a detailed classification report.

    In real life: This step helps you assess the performance of your model on unseen data. The classification report provides metrics like precision, recall, F1-score, and support for each sentiment class.

5. Top Words for Sentiment Classes:

    What it does: Identifies the most influential words for each sentiment class.

    In real life: Understanding which words strongly correlate with particular sentiments can provide insights into why certain predictions are made.

6. Word Frequency Distribution:

    What it does: Analyzes the overall word frequency distribution in the text data and displays the top 20 most frequently occurring words.

    In real life: This analysis gives you an overview of the most common words in your dataset, which can be valuable for understanding the language used.

7. Example Custom Tweet and Prediction:

    What it does: Showcases the practical application of the model by predicting the sentiment of a user-provided tweet.

    In real life: You can use the model to predict sentiments for new, user-generated content, allowing you to apply the sentiment analysis to real-time data.

Note:

    Ensure that your dataset (Twitter_Data.csv) is appropriately formatted with the necessary columns.
    It's crucial to continually update and retrain your model with new data to maintain its accuracy in real-world scenarios.
    Adjust the code as needed based on your specific requirements and dataset characteristics.

This sentiment analysis code provides a foundation for automating sentiment assessment and gaining actionable insights from text data in various applications.




 
