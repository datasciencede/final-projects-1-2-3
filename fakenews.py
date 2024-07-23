import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# Load the datasets
train_path = '/Users/mohitr/Downloads/train.csv'
test_path = '/Users/mohitr/Downloads/test.csv'
predictions_path = '/Users/mohitr/Desktop/fake news 2/predictions.csv'

@st.cache_data
def load_data():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

train_df, test_df = load_data()

# Fill missing values in 'title', 'author', and 'text' columns with 'unknown'
train_df['title'].fillna('unknown', inplace=True)
train_df['author'].fillna('unknown', inplace=True)
train_df['text'].fillna('unknown', inplace=True)

test_df['title'].fillna('unknown', inplace=True)
test_df['author'].fillna('unknown', inplace=True)
test_df['text'].fillna('unknown', inplace=True)

# Combine 'title', 'author', and 'text' into a single feature 'content'
train_df['content'] = train_df['title'] + ' ' + train_df['author'] + ' ' + train_df['text']
test_df['content'] = test_df['title'] + ' ' + test_df['author'] + ' ' + test_df['text']

# Create TF-IDF vectorizer and logistic regression model
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(train_df['content'])
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, train_df['label'])

# Menu bar
st.sidebar.title("Menu")
options = ["Home", "EDA", "Model Training", "Prediction", "Test Your Own Article"]
choice = st.sidebar.radio("Go to", options)

if choice == "Home":
    st.title("Fake News Classification")
    st.write("This application allows you to classify news articles as reliable or unreliable using machine learning techniques.")

elif choice == "EDA":
    st.title("Exploratory Data Analysis")

    st.write("## Distribution of Target Variable")
    fig, ax = plt.subplots()
    sns.countplot(x='label', data=train_df, ax=ax)
    st.pyplot(fig)

    st.write("## Missing Values in Training Data")
    missing_values = train_df.isnull().sum()
    st.write(missing_values)

    st.write("## Distribution of Article Length")
    train_df['text_length'] = train_df['text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    fig, ax = plt.subplots()
    sns.histplot(train_df['text_length'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

    st.write("## Word Cloud for Reliable Articles")
    reliable_text = ' '.join(train_df[train_df['label'] == 0]['text'].dropna().values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reliable_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    st.write("## Word Cloud for Unreliable Articles")
    unreliable_text = ' '.join(train_df[train_df['label'] == 1]['text'].dropna().values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(unreliable_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

elif choice == "Model Training":
    st.title("Model Training")

    X = train_df['content']
    y = train_df['label']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_val_pred = model.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    st.write(f"### Accuracy: {accuracy}")
    st.write(f"### Precision: {precision}")
    st.write(f"### Recall: {recall}")
    st.write(f"### F1 Score: {f1}")

    # Confusion matrix
    st.write("## Confusion Matrix")
    cm = confusion_matrix(y_val, y_val_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Predictions and accuracy
    Y_pred = model.predict(X_val_tfidf)
    score = accuracy_score(y_val, Y_pred)
    st.write(f'Accuracy: {round(score*100, 2)}%')
    cm = confusion_matrix(y_val, Y_pred)
    st.write("## Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['FAKE Data', 'REAL Data'], yticklabels=['FAKE Data', 'REAL Data'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

elif choice == "Prediction":
    st.title("Prediction")

    # Check if predictions already exist
    if os.path.exists(predictions_path):
        st.write("## Existing Predictions")
        predictions_df = pd.read_csv(predictions_path)
        st.write(predictions_df.head(10))
    else:
        st.write("## Generating New Predictions")

        X_test_tfidf = vectorizer.transform(test_df['content'])
        test_predictions = model.predict(X_test_tfidf)

        test_df['label'] = test_predictions

        st.write("## Sample Predictions")
        st.write(test_df[['id', 'title', 'label']].head(10))

        # Save the predictions
        test_df[['id', 'label']].to_csv("predictions_csv", index=False)
        st.write(f"Predictions saved to `{"predictions_csv"}`.")

elif choice == "Test Your Own Article":
    st.title("Test Your Own News Article")

    st.write("Enter the details of the news article you want to classify:")

    title = st.text_input("Title")
    author = st.text_input("Author")
    text = st.text_area("Text")

    if st.button("Classify"):
        if not title or not author or not text:
            st.write("Please fill in all the fields")
        else:
            # Combine the input into a single feature
            input_content = f"{title} {author} {text}"
            
            # Transform the input using the trained TF-IDF vectorizer
            input_tfidf = vectorizer.transform([input_content])
            
            # Predict the label using the trained model
            prediction = model.predict(input_tfidf)[0]
            
            if prediction == 0:
                st.write("### The article is classified as: Reliable")
            else:
                st.write("### The article is classified as: Unreliable")
