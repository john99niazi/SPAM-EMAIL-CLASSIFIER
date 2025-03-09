# 📧 Naive Bayes Spam Classifier

This is a simple **Naive Bayes-based spam classifier** implemented in Python using a **small dataset** due to system limitations. The approach is correct and can be scaled to larger datasets with more computational power.

## 🔍 Overview

The project demonstrates how to classify emails as **spam (1) or not spam (0)** using the **Naive Bayes algorithm**. It preprocesses text, calculates word frequencies, applies Laplace smoothing, and classifies emails based on computed probabilities.

## 🛠 Features

- **Text Preprocessing:** Tokenization and lowercasing  
- **Vocabulary Creation:** Assigns unique indices to words  
- **Word Frequency Calculation:** Counts occurrences per class (spam vs. not spam)  
- **Naive Bayes Probability Calculation:** Uses Laplace smoothing to avoid zero probabilities  
- **Classification Function:** Computes log probabilities and classifies emails  
- **Evaluation:** Measures model accuracy on the given dataset  
- **User Input Testing:** Allows users to input an email for classification  

## 🚀 How It Works

1. Converts emails into lowercase and tokenizes words.
2. Creates a vocabulary and calculates word frequency for both spam and non-spam emails.
3. Applies **Naive Bayes classification** by computing log probabilities.
4. Evaluates model performance using accuracy.
5. Allows the user to test custom emails for classification.

## 📊 Accuracy

The model is evaluated on a **small dataset** and achieves an accuracy of **~85-90%**, depending on the data. Performance can improve with a larger dataset.

## 🔧 Dependencies

- `pandas`
- `numpy`

## 🏃‍♂️ Running the Code

Simply run the script in a Python environment:

```bash
SPAM-EMIAL-CLASSIFIER.py
```

You can then enter a new email to classify whether it is **Spam** or **Not Spam**.

## 📌 Notes

- A **small dataset** was used because my system is too slow to handle large-scale training.
- This implementation follows the **correct Naive Bayes approach**, which can be expanded with more data.
- Performance may improve with **better feature extraction (TF-IDF, stemming, etc.)**.

## 🔮 Future Improvements

- **Support for larger datasets** with optimized training.
- **Integration with Scikit-learn** for improved efficiency.
- **Additional NLP preprocessing** (stopword removal, stemming, lemmatization).
- **TF-IDF transformation** to improve word importance detection.
- **Deploy as a Web API** for real-world email classification.

---

