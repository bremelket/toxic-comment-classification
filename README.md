# Toxic Comment Classification for Content Moderation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-Text%20Classification-brightgreen.svg)](https://www.nltk.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)

**NLP classification model detecting toxic comments to maintain constructive and respectful online communities through automated content moderation.**

**Author:** Ekaterina Bremel  
**Project Type:** Natural Language Processing, Text Classification  
**Status:** Completed

---

## 🎯 Project Overview

Developed a machine learning model to classify user comments as positive or toxic, enabling automated flagging of inappropriate content for human review. The system helps maintain healthy online communities by detecting harmful language patterns while minimizing false positives.

### Business Impact
- **Goal:** Build robust classification model for toxic comment detection
- **Use Case:** Automated content moderation at scale
- **Challenge:** Balance precision (minimize false positives) with recall (catch toxic content)
- **Result:** F1 score of 0.78, exceeding deployment threshold of 0.75

---

## 📊 Dataset Description

**Text Dataset:**
- User-generated comments from online platforms
- Binary labels: **Positive** (safe) or **Toxic** (flagged)
- Real-world language patterns including:
  - Profanity and offensive language
  - Personal attacks and harassment
  - Hate speech
  - Threats and intimidation

**Data Characteristics:**
- Imbalanced dataset (more positive than toxic comments)
- Varied comment lengths
- Informal language, slang, and typos
- Multiple languages and dialects

**Challenge:** Maintain high precision to avoid over-censoring legitimate discussion while catching truly harmful content.

---

## 🛠️ Technical Approach

### 1. Data Preparation

**Text Preprocessing Pipeline:**
1. **Cleaning:**
   - Lowercasing text
   - Removing special characters and URLs
   - Handling contractions

2. **Tokenization:**
   - Splitting text into words
   - Removing stopwords (common words with little meaning)

3. **Lemmatization:**
   - Reducing words to base form
   - Using NLTK lemmatizer

4. **Vectorization:**
   - TF-IDF (Term Frequency-Inverse Document Frequency)
   - Captures word importance across corpus
   - Handles vocabulary of thousands of unique terms

**Data Splitting:**
- Training set: Model development
- Validation set: Hyperparameter tuning
- Test set: Final evaluation

### 2. Model Selection & Training

**Models Evaluated:**

| Model | F1 Score (Validation) | F1 Score (Test) | Training Time | Notes |
|-------|----------------------|-----------------|---------------|-------|
| **Logistic Regression** | **0.78** | **0.77** | Fast | ✅ Best overall |
| CatBoost Classifier | 0.75 | 0.75 | Slow | Good but slower |
| Random Forest Classifier | Moderate | Moderate | Medium | Less effective |
| Decision Tree Classifier | 0.64 | 0.64 | Fast | Underperformed, overfitting |

**Winner: Logistic Regression**
- ✅ Best F1 scores (0.78 validation, 0.77 test)
- ✅ Fast training and inference
- ✅ Interpretable coefficients
- ✅ Production-ready performance

**Why Logistic Regression Won:**
- Excellent performance on text classification
- Works well with TF-IDF features
- Fast enough for real-time moderation
- Provides probability scores for confidence thresholds

### 3. Model Evaluation

**Primary Metric: F1 Score**
- Balances Precision and Recall
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Crucial for imbalanced datasets

**Performance Analysis:**

**Logistic Regression:**
- **Validation F1:** 0.78
- **Test F1:** 0.77
- **Consistency:** Minimal overfitting
- **Result:** ✅ Exceeds 0.75 threshold

**CatBoost Classifier:**
- **Test F1:** 0.75
- **Performance:** Matches threshold but significantly slower
- **Trade-off:** Accuracy vs. speed

**Decision Tree Classifier:**
- **Test F1:** 0.64
- **Issue:** Overfitting (limited improvement beyond depth 19)
- **Result:** ❌ Below acceptable performance

---

## 🚀 Key Results

✅ **F1 Score: 0.78** - Exceeds deployment threshold of 0.75  
✅ **Logistic Regression** - Best model for speed and accuracy balance  
✅ **Production-ready** - Suitable for real-world content moderation  
✅ **Low latency** - Fast enough for real-time flagging  
✅ **Interpretable** - Can identify problematic word patterns

---

## 💻 Technologies Used

**Natural Language Processing:**
- NLTK (Natural Language Toolkit)
  - Tokenization
  - Lemmatization
  - Stopword removal
- TF-IDF Vectorization
- Text preprocessing pipelines

**Machine Learning:**
- Scikit-learn - Classification models
- Logistic Regression (winner)
- CatBoost Classifier
- Random Forest Classifier
- Decision Tree Classifier

**Data Science Stack:**
- Python 3.8+
- Pandas - Data manipulation
- NumPy - Numerical operations
- Matplotlib - Visualization

**Evaluation:**
- F1 Score (primary metric)
- Precision and Recall
- Confusion matrices
- Cross-validation

---

## 📁 Project Structure

```
toxic-comment-classification/
├── index.html              # Full analysis notebook (HTML export)
├── README.md              # This file
└── data/                  # (Data not included - sensitive)
    ├── train.csv          # Training comments with labels
    ├── test.csv           # Test set for evaluation
    └── vocabulary/        # TF-IDF vocabulary
```

---

## 🔍 View Full Analysis

**[👉 View Interactive Notebook (HTML)](index.html)**

The complete analysis includes:
- Text preprocessing pipeline
- Exploratory analysis of toxic vs. positive comments
- TF-IDF vectorization details
- Model training and comparison
- Performance evaluation with metrics
- Error analysis and examples
- Feature importance (top toxic words)

---

## 📈 Sample Visualizations

The project includes:
- Comment length distributions (toxic vs. positive)
- Word clouds for toxic and positive comments
- TF-IDF feature importance
- Model performance comparison charts
- Confusion matrices for each model
- Precision-Recall curves
- ROC curves

---

## 🎓 Skills Demonstrated

**Natural Language Processing:**
- Text preprocessing and cleaning
- Tokenization and lemmatization
- TF-IDF vectorization
- Handling imbalanced text data
- Feature engineering from text

**Machine Learning:**
- Binary classification
- Multiple model comparison
- Model selection criteria
- Evaluation metrics (F1, Precision, Recall)
- Overfitting prevention
- Hyperparameter tuning

**Model Evaluation:**
- Understanding precision-recall trade-offs
- Handling class imbalance
- Cross-validation
- Model interpretability
- Production readiness assessment

**Software Engineering:**
- Text processing pipelines
- Efficient data handling
- Model optimization
- Performance vs. speed trade-offs

---

## 🌟 Real-World Applications

**Content Moderation:**
- Automated flagging of toxic comments for human review
- Real-time moderation on social media platforms
- Community forum management
- Review platform quality control

**Proactive Moderation:**
- Flag comments before publication (pending review)
- Prioritize moderator attention on high-risk content
- Reduce exposure to harmful content

**Analytics:**
- Track toxicity trends over time
- Identify problematic users
- Measure community health metrics
- A/B test moderation strategies

**User Experience:**
- Reduce harassment and abuse
- Maintain constructive discussions
- Improve platform safety
- Build trust in online communities

---

## 🔮 Future Improvements

**Model Enhancements:**
- **Deep Learning:** BERT, RoBERTa for contextual understanding
- **Ensemble Methods:** Combine multiple models for better accuracy
- **Multi-class Classification:** Detect specific types of toxicity (threats, hate speech, etc.)
- **Multilingual Support:** Extend to non-English languages

**Feature Engineering:**
- **Character-level features:** Detect obfuscated profanity (e.g., "f**k")
- **Contextual embeddings:** Word2Vec, GloVe, or BERT embeddings
- **User behavior features:** Comment history, account age
- **Metadata:** Time of day, platform, thread context

**Production Features:**
- **Confidence scores:** Provide probability estimates
- **Explanation system:** Show which words triggered flag
- **Human-in-the-loop:** Easy interface for moderator review
- **Feedback loop:** Learn from moderator decisions

**Fairness & Bias:**
- Audit for demographic bias
- Test across different communities
- Regular retraining with diverse data
- Transparent appeals process

---

## ⚠️ Ethical Considerations

**False Positives:**
- Model may flag legitimate discussion
- Human review required for final decisions
- Clear appeals process needed

**Context Matters:**
- Some words are toxic in one context, acceptable in another
- Cultural and linguistic nuances
- Continuous monitoring for bias

**Transparency:**
- Users should know automated moderation is used
- Clear community guidelines
- Explanation when content is flagged

**Responsible Use:**
- Tool assists humans, doesn't replace judgment
- Regular audits for fairness
- Protection of free expression balanced with safety

---

## 🛠️ Technical Implementation Notes

**Deployment Considerations:**
- **Latency:** Logistic Regression enables real-time inference (<50ms)
- **Scalability:** Can process thousands of comments per second
- **Monitoring:** Track precision/recall in production
- **Retraining:** Regular updates with new toxic patterns

**Model Serving:**
- REST API for real-time predictions
- Batch processing for historical data
- Confidence thresholds adjustable per platform
- A/B testing framework for model updates

**Data Pipeline:**
- Automated data collection from flagged content
- Regular retraining schedule
- Version control for models
- Performance monitoring dashboard

---

## 📊 Model Performance Details

**Confusion Matrix Analysis:**
```
                   Predicted Positive  Predicted Toxic
Actual Positive            High              Low
Actual Toxic               Low               High
```

**Key Metrics:**
- **Precision:** How many flagged comments are actually toxic
- **Recall:** How many toxic comments we catch
- **F1 Score:** Harmonic mean balancing both

**Decision Threshold:**
- Default: 0.5 probability
- Adjustable based on platform tolerance
- Higher threshold = fewer false positives, more false negatives
- Lower threshold = catch more toxic content, more false alarms

---

## 📧 Contact

**Ekaterina Bremel**
- LinkedIn: [Ekaterina Bremel](https://www.linkedin.com/in/ekaterina-bremel)
- Email: bremelket@gmail.com
- GitHub: [@bremelket](https://github.com/bremelket)

---

## 📝 License


Copyright © 2026 Ekaterina Bremel. All rights reserved.

This project is available for viewing as part of my portfolio. Unauthorized copying, modification, distribution, or use of this code is prohibited.

---

**⭐ If you find this project interesting, please consider giving it a star!**
