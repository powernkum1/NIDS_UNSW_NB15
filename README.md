# Machine Learning-Based Network Intrusion Detection System (NIDS)

## Overview

This project focuses on building a Machine Learning-Based Network Intrusion Detection System (NIDS) using the UNSW-NB15 dataset. The goal of the project is to classify network traffic as either normal or malicious by applying multiple machine learning techniques and comparing their performance.

The project was developed as part of CIS735 – Machine Learning for Security.

The UNSW-NB15 dataset was selected because it contains modern network traffic and multiple attack categories, making it more realistic than older intrusion detection datasets such as KDD99.

\---

# Project Goals

* Build a machine learning-based intrusion detection system
* Compare multiple machine learning models
* Evaluate model performance using standard classification metrics
* Analyze feature importance and preprocessing techniques
* Explore tradeoffs between accuracy and computational efficiency

\---

# Dataset

Dataset Used:

* UNSW-NB15 Dataset

Dataset Source:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

\---

# Technologies Used

## Programming Language

* Python

## Libraries

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* scipy

\---

# Machine Learning Models

The project compares multiple machine learning algorithms:

1. Random Forest
2. Support Vector Machine (SVM)
3. Naive Bayes
4. Logistic Regression
5. Deep Neural Networks (future implementation)

\---

# Data Preprocessing

The preprocessing stage includes:

* Handling missing values
* Encoding categorical features
* Feature normalization
* Feature scaling
* Train/test splitting
* Removing duplicate records

\---

# Model Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* False Positive Rate

\---

# Example Workflow

1. Load UNSW-NB15 dataset
2. Preprocess network traffic data
3. Encode categorical variables
4. Train machine learning models
5. Test models on unseen traffic
6. Compare performance metrics
7. Analyze results

\---

# Current Progress

Completed:

* Project topic selection
* Dataset acquisition
* Initial literature review
* Experimental design planning
* Distance function and statistical analysis assignments

In Progress:

* Dataset preprocessing
* Feature engineering
* Baseline model implementation

Planned:

* Deep learning implementation
* Hyperparameter optimization
* Explainable AI analysis
* Final comparative evaluation

\---

# Research Motivation

Traditional signature-based intrusion detection systems often struggle to detect new or unknown attacks. Machine learning approaches can improve detection by learning traffic behavior patterns directly from the data.

The goal of this project is to explore how different machine learning techniques perform on modern intrusion detection tasks using realistic network traffic data.

\---

# Future Improvements

Future work may include:

* Real-time intrusion detection
* Deep learning architectures
* Explainable AI (XAI)
* Feature importance visualization
* Cloud-based deployment
* Streaming network analysis

\---

# References

1. UNSW-NB15 Dataset
https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Moustafa, N., \& Slay, J. (2015).
UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems.
3. Scikit-learn Documentation
https://scikit-learn.org/

\---

# Author

Michael Preko Nkum

Snr Software Developer | Machine Learning for Security

\---

# License

This project is for educational and research purposes.

