 

The goal of this project is to develop a system capable of: 

Classifying reviews as fake or real using both traditional machine learning , deep learning models, Transformers 

Clustering similar reviews to group related feedback together. 

Identifying underlying topics in the reviews to understand customer sentiments and issues. 

 

Approach: 

Data Collection: The Fake Review Dataset, containing product reviews with text and metadata such as ratings, category, label was provided. 

Preprocessing: 

 Importing Libraries 

libraries for preprocessing (pandas, re, nltk),  

tokenization (TreebankWordTokenizer), 

 and machine learning (gensim, sklearn, numpy). 

 Text Preprocessing: 

Tokenization: Split the review text into individual words or subword tokens. 

Stopword Removal: Remove common, unimportant words (e.g., "and", "the"). 

Lemmatization: Convert words to their base or dictionary form (e.g., "running" to "run"). 

TF-IDF/Word2Vec Vectorization: Convert the review text into a matrix of numerical features, capturing the importance of words in each review. 

Fake Review Classification (Supervised Learning): The primary objective is to classify reviews as fake or real based on their content.  

Train-Test Split    The dataset is split into training and testing sets                                         Model Training and Evaluation 

I  trained three Traditional Machine Learning models: Random Forest, Logistic Regression, and Support Vector Machine. 

Random Forest 

Trained using RandomForestClassifier. 

Evaluates predictions using classification_report and accuracy_score. 

Saves the model to a file. 

Logistic Regression 

Trained with LogisticRegression with max_iter=1000 for convergence. 

Evaluates predictions and saves the model. 

Support Vector Machine 

Trained using SVC (default kernel: RBF). 

Evaluates predictions and saves the model. 

Each model's performance is printed with: 

Classification Report: Precision, recall, f1-score, and support for each class. 

Accuracy: Overall classification accuracy. 

Hypertuned the model and saved it as .pkl file. 

 

 

Voting Classifier (Soft Voting) implementation: 

Ensemble Methods: Combine the outputs from different models to improve overall performance 

1. Voting Classifier Overview 

A Voting Classifier combines multiple models (Random Forest, Logistic Regression, SVM) to produce a final prediction: 

Soft Voting: Uses predicted probabilities from each model to compute a weighted average and selects the class with the highest probability. 

Models used: 

Random Forest (rf_model) 

Logistic Regression (lr_model) 

**Support Vector Machine (SVM)** (svm_model` with probability enabled) 

 

2. Training Process 

Fitting: The Voting Classifier is trained on the same dataset (X_train, y_train). 

Probability Estimation: SVM was retrained with probability=True to ensure compatibility with Soft Voting. 

 

3. Evaluation Metrics 

Classification Report 

The Voting Classifier provides precision, recall, F1-score, and support for each class. Here’s a detailed explanation: 

Precision: Proportion of correctly predicted positive observations to total predicted positives. 

Recall: Proportion of correctly predicted positive observations to all actual positives. 

F1-Score: Harmonic mean of precision and recall, balancing the two metrics. 

Support: The number of true instances for each class. 

Overall Accuracy 

Represents the proportion of correctly classified samples. 

Confusion Matrix 

Displays the counts of True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN) for each class 

 

 

 

TensorFlow: Deep learning models (e.g., LSTM, BERT) for review classification. 



Topic Modeling (Unsupervised Learning):Topic modeling techniques are used to extract key themes from the reviews. We use the following methods: 

Latent Dirichlet Allocation (LDA): A probabilistic model that assigns each word in a document to a topic. 

Non-Negative Matrix Factorization (NMF): Factorizes the matrix of word occurrences to extract topics. 

Implementation: 

Text Vectorization: Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert review text into numerical format. 

LDA or NMF: Apply these models to identify topics across all reviews. 

Clustering (Unsupervised Learning): Clustering groups similar reviews together, which helps identify feedback patterns or categorize reviews based on content similarity. 

K-Means Clustering: Assign reviews to K clusters (e.g., product categories, review types). 

Evaluation: Use classification metrics such as accuracy, precision, recall, and F1 score. 

 
