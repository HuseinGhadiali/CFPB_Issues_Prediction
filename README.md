# CFPB Complaints Prediction: Predicting Issues from Consumer Narratives

Welcome to this repository, where we explore the power of machine learning to predict issues based on consumer complaints using the Consumer Financial Protection Bureau (CFPB) dataset. This dataset contains valuable information about consumer experiences in the financial marketplace, and our goal is to accurately predict the issues based on their narratives.

Our project is divided into four Jupyter notebooks, each building upon the previous to create a comprehensive analysis.

## CleaningComplaints: Preparing the Data
The first step in our analysis is to prepare the data. In the `CleaningComplaints` notebook, we remove unnecessary columns and handle missing values. We also preprocess the text data by lemmatizing it, mapping treebank part of speech tags to WordNet part of speech tags from NLTK. This step is crucial in ensuring that our models can accurately understand the content of the complaints.

## ComplaintsTFIDF: Building Predictive Models
Next, we move on to building predictive models. In the `Complaintstfidf` notebook, we split the data into training and testing sets with a test size of 0.2 using `TfidfVectorizer`. We then train three different models: Logistic Regression, Linear SVC, and Naive Bayes. Each model achieves an accuracy of around 60-65%, providing a solid baseline for our analysis.

To better understand our results, we create several visualizations. These include a Confusion Matrix to see where our models are making mistakes, an ROC Curve to evaluate their performance, a WordCloud to visualize the most common words in Consumer Complaints, and a bar plot to compare the accuracy of each model.

## ComplaintstfidfFW: Improving our Models
In the `ComplaintstfidfFW` notebook, we build upon our previous work by using FuzzyWuzzy to merge similar issues together. We also drop value counts less than 100 to focus on the most common issues. These changes help improve the accuracy of our models.

## Complaintstransformers: Fine-tuning with DistilBERT
Finally, in the `ComplaintsTransformers` notebook, we use the simple transformers library and the distilbert model. We fine-tune the model by creating training and testing datasets with a test size of 0.3. The model achieves an accuracy of 72.94%.
