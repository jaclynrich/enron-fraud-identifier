# Identifying Fraud from Enron Financial Data and Email Metadata

The goal of this project is to determine which employees were persons of interest in the fraud case against Enron, using financial and email metadata.  Persons of interest are defined as individuals who were indicted in the fraud, reached a settlement or plea deal with the government, or testified in exchange for immunity.  I used Naive Bayes and Random Forest models in conjunction with feature engineering, feature selection, and parameter tuning to create a model to identify employees connected with the fraud case.

*  poi_id.ipynb - full report containing:
   - data cleaning
   - feature selection
   - feature engineering
   - trying out different models
   - final model
   - dumps classifier to my_classifier.pkl
   - dumps full altered data set to my_dataset.pkl
   - dumps list of all features in the data set (original and engineered) to my_feature_list.pkl
*  poi_id.py - streamlined version of poi_id.ipynb that runs the necessary code to clean the data, generate features, train the best classifier, tune parameters, and dump them to the same pickle files as above
*  tester.py - code that loads pickled files above and tests classifier performance
*  resources.txt - list of resources used

Data:
*  final_project_dataset.pkl - full data set
*  enron_financial_data.pdf - Enron financial data for reference
*  poi_names.txt - list of persons of interest
