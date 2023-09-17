# CS4622-Lab1
CS4622 - Machine Learning : Lab 1

The task in this lab was to develop a model to predict 4 label outputs, by applying feature engineering techniques. We were provided with 2 datasets : "train.csv" and "valid.csv" for this lab.

Given below is the general outline of how the implementation has been done in this notebook:

<ol>
  <li>Read given data into pandas dataframe </li>
  <li>Fill missing values </li>
  <li>For all 4 labels,
    <ol>
      <li>Train a SVM model for all 256 features</li>
      <li>Get accuracy for the trained model</li>
      <li>Perform feature selection and train a new model for the reduced number of features</li>
      <li>Get accuracy scores for the new model and evaluate the performance</li>
    </ol>
  </li>
</ol>

For 'label_4' an additional step of preprocessing was done to oversample the data and ensure an equal distribution.
