import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#mlflow.autolog() #this automatically logs the model, metrics, parameters, etc. without using mlflow.log_metric, mlflow.log_param, mlflow.log_artifact, mlflow.log_input, mlflow.log_output, mlflow.log_model, mlflow.set_tag, mlflow.set_experiment, mlflow.start_run, mlflow.end_run

import dagshub
dagshub.init(repo_owner='amit5631', repo_name='dagshub-mlflow-demo', mlflow=True) # for dagshub initialization using mlflow

mlflow.set_tracking_uri("https://dagshub.com/amit5631/dagshub-mlflow-demo.mlflow") # for dagshub

#mlflow.set_tracking_uri("http://127.0.0.1:5000/") # for local mlflow server

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Random Forest model
max_depth = 1

# apply mlflow

mlflow.set_experiment('iris-dt')

with mlflow.start_run():

    dt = DecisionTreeClassifier(max_depth=max_depth)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)

    mlflow.log_param('max_depth', max_depth)

    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    # Save the plot as an artifact
    plt.savefig("confusion_matrix.png")

    # mlflow code
    mlflow.log_artifact("confusion_matrix.png")

    # log the current file
    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(dt, "decision tree")

    mlflow.set_tag('author','amit')
    mlflow.set_tag('model','decision tree')

    import pandas as pd
    #logging the dataset
    test_df=pd.DataFrame(X_test,columns=iris.feature_names)
    test_df['variety']=y_test

    train_df=pd.DataFrame(X_train,columns=iris.feature_names)
    train_df['variety']=y_train

    train_df=mlflow.data.from_pandas(train_df)
    test_df=mlflow.data.from_pandas(test_df)

    mlflow.log_input(train_df)
    mlflow.log_input(test_df)
    
   
