import argparse
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree



def hoge(input_folder: str):
    print(f"Hello, {input_folder}!")


def main():
    parser = argparse.ArgumentParser(description="A simple Python script.")
    parser.add_argument("input_folder", type=str, help="The name to greet.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args(['-v',"plots"])
    hoge(args.input_folder)
    if args.verbose:
        print("Verbose mode enabled.")


def plot_iris():
    data = sns.load_dataset("iris") 
    plot = sns.FacetGrid(data, col="species") 
    plot.map(plt.plot, "sepal_width") 
    plt.show()

def predictIrisFlower():
    print("\n")
    iris = datasets.load_iris()
    #print(iris.data.shape)
    X = iris.data
    y = iris.target
   
  
    #split the data into training and test sets, 80% training, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    #predict the test set
    y_pred = model.predict(X_test)
    print(f"Flower detection Accuracy: {accuracy_score(y_test, y_pred)}")

    #predict a new flower
    new_flower = [[5.1, 3.5, 1.5, 0.2]]
    prediction = model.predict(new_flower)
    iris_type = ['Setosa', 'Versicolour', 'Virginica']
    print(f"Iris Flower Prediction: {iris_type[prediction[0]]}")

    plot_iris()
    
def detectBreastCancer():
    print("\n")
    breast_cancer = datasets.load_breast_cancer()
    X = breast_cancer.data
    y= breast_cancer.target
    #print(breast_cancer.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  
    print(f"BreastCancer detection Accuracy: {accuracy_score(y_test, y_pred)}")



def predictDiabetes():
    print("\n")
    df = pd.read_csv('diabetes.csv')
    #print(df.head())

    X = df.drop('Outcome',axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)
    classifier_rf.fit(X_train, y_train)
    #print(classifier_rf.oob_score_)

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    params = {
        'max_depth': [2,3,5,10,20],
        'min_samples_leaf': [5,10,20,50,100,200],
        'n_estimators': [10,25,30,50,100,200]
    }


    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf,
                            param_grid=params,
                            cv = 4,
                            n_jobs=-1, verbose=1, scoring="accuracy")

  
    grid_search.fit(X_train, y_train)
    
    y_pred = classifier_rf.predict(X_test)  
    print(f"detection Accuracy of Diabetics patient: {accuracy_score(y_test, y_pred)}")
    
    #print(grid_search.best_score_)
    rf_best = grid_search.best_estimator_
    #print(rf_best)
 

    
if __name__ == "__main__":
    detectBreastCancer()
    predictDiabetes()
    predictIrisFlower()
   
    #main()