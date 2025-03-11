import argparse
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import datasets
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, export_graphviz
from scipy.stats import randint
from scipy import stats
import graphlib
import graphviz
from IPython.display import Image
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage



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

    sns.tit("Diabetes Detection")
    sns.pairplot(df, hue='Outcome')
    plt.show()
 


def detectMarketing():
    print("\n")
    bank_data = pd.read_csv('bank-direct-marketing-campaigns.csv')
    bank_data['default'] = bank_data['default'].map({'no':0,'yes':1,'unknown':0})
    bank_data['y'] = bank_data['y'].map({'no':0,'yes':1})

    X = bank_data.drop('y', axis=1)
    y = bank_data['y']

    # Convert categorical string values to numeric values
    X = pd.get_dummies(X, drop_first=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy*100)

#reference https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
def hierarchicalCluster():
    x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

    data = list(zip(x, y))
    print(data)

    linkage_date = linkage(data, method='ward', metric='euclidean')
    dendrogram(linkage_date)
    plt.show()

    hierarchical_cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
    labels = hierarchical_cluster.fit_predict(data)
    print(labels)
    plt.scatter(x, y, c=labels)
    plt.show()


#reference: https://www.w3schools.com/python/python_ml_logistic_regression.asp
def logit2prob(logr, x):
    log_odds = logr.intercept_ + logr.coef_[0] * x
    odds = np.exp(log_odds) 
    prob = odds / (1 + odds)
    return prob



def logisticRegression():
    X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)

    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    logr = linear_model.LogisticRegression()
    logr.fit(X, y)

    predicted = logr.predict(np.array([3.46]).reshape(-1,1))
    print(predicted)

    log_odds = logr.coef_
    odds = np.exp(log_odds)
    print(odds)


    print(logit2prob(logr, X))
    
    
if __name__ == "__main__":
    #detectBreastCancer()
    #predictDiabetes()
    #predictIrisFlower()
    #detectMarketing()

    #hierarchicalCluster()
    logisticRegression()
   
   

    #main()