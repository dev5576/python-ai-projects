import argparse
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#from sklearn.feature_selection import RFE
#from sklearn.ensemble import RandomForestClassifier



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


def predictIrisFlower():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    #split the data into training and test sets, 80% training, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    #predict the test set
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    #predict a new flower
    new_flower = [[5.1, 3.5, 1.5, 0.2]]
    prediction = model.predict(new_flower)
    iris_type = ['Setosa', 'Versicolour', 'Virginica']
    print(f"Prediction: {iris_type[prediction[0]]}")



if __name__ == "__main__":
    predictIrisFlower()
    main()