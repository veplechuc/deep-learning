import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


feature_names = [
    "party",
    "handicapped-infants",
    "water-project-cost-sharing",
    "adoption-of-the-budget-resolution",
    "physician-fee-freeze",
    "el-salvador-aid",
    "religious-groups-in-schools",
    "anti-satellite-test-ban",
    "aid-to-nicaraguan-contras",
    "mx-missle",
    "immigration",
    "synfuels-corporation-cutback",
    "education-spending",
    "superfund-right-to-sue",
    "crime",
    "duty-free-exports",
    "export-administration-act-south-africa",
]

# load the data from file
voting_data = pd.read_csv(
    "./../data/house-votes-84.data.txt", na_values=["?"], names=feature_names
)

print(voting_data.head())
# check info looking at the descibe
print(voting_data.describe())

# dropping missing data regardint the 435 and there are different values on different columns
voting_data.dropna(inplace=True)
print(voting_data.describe())

# replacing char vlues by numbers
voting_data.replace(("y", "n"), (1, 0), inplace=True)
voting_data.replace(("democrat", "republican"), (1, 0), inplace=True)

all_features = voting_data[feature_names].values
all_classes = voting_data["party"].values

print("All features ->", all_features)
print("All classes ->", all_classes)


def create_model():

    model = Sequential()

    model.add(Dense(32, input_dim=16, kernel_initializer="normal", activation="relu"))
    # hidden layer of 16 units
    model.add(Dense(16, kernel_initializer="normal", activation="relu"))
    # output layer binary classification
    model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# wrap keras model ina acompatible with scikit-learn
estimator = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)

# use scikit-learn to evaluate the model
cv_scores = cross_val_score(estimator, all_features, all_classes, cv=10)
print(cv_scores.mean())
