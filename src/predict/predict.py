import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

feature_names = [" BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]
#  importing the mammographic_masses.data.txt file into a Pandas dataframe

# convert missing data (indicated by a ?) into NaN
#  add the appropriate column names (BI_RADS, age, shape, margin, density, and severity):
df = pd.read_csv(
    "../predict/data/mammographic_masses.data.txt",
    sep=",",
    na_values=["?"],
    names=feature_names,
)
# prints first rows
# print(df.head())

# Evaluate whether the data needs cleaning
# print(df.describe())
df.dropna(inplace=True)
# print(df.describe())

# Create an array that extracts only the feature data we want to
#  work with (age, shape, margin, and density) and another array that contains the classes (severity)
features_names = ["Age", "Shape", "Margin", "Density"]
all_features = df[features_names].values
all_classes = df["Severity"].values
print(all_features)

# normalize the attribute data.
scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

# Now set up an actual MLP model using Keras:
def create_model():

    model = Sequential()
    # 4 feature inputs going into an 6-unit layer (more does not seem to help - in fact you can go down to 4)
    model.add(Dense(6, input_dim=4, kernel_initializer="normal", activation="relu"))
    # hidden layer of 4 units
    model.add(Dense(4, kernel_initializer="normal", activation="relu"))
    # output layer binary classification
    model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


# wrap keras model ina acompatible with scikit-learn
estimator = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
# use scikit-learn to evaluate the model
cv_scores = cross_val_score(estimator, all_features, all_classes, cv=10)
print(cv_scores.mean())

