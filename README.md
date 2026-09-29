# ANN Airline Passenger Satisfaction Prediction

A machine learning project that uses an Artificial Neural Network (ANN) to predict airline passenger satisfaction based on passenger information, travel details, and service ratings.

## Project Overview

The goal of this project is to predict whether an airline passenger is:

- Satisfied
- Neutral or Dissatisfied

The project uses an Artificial Neural Network built with TensorFlow/Keras and includes data preprocessing, categorical encoding, feature scaling, train-test splitting, and binary classification.

## Dataset

The project uses the Airline Passenger Satisfaction dataset.

The dataset contains passenger information, travel details, delays, and ratings for different airline services.

### Target Variable

`Satisfaction`

The target contains two classes:

- `Satisfied`
- `Neutral or Dissatisfied`

## Features Used

The model uses features including:

- Gender
- Age
- Customer Type
- Type of Travel
- Class
- Flight Distance
- Departure Delay
- Arrival Delay
- Departure and Arrival Time Convenience
- Check-in Service
- Online Boarding
- On-board Service
- Seat Comfort
- Leg Room Service
- Cleanliness
- Food and Drink
- In-flight Service
- In-flight Wifi Service
- In-flight Entertainment
- Baggage Handling

The following columns were removed during preprocessing:

- ID
- Ease of Online Booking
- Gate Location

## Data Preprocessing

The preprocessing pipeline handles numerical and categorical features separately.

### Numerical Features

The numerical preprocessing consists of:

1. Missing-value imputation using the median
2. Standardization using `StandardScaler`

### Categorical Features

The categorical preprocessing consists of:

1. Missing-value imputation using the most frequent value
2. One-hot encoding using `OneHotEncoder`

The preprocessing is implemented using Scikit-learn's `ColumnTransformer` and `Pipeline`.

```python
num_transformation = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaling", StandardScaler())
])

cat_transformation = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
