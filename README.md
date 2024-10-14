# MyML
 This machine learning project is based on the book "Hands-on Machine Learning + Deep Learning."

## CH 1: k-Nearest Neighbors (KNN) Classifier
In this chapter, we implement the k-Nearest Neighbors (KNN) algorithm to classify two types of fish—Bream and Smelt—based on their length and weight. The dataset includes labeled fish measurements, and we explore visualization, model training, and prediction accuracy.

### Key Code
```python
# Data visualization
plt.scatter(bream_length, bream_weight, label='Bream')
plt.scatter(smelt_length, smelt_weight, label='Smelt')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/2df7947b-d0d6-4b8a-9299-ba06f9d88c56)

```python
# Combine lengths and weights for fish data
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1] * 35 + [0] * 14  # 1 for Bream, 0 for Smelt

# Train KNN model
kn = KNeighborsClassifier(n_neighbors=5)
kn.fit(fish_data, fish_target)
```

## CH 2: Data Preprocessing and Standardization with k-Nearest Neighbors (KNN)
In Chapter 2, we focus on how data preprocessing, especially standardization, can significantly affect model accuracy. We implement K-Nearest Neighbors (KNN) for classifying fish species (Bream and Smelt) and demonstrate the importance of properly handling data scales.
### Key Code
```python
# Standardize the data (Z-score normalization)
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std

# Train KNN with standardized data
kn.fit(train_scaled, train_target)
print(kn.score(test_scaled, test_target))  # Accuracy after standardization
distances, indexes = kn.kneighbors([new])  # Find the nearest neighbors of the new fish.
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.show()  # The plot has changed because we standardized the features.
```
![image](https://github.com/user-attachments/assets/30d31f2d-e991-485c-9e8f-e12efb1355b8)

- Problem: Prediction errors due to different scales of features (length vs. weight).
- Solution: Standardize features to have a mean of 0 and a standard deviation of 1 using Z-score normalization.
- Result: After standardization, the KNN model achieved 100% accuracy, demonstrating the importance of data preprocessing.

## CH 3: Regression (KNN and Linear) and Regularization
In Chapter 3, we apply both K-Nearest Neighbors (KNN) regression and Linear Regression to predict the weight of a perch based on its length, height, and width. We also explore Polynomial Regression to fit non-linear data and introduce Regularization techniques like Ridge and Lasso to prevent overfitting.

### Key Code
- KNN Regression:
```python
knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)
print(knr.predict([[50]]))  # Predicting the weight of a 50 cm perch
```

- Linear Regression:
```python
lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.predict([[50]]))  # Predicting the weight of a 50 cm perch using Linear Regression
```

- Polynomial Regression:
```python
poly = PolynomialFeatures(degree=2, include_bias=False)
train_poly = poly.fit_transform(train_input)
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))  # Predicting using polynomial features
```

- Regularization (Ridge and Lasso):
```python
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(test_scaled, test_target))  # Ridge regression evaluation

lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(test_scaled, test_target)) # Lasso regression evaluation
```
Ridge Regression: R² Scores vs. Alpha Values
![image](https://github.com/user-attachments/assets/bf8f6c27-3e94-4af5-ad47-5308a9ce4c46)

The best alpha value is where the two lines are closest, and the test score is highest, which is at -1, or 10^-1.

Lasso Regression: R² Scores vs. Alpha Values
![image](https://github.com/user-attachments/assets/63f92935-4362-40f3-a8e0-ec54f6f0098a)

Moving left shows overfitting, and moving right shows underfitting. The ideal alpha is around 1, which is 10^1.
