# MyML
 This machine learning project is based on the book "Hands-on Machine Learning + Deep Learning."

## CH 1: k-Nearest Neighbors (KNN) Classifier
In this chapter, we implement the k-Nearest Neighbors (KNN) algorithm to classify two types of fish—Bream and Smelt—based on their length and weight. The dataset includes labeled fish measurements, and we explore visualization, model training, and prediction accuracy.

### Key Code
from sklearn.neighbors import KNeighborsClassifier

# Fish data: Bream (1) and Smelt (0)
fish_data = [[l, w] for l, w in zip(bream_length + smelt_length, bream_weight + smelt_weight)]
fish_target = [1] * 35 + [0] * 14  # 1 for Bream, 0 for Smelt

# Train KNN model
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
