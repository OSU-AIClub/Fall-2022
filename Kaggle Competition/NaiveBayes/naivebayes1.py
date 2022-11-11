""" Imports """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dataset import X_train, X_val, y_train, y_val

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# Process Data
le = LabelEncoder()
y_train = le.fit_transform(y_train)
le_val = LabelEncoder()
y_val = le_val.fit_transform(y_val)

vec = CountVectorizer()
vec = vec.fit(X_train)
train_x_bow = vec.transform(X_train)
val_x_bow = vec.transform(X_val)

""" Train """
# Initialize Model and Fit it to the Training Data
classifier = MultinomialNB()

alpha_ranges = {'alpha': [0.001, 0.01, 0.1, 1, 10.0, 100]}
grid_search = GridSearchCV(classifier, param_grid=alpha_ranges, scoring='accuracy', cv=3, return_train_score=True)
grid_search.fit(train_x_bow, y_train)

alpha = [0.001, 0.01, 0.1, 1, 10.0, 100]
train_acc = grid_search.cv_results_['mean_train_score']
train_std = grid_search.cv_results_['std_train_score']

test_acc = grid_search.cv_results_['mean_test_score']
test_std = grid_search.cv_results_['std_test_score']

plt.plot(alpha, train_acc, label="Training Score", color='b')
plt.plot(alpha, test_acc, label="Cross Validation Score", color='r')

plt.title("Validation Curve with Naive Bayes Classifier")
plt.xlabel("Alpha")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.legend(loc = 'best')
plt.show()

print(grid_search.best_estimator_)

classifier = MultinomialNB(alpha=1)
classifier.fit(train_x_bow, y_train)

predict = classifier.predict(val_x_bow)

print("Accuracy is ", accuracy_score(y_val, predict))
print(classification_report(y_val, predict))


""" Inference on Test Dataset"""
#! TO BE IMPLEMENTED