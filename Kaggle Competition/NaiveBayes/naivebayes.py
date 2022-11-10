""" Imports """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dataset import X_train, X_val, y_train, y_val, X_test, n_classes
from model import MultinomialNaiveBayes
from tokenizer5 import Tokenizer

""" Train """
# Initialize Model and Fit it to the Training Data
MNB = MultinomialNaiveBayes(
    classes=n_classes, 
    tokenizer=Tokenizer()
).fit(X_train, y_train)

# Predict sentiment on the validation dataset split
y_hat = MNB.predict(X_val)

""" Validation Metrics """
# Get accuracy on the validation split
print(f"Validation Accuracy: {accuracy_score(y_val, y_hat)}")

# More performance metrics
print(classification_report(y_val, y_hat))

# ## Prdouce Confusion Matrix to  Visualize Results
# To visualize our True Positive, True Negative, False Positive, and False Negative predictions, we will plot a confusion matrix using `seaborn` and `pyplot`.

# Produce confusion matrix of true positives, true negatives, false postives, and false negatives
cnf_matrix = confusion_matrix(y_val, y_hat)

# Create Plot
class_names = ["negative", "positive"]
fig,ax = plt.subplots()

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Blues", fmt="d", cbar=False, xticklabels=class_names, yticklabels=class_names)
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.ylabel('Actual sentiment')
plt.xlabel('Predicted sentiment')
plt.show()

""" Inference on Test Dataset"""

# Get input values and predict sentiment
predictions = MNB.predict(X_test)
submission = {'sentiment': predictions}

# Convert predictions to a Pandas Dataframe   
submission_df = pd.DataFrame(submission)
submission_df.index.name = 'id'

# Save predictions to a CSV file for submission
submission_df.to_csv('submission.csv')

