import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    accuracy_score
)
from yellowbrick.classifier import (
    ClassificationReport,
    DiscriminationThreshold,
)
import warnings
warnings.filterwarnings("ignore")


## X_train, X_test, y_train, y_test = train_test_split(
##    data.drop(labels=['target'], axis=1),  # drop the target
##    data['target'],  # just the target
##    test_size=0.3,
##    random_state=0)
    
## accuracy_score(y_test, y_test_base)
## accuracy_score(y_test, rf.predict(X_test))
## Percentage of the minority correctly classified

def return_minority_perc(y_true, y_pred):
    minority_total = np.sum(y_true)
    minority_correct = np.sum(np.where((y_true==1)&(y_pred==1),1,0))
    return minority_correct / minority_total *100
    
### return_minority_perc(y_test, y_test_base)

## Precision = tp / (tp + fp)
## Recall = tp / (tp + fn)
## F1 = 2 * (precision * recall) / (precision + recall)
## Support = Number of cases on each class

## precision_score(y_test, rf.predict(X_test))
## recall_score(y_test, rf.predict(X_test), pos_label=1)
## f1_score(y_test, rf.predict(X_test), pos_label=1)
 
## All metrics at once
precision, recall, fscore, support = precision_recall_fscore_support(
    y_test, rf.predict(X_test), pos_label=1,
)
print('Random Forests Precision: ', precision)
print('Random Forests Recall: ', recall)
print('Random Forests f-score: ', fscore)
print('Support: ', support)

## Full classification report
visualizer = ClassificationReport(rf)
visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()                       # Finalize and show the figure

## Precision and Recall, vs probability threshold

visualizer = DiscriminationThreshold(logit,
                                     n_trials=1,
                                     cv=0.5,
                                     argmax='fscore',
                                     random_state=0,
                                     is_fitted='auto',
                                     exclude = "queue_rate")

visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()

## Adding Extra changes

