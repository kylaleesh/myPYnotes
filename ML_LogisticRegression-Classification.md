# ML models - Supervised:
## Logistic Regression
* Purpose: Predicting the probability of a binary outcome.
* Output: Probability values between 0 and 1.
* Example: Predicting whether an email is spam or not (binary classification) based on features like email content, sender, etc.

### Data Processing
#### Dummy Encoding
```py
# One-hot encode them by 'get_dummies()' method
data = pd.get_dummies(adult, columns=['workclass', 'marital-status', 'occupation', 'relationship', 'race'], drop_first=True)
```
#### Check for Multi-collinearity
```py
#multicollinearity - correlation
data.corr()

# for easier eye-balling
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), xticklabels=True, annot = True, fmt = '.1f',yticklabels=True,cmap='RdYlGn')
```
#### VIF to remove var with high collinrearity
```py
from statsmodels.stats.outliers_influence import variance_inflation_factor
data_wo_income = data.drop(['income'],axis=1)
# calculating the VIF for each feature
vif_data = [variance_inflation_factor(data_wo_incom.values, i) for i in range(len(data_wo_incom.columns))]

VIF = []
for col, vif in zip(data_wo_incom.columns, vif_data):
    col_list = data_wo_incom
    VIF.append(vif)
    
vif_df = pd.DataFrame(data ={'Column name':data_wo_incom.columns , "VIF":VIF})
print(vif_df.sort_values(by= 'VIF', ascending = False))

# continue droppping var with  high VIF and do more rounds of VIF
```
### Model Fitting
#### Split data into train, test
```py
from sklearn.model_selection import train_test_split
X = data.drop(columns='y', axis=1)
y = data['y']

# Perform split for train:test ~ 70:30, here train contains (train + validation) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
```
#### Scale Data (numeric data)
```py  
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scale_cols = X_train.columns

X_train[scale_cols] = sc.fit_transform(X_train[scale_cols])
X_test[scale_cols] = sc.transform(X_test[scale_cols])
```
#### Fit into model (Sklearn) & evaluate
```py
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

# fit model
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# evaluate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, f1_score)
accuracy_score(y_test, y_pred)
```

#### Fit into model (statsmodels.api) & iterate
```py
import statsmodels.formula.api as sm
import statsmodels.api as sm_api

X_train = sm_api.add_constant(X_train)
X_test = sm_api.add_constant(X_test)

# iteration 1
model = sm_api.Logit(y_train, X_train)
result = model.fit()
result.summary()

# drop cols with high p-value and iterate again

# Predicted Probability Values and getting labels
y_pred = model.predict(params=result.params, exog=X_test)
y_pred_labels = (y_pred>0.5).astype(int) # Default (Random) Model threshold of 0.5

```

### Evaluate Regression Model - metrics & cross_val_score
```py
# cross validation score
from sklearn.model_selection import cross_val_score

log_reg = LogisticRegression()
scores = cross_val_score(log_reg, X_train2, y_train, cv=5, scoring='accuracy')
score.mean()

# Confusion Matrix and % of FP and FN
conf_mat = confusion_matrix(y_test, y_pred_labels)
print("% of False Positive: {}".format(conf_mat[0][1]*100/(conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])))
print("% of False Negative: {}".format(conf_mat[1][0]*100/(conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])))

# Roc-Auc curve :plotting TTrue positive rate (TPR) against False positive rate (FPR)
# intuition is to increase the TPR
fpr, tpr, _ = roc_curve(y_test, y_pred_labels)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
# ROC-AUC score:
roc_auc_score(y_test, y_pred_labels)

# Classification Report - precision, recall, f1-score, support
classification_report(y_test, y_pred_labels)

# and also...
# KS-chart, KS-statistic, (concordance, discordance, ties and Somers-D)
```
## RANDOM NOTE - Class Imbalances
```py
import imblearn
from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_under, y_under = undersampler.fit_resample(X, y)
Counter(y_under)

from imblearn.over_sampling import RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_over, y_over = oversampler.fit_resample(X, y)
Counter(y_over)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X,y)
Counter(y_smote)
```