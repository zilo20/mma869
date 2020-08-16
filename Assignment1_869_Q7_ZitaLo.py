# Name: Zita Lo
# Student Number: 20196119
# Program: MMA
# Cohort: Winter 2021
# Course Number: MMA 869
# Date: August 16, 2020


# Answer to Question 7 Task 2


# Import packages
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Allow better display for multiline code
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Read in data
df=pd.read_csv('OJ.csv',index_col=[0])

# # -----------------------
# # Data Exploration
# # -----------------------
# Generate a Profile Report on the df to better understand the data. ONLY REQUIRE TO RUN ONCE

#profile = ProfileReport(df)
#profile.to_file('Orange Juice Profiling Report.html')

# Understand the data, features and class types: 1070 instances and 18 features
df.head()
df.info()
df.describe()

# Understand how many features there are, and which are categorical vs numeric
n_features = df.shape[1]

cat_col_names = list(df.select_dtypes(include=np.object).columns)
num_col_names = list(df.select_dtypes(include=np.number).columns)

print('cat_col_names: {}'.format(cat_col_names))
print('num_col_names: {}'.format(num_col_names))

# Descriptive stats of the categorical features: 2 categorical features
df.describe(include=[np.object]).transpose()

# Descriptive stats of the numeric features: 16 numeric features
df.describe(include=[np.number]).transpose()

# Draw a correlation plot using seaborn heatmap. Understand the correlation between features
figure = plt.figure(figsize=(10, 8))
sns.heatmap(df.corr())
plt.tight_layout()

# Observation: Lots of highly correlated features found in the data. Need to take a closer look on some of those

# Display the number values on the correlation pairs
df.corr()

# Observations - For example these pairs have really high corr values: 
# 0.998793 DISCMM vs PctDiscMM
# 0.999022 DiscCH vs PctDiscCH
#-0.823908 DISCMM vs PriceDiff
# 0.846868 DISCMM vs SalePriceMM
# Will drop some of these features in the preprocessing data section

# # -------------------------------
# # Q7 Task 2a - Preprocess Data
# # -------------------------------

# Convert StoreID class type from interger to object
df['StoreID'] = df['StoreID'].astype('object')

# Review and compare the three features related to 'store' before consolidating
# The plan is to make dummies for each storeID
df['StoreID'].value_counts()
df['STORE'].value_counts()
df['Store7'].value_counts()

# Week of Purchase - Minimum value is 227. Deduct 226 from all values to make the value start from one
# The values then range from 1 to 52. A full year data

df['WkofPurchase'] = df['WeekofPurchase']-226
df.drop('WeekofPurchase', axis=1, inplace=True)

# One Hot encoding: Encode the value to four quarters instead of displaying them only from 1-52
df['Q1'] = np.where((df['WkofPurchase'] <=  13),1,0)
df['Q2'] = np.where((df['WkofPurchase'] >= 14) & (df['WkofPurchase'] <= 26),1,0)
df['Q3'] = np.where((df['WkofPurchase'] >= 27) & (df['WkofPurchase'] <= 39),1,0)
df['Q4'] = np.where((df['WkofPurchase'] >= 40) & (df['WkofPurchase'] <= 52),1,0)

# Review the changes and ensure they are correct. Choose random instances to view
df.sample(n=20, random_state=101).head(20)

# Create dummies on the StoreID and Purchase features
df = pd.get_dummies(df,columns=['StoreID','Purchase'], prefix=['StoreID','Purchase'])

# View results
df.head()

# Drop features e.g. 'Store7','STORE','WkofPurchase' that have same info as other features
# Drop feature 'Purchase_MM','StoreID_4','Q2' to avoid collinearity after creating dummies. 'Purchase_CH' will be the target
# Drop features that have high collinearity with other features and features that are displaying same info as other features
# e.g. 'PriceCH' &'PriceMM' vs 'ListPriceDiff' - 'ListPriceDiff' are the differences of 'PriceCH' &'PriceMM'. Thus 'PriceCH' &'PriceMM' can be removed 
# e.g. 'SalePriceCH' &'SalePriceMM' vs 'PriceDiff' - 'PriceDiff' are the differences of 'SalePriceCH' &'SalePriceMM'. Thus 'SalePriceCH' &'SalePriceMM' can be removed
# e.g. 'PctDiscCH' & PctDiscMM' vs 'DiscCH' &'DiscMM' - These pairs are presenting in diff way but same info. One pair can be removed

df.drop(['Store7','STORE','Purchase_MM','PriceCH','PriceMM','SalePriceCH','SalePriceMM','PctDiscCH','PctDiscMM','StoreID_4','WkofPurchase','Q2'], axis=1, inplace=True)

# Plot correlation heatmap again after the data preprocessing to review

figure = plt.figure(figsize=(10, 8))
sns.heatmap(df.corr())
plt.tight_layout()

# Plot correlation values again to review
df.corr()

# Check if any missing values in the data

plt.figure(figsize=(20,6))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.tight_layout()


df.isnull().sum()
# observation: no missing value


# Display the pre-processed data and info before moving onto the next stage
# Dataframe has 15 features and 1070 instances. Target feature is 'Purchase_CH'
df.head()
df.info()
df.describe()

# # -----------------------
# # Standardized Data
# # -----------------------
# Standardize data before splitting into training and testing set
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler() object called scaler
scaler = StandardScaler()

# Fit scaler to the features except the target feature 'Purchase_CH'
scaler.fit(df.drop('Purchase_CH',axis=1))

# Use the .transform() method to transform the features to a scaled version
scaled_features = scaler.transform(df.drop('Purchase_CH',axis=1))

# View the scaled data with header
x = pd.DataFrame(scaled_features,columns=df.columns[:14])
x.head()

# # -------------------------------
# # Q7 Task 2b - Split Train and Test 
# # -------------------------------

# Split data into training and testing set 
from sklearn.model_selection import train_test_split

# y is the target
y = df['Purchase_CH']

# Test vs train ratio is 0.25: 0.75. Random state is set to 101
# Set up stratify parameter will preserve the proportion of target as in original dataset, in the train and test datasets as well
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=101)

# Print counts
# Y (train) counts: 1 (489); 0 (313)
print('Y (train) counts:')
print(y_train.value_counts())

# Y (test) counts: 1 (164); 0 (104)
print('Y (test) counts:')
print(y_test.value_counts())
    

# # -------------------------------
# # Q7 Task 2c - Build Models
# # -------------------------------

# To begin, train data and evaluate with different models/ensembles
# Return a list of the performance of all the models and identify the top three to work with

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting  

# import performance metrics packages
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

import time

# load function
def do_all_for_dataset(X_train, X_test, y_train, y_test):

    
    nb = GaussianNB()   
    lr = LogisticRegression(random_state=101, solver='lbfgs', max_iter=5000)
    dt = DecisionTreeClassifier(random_state=101)
    knn = KNeighborsClassifier(n_neighbors=7)
    rf = RandomForestClassifier(random_state=101, n_estimators=200)
    ada = AdaBoostClassifier(random_state=101, n_estimators=200)


    est_list = [('DT', dt), ('LR', lr), ('NB', nb), ('RF', rf), ('ADA', ada)]
       
    dict_classifiers = {
        "LR": lr, 
        "NB": nb,
        "DT": dt,
        "KNN": knn,
        "Voting": VotingClassifier(estimators = est_list, voting='soft'),
        "Bagging": BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=101),
        "RF": rf,
        "ExtraTrees": ExtraTreesClassifier(random_state=101, n_estimators=200),
        "Adaboost": ada,
        "GBC": GradientBoostingClassifier(random_state=101, n_estimators=200),
        "Stacking": StackingClassifier(estimators=est_list, final_estimator=LogisticRegression()),
    }
    
    model_results = list()
    
    for model_name, model in dict_classifiers.items():
        start = time.time()
        y_pred = model.fit(X_train, y_train).predict(X_test)
        end = time.time()
        total = end - start
        
        accuracy       = accuracy_score(y_test, y_pred)
        f1             = f1_score(y_test, y_pred)
        recall         = recall_score(y_test, y_pred)
        precision      = precision_score(y_test, y_pred)
        roc_auc        = roc_auc_score(y_test, y_pred)
    
        df = pd.DataFrame({"Method"    : [model_name],
                           "Time"      : [total],
                           "Accuracy"  : [accuracy],
                           "Recall"    : [recall],
                           "Precision" : [precision],
                           "F1"        : [f1],
                           "AUC"       : [roc_auc],
                          })
        model_results.append(df)
   

    dataset_results = pd.concat([m for m in model_results], axis = 0).reset_index()

    dataset_results = dataset_results.drop(columns = "index",axis =1)
    # Evaluate based on accuracy and F1 score. For preliminary, sort by accuracy
    dataset_results = dataset_results.sort_values(by=['Accuracy'], ascending=False)
    dataset_results['Rank'] = range(1, len(dataset_results)+1)
    
    return dataset_results

# Save the results of each dataset into a list
results = list()

# Call function with input parameters define from the split training and testing section
r = do_all_for_dataset(X_train, X_test, y_train, y_test)

# Append results into the list
results.append(r)

# View result list
r

# Uncomment this line to export the result list to csv
# r.to_csv('rank.csv')

# Result: LR, Adaboost and GBC are the top 3 models
# Accuracy: LR 0.854478; Adaboost 0.839552; GBC 0.835821
# F1: LR 0.883582; Adaboost 0.870871; GBC 0.865854
# Work on these three models next to fine tune

# # -----------------------
# # Pre-define functions for plots
# # -----------------------

from sklearn.metrics import roc_curve, auc, classification_report, cohen_kappa_score,log_loss
def plot_roc(clf, X_test, y_test, name, ax, show_thresholds=True):
    y_pred_rf = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, y_pred_rf)

    ax.plot([0, 1], [0, 1], 'k--');
    ax.plot(fpr, tpr, label='{}, AUC={:.2f}'.format(name, auc(fpr, tpr)));
    ax.scatter(fpr, tpr);

    if show_thresholds:
        for i, th in enumerate(thr):
            ax.text(x=fpr[i], y=tpr[i], s="{:.2f}".format(th), fontsize=10, 
                     horizontalalignment='left', verticalalignment='top', color='black',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', alpha=0.1));
        
    ax.set_xlabel('False positive rate', fontsize=18);
    ax.set_ylabel('True positive rate', fontsize=18);
    ax.tick_params(axis='both', which='major', labelsize=12);
    ax.grid(True);
    ax.set_title('ROC Curve', fontsize=18)

# # -----------------------
# # Logistic Regression
# # -----------------------
from sklearn.linear_model import LogisticRegression

# Parameters identified from GridSearchCV (in the Hyperparameter Tuning section)
# With this setting, the accuracy (0.83 instead of 0.85) and F1 score dropped slightly
# lr_clf = LogisticRegression(random_state=101, class_weight='balanced', C = 1, solver='liblinear',max_iter=5000)

# Apply this set of parameters from Hyperparameter Tuning (without class_weight='balanced' and generated this set)
lr_clf = LogisticRegression(random_state=101, C = 1, solver='lbfgs')

lr_clf.fit(X_train, y_train)


# Predict using LR model
y_pred_lr = lr_clf.predict(X_test)

# LR Model performance
# Print confusion matrix
print(confusion_matrix(y_test, y_pred_lr))

# Print classification report
print(classification_report(y_test, y_pred_lr))

# Print accuracy, kappa, F1 and log loss
print("Accuracy = {:.2f}".format(accuracy_score(y_test, y_pred_lr)))
print("Kappa = {:.2f}".format(cohen_kappa_score(y_test, y_pred_lr)))
print("F1 Score = {:.2f}".format(f1_score(y_test, y_pred_lr)))
print("Log Loss = {:.2f}".format(log_loss(y_test, y_pred_lr)))

# Plot ROC curve for LR
plt.style.use('default')
figure = plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 1, 1)
plot_roc(lr_clf, X_test, y_test, "Logistic Regression", ax)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()

# Interpreting LR model
# View the coefficient and intercept

feat = df.columns[:14]
df_lrcoef = pd.DataFrame(lr_clf.coef_).transpose()
df_lrcoef.columns =['coefficient'] 
df_feat = pd.DataFrame(feat,columns=['features'])

df_lrcoef = pd.concat([df_feat,df_lrcoef],axis=1)
df_lrcoef

# LR model intercept: 0.82907609
lr_clf.intercept_

# Create a Dataframe that has the test data and the predicted value
feat_name = list(df.columns[:14].values)
df_X_test = pd.DataFrame(scaler.inverse_transform(X_test[:]),columns = feat_name)
df_y_pred = pd.DataFrame(y_pred_lr, columns=['Pred_Purchase_CH'])

df_lr = pd.concat([df_X_test,df_y_pred],axis=1)
df_lr

# Export the result to csv. Uncomment this line to export file
# df_lr.to_csv('LR_prediction.csv')

# # -----------------------
# # AdaBoost
# # -----------------------

# Parameters identified from GridSearchCV (in the Tuning section)
ada_clf = AdaBoostClassifier(random_state=101, n_estimators=100)

ada_clf.fit(X_train, y_train)

# Predict using AdaBoost model
# Print confusion matrix
y_pred_ada = ada_clf.predict(X_test)

# AdaBoost model performance
print(confusion_matrix(y_test, y_pred_ada))

# Print classification report
print(classification_report(y_test, y_pred_ada))

# Print accuracy, kappa, F1 and log loss
print("Accuracy = {:.2f}".format(accuracy_score(y_test, y_pred_ada)))
print("Kappa = {:.2f}".format(cohen_kappa_score(y_test, y_pred_ada)))
print("F1 Score = {:.2f}".format(f1_score(y_test, y_pred_ada)))
print("Log Loss = {:.2f}".format(log_loss(y_test, y_pred_ada)))

# Plot ROC curve for AdaBoost
plt.style.use('default')
figure = plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 1, 1)
plot_roc(ada_clf, X_test, y_test, "AdaBoost", ax)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()

# # -----------------------
# # GradientBoostingClassifier
# # -----------------------
# Parameters identified from GridSearchCV (in the Tuning section) but result turns out 0.01 less accurate with this set
#gbc_clf = GradientBoostingClassifier(random_state=101, n_estimators=100, max_features='log2')

# Removed max_features from GridSearchCV and this is another set of best parameter
gbc_clf = GradientBoostingClassifier(random_state=101, n_estimators=50)
gbc_clf.fit(X_train, y_train)

# Predict using GradientBoostingClassifier model
y_pred_gbc = gbc_clf.predict(X_test)

# GBC model performance
# Print confusion matrix
print(confusion_matrix(y_test, y_pred_gbc))

# Print classification report
print(classification_report(y_test, y_pred_gbc))

# Print accuracy, kappa, F1 and log loss
print("Accuracy = {:.2f}".format(accuracy_score(y_test, y_pred_gbc)))
print("Kappa = {:.2f}".format(cohen_kappa_score(y_test, y_pred_gbc)))
print("F1 Score = {:.2f}".format(f1_score(y_test, y_pred_gbc)))
print("Log Loss = {:.2f}".format(log_loss(y_test, y_pred_gbc)))

# Plot ROC curve for GBC
plt.style.use('default')
figure = plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 1, 1)
plot_roc(gbc_clf, X_test, y_test, "GradientBoostingClassifier", ax)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()

# # -------------------------------
# # Hyperparameter Tuning
# # -------------------------------

# Tuning using GridSearchCV to find the optimal parameters. Choosing the best values for hyperparameters for the three models
from sklearn.model_selection import GridSearchCV

# Logistic Regression Hyperparameter Tuning
param_grid = {'C':[1,10,100,1000,10000],'solver': ['lbfgs', 'liblinear']} 
grid = GridSearchCV(LogisticRegression(),param_grid,verbose=3)

grid.fit(X_train, y_train)
# Result: Done  50 out of  50 | elapsed:    0.2s finished

grid.best_params_
# Result:  {'C': 1, 'solver': 'lbfgs'}

grid.best_estimator_
# Result: LogisticRegression(C=1)

# # -----------------------

# AdaBoost Hyperparameter Tuning
param_grid = {'n_estimators':[50,100,200,400,800,1000,2000]} 
grid_ada = GridSearchCV(AdaBoostClassifier(),param_grid,verbose=3)

grid_ada.fit(X_train, y_train)
# Result: Done  35 out of  35 | elapsed:   29.5s finished

grid_ada.best_params_
# Result: {'n_estimators': 100}

grid_ada.best_estimator_
# Result: AdaBoostClassifier(n_estimators=100)

# # -----------------------

# GradientBoostClassifier Hyperparameter Tuning
param_grid_gbc = {'n_estimators':[50,100,200,400,800,1000,2000],'loss' : ['deviance', 'exponential'] } 
grid_gbc = GridSearchCV(GradientBoostingClassifier(),param_grid_gbc,verbose=3)

grid_gbc.fit(X_train, y_train)
# Result: Done  70 out of  70 | elapsed:   35.2s finished

grid_gbc.best_params_
# Result: {'loss': 'deviance', 'n_estimators': 50}

grid_gbc.best_estimator_
# Result: GradientBoostingClassifier(n_estimators=50)

