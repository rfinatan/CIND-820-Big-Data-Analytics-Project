#initialize dependencies

import pandas as pd

# --- SECTION 1: INITIAL ANALYSIS ---

# 1.1 - Fundamnental Data Analysis

#initial data import
occurence_raw = pd.read_csv('MARSISdb_MDOTW_VW_OCCURRENCE_PUBLIC.csv')
vessel_raw = pd.read_csv('MARSISdb_MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC.csv')

#merge the raw data sets
marsis_raw = pd.merge(
    occurence_raw,
    vessel_raw,
    how = 'inner',
    left_on = 'OccID',
    right_on = 'OccID')

#view structure of the dataset
marsis_raw.info()

#view attribute data types
marsis_raw.dtypes

#view structure of combined data frame using pandas
pd.DataFrame.describe(marsis_raw)

#create a dataframe of the datatypes in the dataset
marsis_data_types = marsis_raw.dtypes.to_frame('dtypes').reset_index()
marsis_data_types.rename(columns = {'index':'raw_attribute'}, inplace = True)

#compare identified data types to provided data dictionary
marsis_dd = pd.read_csv("MARSISdb-dd-processed.csv")
marsis_dd = marsis_dd[marsis_dd.table_name == 'MDOTW_VW_OCCURRENCE_PUBLIC']
data_type_comparison = pd.merge(
    marsis_data_types,
    marsis_dd,
    how = "inner",
    left_on = "raw_attribute",
    right_on = "column_name"
)

# --- SECTION 2: DATA CLEANING ---

# 2.1 - Basic Data Cleaning, Primitive Dimensionality Reduction

#remove attributes with no values
marsis_no_nas = marsis_raw.copy(deep=True)
marsis_no_nas.dropna(how='all', axis='columns', inplace=True)

#some columns are duplicated in English and French 
#only require the English attribute variants, remove the French 
marsis_eng_only = marsis_no_nas.copy(deep=True)
marsis_eng_only.drop(marsis_eng_only.filter(regex='DisplayFre').columns, axis=1, inplace=True)

#there remain many columns with high percentage of NAs in the record counts
#determine each column's percentage of NA records
nas_percentage = (marsis_eng_only.isna().mean().round(2) * 100).to_frame(name = 'percentage')
#determine the columns with high percentage of NAs, more specifically greater than 80% NAs
nas_80_percent = nas_percentage.loc[nas_percentage['percentage'] > 80]

#visualize distribution of NA percentages
import matplotlib.pyplot as plt
import numpy as np
plt.hist(nas_percentage['percentage'], bins = 20)
plt.xticks(np.arange(0, max(nas_percentage['percentage']), 5))
plt.title('Number of Attributes and Their % of NA Records')
plt.xlabel('Percentage of Attribute NA Records')
plt.ylabel('Number of Attributes')
plt.show()

#create a list of the attributes that have higher than 80% NA records
nas_80_percent.reset_index(inplace=True)
nas_80_percent_list = list(nas_80_percent['index'])

#remove the attributes with more than 80% NA records
marsis_no_nas_80 = marsis_eng_only.drop(columns = nas_80_percent_list)

#there are also columns that have NA values masked as NULL strings
#identify the columns with all NULL string records
marsis_nulls = marsis_no_nas_80.loc[: , ((marsis_no_nas_80 == 'Null').any())]
marsis_nulls = list(marsis_nulls.columns)

#remove the columns with all NULL string records
marsis_basic_cleaned = marsis_no_nas_80.drop(columns = marsis_nulls)

# 2.2 - Contextual Data Cleaning

#given the context of the study, the area of focus is occurrences resulting in deaths
marsis_processed = marsis_basic_cleaned.copy(deep=True)

#many occurrences ending in a death have the same OccID and OccNo
#this is because each individual phase of the recorded occurrence is filed under the same OccID and OccNo
#to identify the unique cases of deadly occurrences, duplicate OccIDs must be dropped, and only the most recent is kept
#the most recent OccID is kept because it houses the final investigative data leading to the occurrence's result

#all OccDate instances have a 12:00:00 timestamp applied to them, whether actual or not
#strip timestamp from OccDate attribute records
marsis_processed['OccDate'] = marsis_processed['OccDate'].str.replace(' 12:00:00 AM', '')

#some OccTime instances have NA values; assume 00:00:00 timestamp given lack of information
marsis_processed['OccTime'] = marsis_processed['OccTime'].fillna('00:00:00')

#concatenate date and time attributes
marsis_processed['OccDateTime'] = marsis_processed['OccDate'] + ' ' + marsis_processed['OccTime']

#ensure date attributes are of dtype 'datetime'
marsis_processed['OccDateTime'] = pd.to_datetime(marsis_processed['OccDateTime'])

#drop duplicate OccId instances, but keep latest OccId instance
marsis_processed_unique = marsis_processed.sort_values('OccDateTime').drop_duplicates('OccID', keep = 'last')

# 2.2.1 - Removing Duplicate Attributes and Keeping Integer Encoding

#some attributes are duplicates of other attributes, but only exist for data classification purposes
#the attributes under study already have been encoded via integer encoding
#remove attributes that have an equivalent attribute, but keep the attribute that has integer encoding

#OccID and OccNo represent the same occurrence, but only one is necessary for classification
marsis_processed = marsis_processed_unique.drop(columns = 'OccNo')

#OccClassID and OccClassDisplayEng are identical, but only OccClassID is integer encoded
marsis_processed = marsis_processed.drop(columns = 'OccClassDisplayEng')

#the same duplicate attribute (integer encoding vs non-encoded) scenario applies to all attributes ending in '[...]DisplayEng'
marsis_processed.drop(marsis_processed.filter(regex='DisplayEng').columns, axis=1, inplace=True)

# 2.2.2 - Removing Attributes Unfit for Encoding

#the dataset contains a set of attributes too varied or complex in their contents to be encoded
#remove attributes unfit for encoding 
unfit_for_encoding = ['IICName', 'Summary', 'NearestLocationDescription', 'OccDate', 'OccTime', 'OccDateTime', 'WindDirection']

marsis_processed = marsis_processed.drop(columns = unfit_for_encoding)

# 2.2.3 - Removing Attributes with Low Variance

#determine variance of records per column in dataset
attribute_variance = marsis_processed.var(numeric_only = True).to_frame()

#remove low variance attributes from dataset
low_variance_attributes = ['IncludedInDailyEnum', 'MajorChangesIncludedInDaily', 'LatEnum', 'LongEnum']

marsis_processed = marsis_processed.drop(columns = low_variance_attributes)

# 2.2.4 - Removing Attributes Irrelevant to Hypothesis

#the dataset contains a set of attributes that are administrative, indicating when the report was released, filed, closed, and reported on by the Ministry of Transportation
#these attributes would not be useful in predicting a test dataset because they only serve for reporting or record identification purposes
#remove administrative attributes
administrative_attributes = ['ReleasedDate', 'OccClosedDate', 'EntryDate', 'ReportedDate', 'ReportedByID', 'OccID', 'TimeZoneID']

marsis_processed = marsis_processed.drop(columns = administrative_attributes)


# --- SECTION 3: Target Attribute Creation and Building Train, Validation, Testing Sets

# 3.1 - Target Variable Creation

#to create a target variable, determine the number of levels required for fatality classification

#store distribution of TotalDeaths
marsis_deadly = marsis_processed[marsis_processed['TotalDeaths'] > 0]

#the range of deaths is between 0, and 84
#most deaths are distributed between 0 and 5
plt.hist(marsis_deadly['TotalDeaths'], bins = 100)
plt.xticks(np.arange(0, max(marsis_deadly['TotalDeaths']), 1))
plt.title('Distribution of TotalDeaths')
plt.xlabel('Number of Deaths per Occurrence')
plt.ylabel('Number of Occurrences')
plt.xlim(0, 10)
plt.show()

#the average death count for fatal occurrences is 1.45 deaths
import statistics
statistics.mean(marsis_deadly['TotalDeaths'])

#since the number of average deaths is less than 2, the target variable death classifier will be a binary Yes or No
#add the target variable OccDeathClassID with the classification 1 = yes, 0 = no

marsis_processed['OccDeathClassID'] = np.where(marsis_processed['TotalDeaths']!= 0, 1, 0)

# 3.2 - Data Splitting

import numpy as np
from sklearn.model_selection import train_test_split

#create input array (excluding target variable and its direct attribute dependents)
x = marsis_processed.loc[:, ~marsis_processed.columns.isin(['OccDeathClassID', 'TotalDeaths', 'OccClassID', 'ImoClassLevelID'])]

#create output array (including target variable)
y = marsis_processed['OccDeathClassID'].to_frame()

#split input and output subsets with sklearn.model_selection.train_test_split() function call
#NOTE: the processed dataset is imbalanced as it contains a significant number of records that belong to the OccDeathClassID 0 class
#NOTE: recommended to stratify splits to have the same ratio of OccDeathClassID class records in training and testing sets

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size = 0.7, 
    test_size = 0.3, 
    #random_state = 4,
    stratify = y)

#verify proportionality of stratification in output array
y_train.dtypes
stratified_y_train = len(y_train[y_train['OccDeathClassID'] == 1]) / len(y_train)
print(stratified_y_train)

stratified_y_test = len(y_test[y_test['OccDeathClassID'] == 1]) / len(y_test)
print(stratified_y_test)

# --- SECTION 4: FORMAL DIMENSIONALITY REDUCTION ---

import xgboost
import numpy as np
import seaborn as sns
import sklearn.model_selection
from matplotlib import pyplot
from math import sqrt

# 4.1 - Removing Attributes with High Correlations

#verify if there are attributes that are too closely correlated and are dependents of the target variable
attribute_correlation = marsis_processed.corr()
sns.heatmap(marsis_processed.corr(), fmt='.2g',cmap= 'coolwarm')

#only two attributes are highly correlated: 'OccClassID' and 'TotalDeaths', which have already been removed from the training and test sets

# 4.2 - Dimensionality Reduction by Random Forests Ensemble

#given the dataset is very unbalanced between occurrences that have deaths and those that do not, create a parameter that accounts for imbalance
#with XGBRFClassifier(), pass scale_pos_weight = x, where x is the total_negative_examples / total_positive_examples

#use tweaking param to adjust scale_pos_weight impact on False Positive and False Negative results
tweaking_param = 0.45
estimate = ((len(y_train[y_train['OccDeathClassID'] == 0])) / (len(y_train[y_train['OccDeathClassID'] == 1]))) * tweaking_param

#define the random forest ensembles model
heuristic_parameter = sqrt(len(marsis_processed.columns)) / len(marsis_processed.columns)
model = xgboost.XGBRFClassifier(n_estimators = 100, subsample = 0.8, colsample_bynode = heuristic_parameter, scale_pos_weight = estimate)

#evaluate the model using repeated k-fold cross validation, with three repeats and 10 folds
cv = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#evaluate the model and collect the scores
n_scores = sklearn.model_selection.cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

#evaluate model accuracy by taking the mean of the cross validation scores, recording its standard deviation
print('Mean Accuracy: %.2f (%.2f)' % (np.mean(n_scores), np.std(n_scores)))

#fit the random forests model to the training data
model.fit(x_train, y_train)

#store feature importance
feature_important = model.get_booster().get_score(importance_type='gain')
keys = list(feature_important.keys())
values = list(feature_important.values())

feature_importance_df = pd.DataFrame(data=values, index=keys, columns=["score"])

#visualize feature importance
xgboost.plot_importance(model, title = 'Top 15 Most Important Features in Fatality Occurrences', max_num_features = 15)
pyplot.show()

# --- SECTION 5: MODEL BUILDING AND MODEL COMPARISONS ---

# 5.1 - Continuation of Random Forests Ensemble, Model Building and Evaluation

#make predictions for test data and evaluate
from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#test and pred results
#accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

results_comparison = x_test.assign(target = y_test.values, prediction = y_pred)

#visualize test and pred results
disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels= ['Not Deadly', 'Deadly'])
disp.ax_.set_title('Confusion Matrix for Random Forests')
plt.show()

#visualize ROC AUC
fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
auc = sklearn.metrics.roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.suptitle('ROC Curves for Weighted Random Forest')
plt.show()

#save results for iterative plotting
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

result_table = result_table.append({'classifiers':'Weighted Random Forest',
                                    'fpr':fpr, 
                                    'tpr':tpr, 
                                    'auc':auc}, ignore_index=True)

# 5.1.1 - Redefine the Model's Optimal Weighting using Grid Search
from sklearn.model_selection import GridSearchCV

#define grid weights and parameters
weights = [0.1, 0.25, 0.5, 0.75, 1, 10, 25, 50, 75, 100]
param_grid = dict(scale_pos_weight=weights)

#evaluate the model using repeated k-fold cross validation, with three repeats and 10 folds
cv = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#define grid search execution
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
grid_result = grid.fit(x_train, y_train)

#report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
#redefine the model given best defined weighting
model = xgboost.XGBRFClassifier(n_estimators = 100, subsample = 0.8, scale_pos_weight = (list(grid_result.best_params_.values())[0]))

#fit the random forests model to the training data
model.fit(x_train, y_train)

#make predictions for test data and evaluate
from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#test and pred results
#accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

results_comparison = x_test.assign(target = y_test.values, prediction = y_pred)

#visualize test and pred results
disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels= ['Not Deadly', 'Deadly'])
disp.ax_.set_title('Confusion Matrix for Weight Optimized Random Forests')
plt.show()

#visualize ROC AUC
fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
auc = sklearn.metrics.roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.suptitle('ROC Curves for Weight Optimized Random Forest')
plt.show()

#save results for iterative plotting
result_table = result_table.append({'classifiers':'Weight Optimized Random Forest',
                                    'fpr':fpr, 
                                    'tpr':tpr, 
                                    'auc':auc}, ignore_index=True)

# 5.1.2 - Balanced Random Forests Algorithm
##since imbalanced-learn models cannot handle NANs like XGBOOST, use an imputation method for NANs
learning_sets = [x_train, y_train]
for set in learning_sets:
    set.apply(pd.to_numeric, errors = 'coerce')
test_sets = [x_test, y_test]
for set in test_sets:
    set.apply(pd.to_numeric, errors = 'coerce')

imp_x_train = x_train.copy(deep=True)
imp_x_train.fillna(imp_x_train.mean(), inplace = True)

imp_x_test = x_test.copy(deep=True)
imp_x_test.fillna(imp_x_test.mean(), inplace = True)

imp_y_train = y_train.copy(deep=True)
imp_y_train.fillna(imp_y_train.mean(), inplace = True)
imp_y_train = np.ravel(imp_y_train)

imp_y_test = y_test.copy(deep=True)
imp_y_test.fillna(imp_y_test.mean(), inplace = True)
imp_y_test = np.ravel(imp_y_test)

from imblearn.ensemble import BalancedRandomForestClassifier

balanced_model = BalancedRandomForestClassifier(n_estimators=10)

#evaluate the model using repeated k-fold cross validation, with three repeats and 10 folds
cv = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#evaluate the model and collect the scores
scores = sklearn.model_selection.cross_val_score(model, imp_x_train, imp_y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % np.mean(scores))

#fit the random forests model to the training data
balanced_model.fit(imp_x_train, imp_y_train)

#make predictions for test data and evaluate
y_balanced_pred = balanced_model.predict(imp_x_test)
balanced_accuracy = accuracy_score(imp_y_test, y_balanced_pred)
print("Accuracy: %.2f%%" % (balanced_accuracy * 100.0))

#test and pred results
#accuracy metrics
from sklearn.metrics import accuracy_score
print(accuracy_score(imp_y_test, y_balanced_pred))

#classification report
from sklearn.metrics import classification_report
print(classification_report(imp_y_test, y_balanced_pred))

#visualize test and pred results
disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
    imp_y_test,
    y_balanced_pred,
    display_labels= ['Not Deadly', 'Deadly'])
disp.ax_.set_title('Confusion Matrix for Balanced Random Forest')
plt.show()

#visualize ROC AUC
fpr, tpr, _ = sklearn.metrics.roc_curve(imp_y_test, y_balanced_pred)
auc = sklearn.metrics.roc_auc_score(imp_y_test, y_balanced_pred)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.suptitle('ROC Curves for Balanced Random Forest')
plt.show()

#save results for iterative plotting
result_table = result_table.append({'classifiers':'Balanced Random Forest',
                                    'fpr':fpr, 
                                    'tpr':tpr, 
                                    'auc':auc}, ignore_index=True)

#plot comparison figure
for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(result_table.loc[i]['classifiers'], result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='grey', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('ROC Curve Analysis')
plt.legend(loc=4)
plt.show()

# 5.2 - Logistic Regression, Model Building and Evaluation

from sklearn.linear_model import LogisticRegression

#since sklearn models cannot handle NANs like XGBOOST, use the imputed variables already created to deal with NANs
#define the logistic regression model
#account for unbalanced dataset as in XGBOOST using class_weight hyperparameter
w = {0:2.1361, 1:97.8639}
log_model = LogisticRegression(random_state=4, class_weight=w, max_iter=10000)

#optional: hide ConvergenceWarnings for logistic regression output
import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)

#evaluate the model using repeated k-fold cross validation, with three repeats and 10 folds
cv = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#evaluate the model and collect the scores
n_log_scores = sklearn.model_selection.cross_val_score(log_model, imp_x_train, imp_y_train, scoring='accuracy', cv=cv, n_jobs=-1)

# report performance
print('Mean Accuracy: %.2f (%.2f)' % (np.mean(n_log_scores), np.std(n_log_scores)))

#fit the model to the training data
log_model.fit(imp_x_train, imp_y_train)

#make predictions for test data and evaluate
y_log_pred = log_model.predict(imp_x_test)
log_accuracy = accuracy_score(imp_y_test, y_log_pred)
print("Accuracy: %.2f%%" % (log_accuracy * 100.0))

#test and pred results
#accuracy metrics
from sklearn.metrics import accuracy_score
print(accuracy_score(imp_y_test, y_log_pred))

#classification report
from sklearn.metrics import classification_report
print(classification_report(imp_y_test, y_log_pred))

#visualize test and pred results
disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
    imp_y_test,
    y_log_pred,
    display_labels= ['Not Deadly', 'Deadly'])
disp.ax_.set_title('Confusion Matrix for Weighted Logistic Regression')
plt.show()

#visualize ROC AUC
fpr, tpr, _ = sklearn.metrics.roc_curve(imp_y_test, y_log_pred)
auc = sklearn.metrics.roc_auc_score(imp_y_test, y_log_pred)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.suptitle('ROC Curves for Weighted Logistic Regression')
plt.show()

#save results for iterative plotting
result_table = result_table.append({'classifiers':'Weighted Logistic Regression',
                                    'fpr':fpr, 
                                    'tpr':tpr, 
                                    'auc':auc}, ignore_index=True)

#logistic regression variable importance
log_importance = pd.Series(log_model.coef_[0], index = imp_x_test.columns)
log_importance = log_importance.sort_values(ascending=False)
log_importance.nlargest(15).plot(kind = 'barh', title = 'Weighted Logistic Regression Feature Importance')

# 5.3 - Naive Bayes Classifier, Model Building and Evaluation

from sklearn.naive_bayes import GaussianNB

#define the naive bayes model
nb_model = GaussianNB()

#evaluate the model using repeated k-fold cross validation, with three repeats and 10 folds
cv = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#evaluate the model and collect the scores
n_nb_scores = sklearn.model_selection.cross_val_score(nb_model, imp_x_train, imp_y_train, scoring='accuracy', cv=cv, n_jobs=-1)

# report performance
print('Mean Accuracy: %.2f (%.2f)' % (np.mean(n_nb_scores), np.std(n_nb_scores)))

#fit the model to the training data
nb_model.fit(imp_x_train, imp_y_train)

#make predictions for test data and evaluate
y_nb_pred = nb_model.predict(imp_x_test)
nb_accuracy = accuracy_score(imp_y_test, y_nb_pred)
print("Accuracy: %.2f%%" % (nb_accuracy * 100.0))

#test and pred results
from sklearn.metrics import classification_report

print(classification_report(imp_y_test, y_nb_pred))

#visualize test and pred results
disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
    imp_y_test,
    y_nb_pred,
    display_labels= ['Not Deadly', 'Deadly'])
disp.ax_.set_title('Confusion Matrix for Naive Bayes')
plt.show()

#visualize ROC AUC
fpr, tpr, _ = sklearn.metrics.roc_curve(imp_y_test, y_nb_pred)
auc = sklearn.metrics.roc_auc_score(imp_y_test, y_nb_pred)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.suptitle('ROC Curves for Naive Bayes')
plt.show()

#save results for iterative plotting
result_table = result_table.append({'classifiers':'Naive Bayes',
                                    'fpr':fpr, 
                                    'tpr':tpr, 
                                    'auc':auc}, ignore_index=True)

# 5.4 - Consolidated Evaluation of Models

#plot comparison figure
for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(result_table.loc[i]['classifiers'], result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='grey', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('ROC Curve Analysis')
plt.legend(loc=4)
plt.show()

#store validation results in variables
results = []
results.append(n_scores)
results.append(n_log_scores)
results.append(n_nb_scores)

model_names = ['Random Forests', 'Logistic Regression', 'Naive Bayes']

#plot results to boxplot
fig = plt.figure()
fig.suptitle('Comparison of Algorithm Accuracy')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(model_names)
plt.show()

