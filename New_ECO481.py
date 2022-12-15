#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 23:36:35 2022

@author: jonathanchien
"""
#Read the file and do simple descriptive statistics
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("/Users/jonathanchien/Documents/ECO481H Datahub/gss_clean.csv")
data.columns = data.columns.str.replace('regilion_importance', 'religion_importance')
new = data[["caseid", "age", "total_children", "sex", "marital_status", "education", "hh_size", "hh_type", "partner_education", 
                "average_hours_worked", "partner_main_activity", "income_family", "occupation", "children_in_household",  
                "religion_participation", "living_arrangement", "vis_minority", "province"]]

new_data = new.dropna()
descriptive_data = new_data.describe()
descriptive_data

print(new_data["hh_type"].unique())
df=new_data.groupby(by="hh_type")
df
#do grouping
#first I need to drop the "Other" and "Don't know" values in the data table
new_data.drop(new_data.loc[new_data['hh_type']=='Other'].index, inplace=True)
new_data.drop(new_data.loc[new_data['hh_type']=="Don't know"].index, inplace=True)
#after dropping the two values, then we rename the low-rise and high-rise apartment to apartment to generalize these 
#two values into one category
new_data['hh_type'] = new_data['hh_type'].replace(['High-rise apartment (5 or more stories)', 'Low-rise apartment (less than 5 stories)'], ['Apartment', 'Apartment'])
new_data['hh_type'] = new_data['hh_type'].replace(['Single detached house'], ['House'])
#Drop all the values that does not give much insights
new_data.drop(new_data.loc[new_data["average_hours_worked"]=="Don't know"].index, inplace=True)
new_data.drop(new_data.loc[new_data["partner_main_activity"]=="Other"].index, inplace=True)
new_data.drop(new_data.loc[new_data["occupation"]=='Uncodable'].index, inplace=True) 
new_data.drop(new_data.loc[new_data["religion_participation"]== "Don't know"].index, inplace=True) 
new_data.drop(new_data.loc[new_data["living_arrangement"]== 'Other living arrangement'].index, inplace=True) 
new_data.drop(new_data.loc[new_data["vis_minority"]== "Don't know"].index, inplace=True) 
#descriptive Statistics showing the apartment and house counts for all the provinces
new_data['hh_type'].value_counts('Apartment')
new_data123 = new_data.groupby(['province', 'hh_type'])['caseid'].count()
new_data123 = new_data123.to_frame(name="Counts")
new_data123 = new_data123.reset_index()
df1 = new_data123[new_data123['hh_type']=='Apartment']
df1.rename({'Counts': 'Apartment Counts'}, axis=1, inplace=True)
df1.plot(kind = 'bar',
        x = 'province',
        y = 'Apartment Counts',
        color = 'green')
df2 = new_data123[new_data123['hh_type']=='House']
df2.rename({'Counts': 'House Counts'}, axis=1, inplace=True)
df2.plot(kind = 'bar',
        x = 'province',
        y = 'House Counts',
        color = 'blue')

#Since we would want to take out the biases that other provinces have a low value of apartment counts. To avoid these outliers, 
#We will be going to drop the provinces that has a low apartment count 
new_data.drop(new_data.loc[new_data["province"]== 'Manitoba'].index, inplace=True) 
new_data.drop(new_data.loc[new_data["province"]== 'Prince Edward Island'].index, inplace=True) 
new_data.drop(new_data.loc[new_data["province"]== 'Newfoundland and Labrador'].index, inplace=True) 
new_data.drop(new_data.loc[new_data["province"]== 'Nova Scotia'].index, inplace=True)                          
new_data.drop(new_data.loc[new_data["province"]== 'Saskatchewan'].index, inplace=True)                                                    
new_data.drop(new_data.loc[new_data["province"]== 'New Brunswick'].index, inplace=True)                            


#Actual Analysis 
from sklearn import tree 
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle 
from sklearn.preprocessing import LabelEncoder

## Firstly, I am going to encode the categorial objects to integers
letters = ["Male", "Female", "Female", "Male", "Male"]
le_1 = LabelEncoder()
le_1.fit(letters)
#Checking what it has uniquely identified:
le_1.classes_
new_data["sex"] = le_1.transform(new_data["sex"])

#Need to separate the education into another category 
#Split the education level to two different education categories
#Higher education is education levels that are post-secondary
new_data['education'].unique()

#Encode the categorical objects to integers for scaling at the end
let2 = ['High school diploma or a high school equivalency certificate',
       'Trade certificate or diploma',
       'College, CEGEP or other non-university certificate or di...',
       "Bachelor's degree (e.g. B.A., B.Sc., LL.B.)",
       "University certificate or diploma below the bachelor's level",
       'Less than high school diploma or its equivalent',
       'University certificate, diploma or degree above the bach...']
le_2 = LabelEncoder()
le_2.fit(let2)
#Checking what it has uniquely identified:
le_2.classes_
new_data['education'] = le_2.transform(new_data['education'])
#Higher education: 0, Lower education: 1
                                
#Categorize the partner's education into higher and lower educations as well
new_data["partner_education"].unique()

#Encode the categorical objects to integers for scaling at the end
let2 = ['Trade certificate or diploma',
       "Bachelor's degree (e.g. B.A., B.Sc., LL.B.)",
       'College, CEGEP or other non-university certificate or d...',
       'High school diploma or a high school equivalency certi...',
       'Less than high school diploma or its equivalent',
       'University certificate, diploma or degree above the ba...',
       "University certificate or diploma below the bachelor's level"]
le_2 = LabelEncoder()
le_2.fit(let2)
#Checking what it has uniquely identified:
le_2.classes_
new_data["partner_education"] = le_2.transform(new_data["partner_education"])
#Higher education: 0, Lower education: 1

#Categorize average working hour
new_data["average_hours_worked"].unique()

#Encode the categorical objects to integers for scaling at the end
let3 = ['30.0 to 40.0 hours', '50.1 hours and more', '40.1 to 50.0 hours',
       '0.1 to 29.9 hours', '0 hour']
le_3 = LabelEncoder()
le_3.fit(let3)
#Checking what it has uniquely identified:
le_3.classes_
new_data["average_hours_worked"] = le_3.transform(new_data["average_hours_worked"])
#Living alone: 0, Living together: 1

#Categorize martial status
new_data["marital_status"].unique()

#Encode the categorical objects to integers for scaling at the end
let4 = ['Single, never married', 'Married', 'Living common-law',
       'Separated', 'Divorced', 'Widowed']
le_4 = LabelEncoder()
le_4.fit(let4)
#Checking what it has uniquely identified:
le_4.classes_
new_data["marital_status"] = le_4.transform(new_data["marital_status"])
#Living alone: 0, Living together: 1

#Categorize average working hours
new_data["partner_main_activity"].unique()

#Encode the categorical objects to integers for scaling at the end
let5 = ['Working at a paid job or business', 'Going to school', 'Retired',
       'Long term illness', 'Looking for paid work', 'Household work',
       'Maternity/paternity/parental leave', 'Caring for children',
       'Volunteering or care-giving other than for children']
le_5 = LabelEncoder()
le_5.fit(let5)
#Checking what it has uniquely identified:
le_5.classes_
new_data["partner_main_activity"] = le_5.transform(new_data["partner_main_activity"])
#Employed: 0, Unemployed: 1

#Categorize income
new_data["income_family"].unique()

#Encode the categorical objects to integers for scaling at the end
let6 = ['$25,000 to $49,999', '$75,000 to $99,999', '$50,000 to $74,999',
       '$125,000 and more', '$100,000 to $ 124,999', 'Less than $25,000']
le_6 = LabelEncoder()
le_6.fit(let6)
#Checking what it has uniquely identified:
le_6.classes_
new_data["income_family"] = le_6.transform(new_data["income_family"])
#High Income: 0, Low Income: 1

#Categorize occupation
new_data["occupation"].unique()

#Encode the categorical objects to integers for scaling at the end
let7 = ['Sales and service occupations',
       'Trades, transport and equipment operators and related oc...',
       'Business, finance, and administration occupations',
       'Occupations in education, law and social, community and ...',
       'Management occupations',
       'Occupations in art, culture, recreation and sport',
       'Occupations in manufacturing and utilities',
       'Natural and applied sciences and related occupations',
       'Health occupations',
       'Natural resources, agriculture and related production oc...']
le_7 = LabelEncoder()
le_7.fit(let7)
#Checking what it has uniquely identified:
le_7.classes_
new_data["occupation"] = le_7.transform(new_data["occupation"])
#Industrial Industry: 0, Service Industry: 1

#Categorize have children or not
new_data["children_in_household"].unique()

#Encode the categorical objects to integers for scaling at the end
let8 = ['No child', 'Two children', 'One child', 'Three or more children']
le_8 = LabelEncoder()
le_8.fit(let8)
#Checking what it has uniquely identified:
le_8.classes_
new_data["children_in_household"] = le_8.transform(new_data["children_in_household"])
#Have children: 0, No children: 1

#Categorize religious participation
new_data["religion_participation"].unique()


#Encode the categorical objects to integers for scaling at the end
let9 = ['Once or twice a year', 'Not at all', 'At least once a month',
       'At least 3 times a year', 'At least once a week']
le_9 = LabelEncoder()
le_9.fit(let9)
#Checking what it has uniquely identified:
le_9.classes_
new_data["religion_participation"] = le_9.transform(new_data["religion_participation"])
#Have children: 0, No children: 1

#Categorize religious participation
new_data["living_arrangement"].unique()

let10 = ['Alone', 'Spouse only',
       'Spouse and single child under 25 years of age',
       'Spouse and single child 25 years of age or older',
       'No spouse and single child under 25 years of age',
       'Spouse and other', 'Living with two parents',
       'No spouse and single child 25 years of age or older',
       'Living with one parent', 'Spouse and non-single child(ren)']
le_10 = LabelEncoder()
le_10.fit(let10)
#Checking what it has uniquely identified:
le_10.classes_
new_data["living_arrangement"] = le_10.transform(new_data["living_arrangement"])
#Have children: 0, No children: 1

#Categorize visible minority
new_data["vis_minority"].unique()

#Encode the categorical objects to integers for scaling at the end
let11 = ['Not a visible minority', 'Visible minority', 'Visible minority', 
         'Not a visible minority', 'Not a visible minority']
le_11 = LabelEncoder()
le_11.fit(let11)
#Checking what it has uniquely identified:
le_11.classes_
new_data["vis_minority"] = le_11.transform(new_data["vis_minority"])
#Have children: 0, No children: 1
#//////////////////////////////////////////////////////////////////////////////
#Before doing further analysis, I will need to drop the province column since we will be using the following
#data to represent the general population 
new_data = new_data.drop(labels='province', axis = 1) 
#Categorize 
#sort out to a new dataset 
only_hh = new_data['hh_type']
df = only_hh.to_frame(name="hh_type")
#drop the column caseid and hh_type
new_data = new_data.drop(labels='caseid', axis = 1)
new_data = new_data.drop(labels='hh_type', axis = 1)
#Then let us try to split the outcome variable to training and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(new_data, df, test_size = 0.2, random_state=0)


#Save a list of the features we want to use in the classifier 
Features = list(new_data.columns)

df_shuffled = shuffle(new_data,random_state = 2)

#//////////////////////////////////////////////////////////////////////////////
# import required modules
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
##Decision tree 
from sklearn import tree 

dt_ = tree.DecisionTreeClassifier(max_depth = 4, random_state = 0)
dt_ = dt_.fit(x_train, y_train)

fig = plt.figure(figsize=(25,20))
tree.plot_tree(dt_, 
               feature_names=Features,
               class_names=['Apartment', 'House'],
               filled=True, impurity=True, rounded=True, fontsize = 8)

y_score2 = dt_.predict_proba(x_test)[:,1]

from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score

fpr, tpr, _  = roc_curve(y_test,y_score2, pos_label=dt_.classes_[1])

roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

#AUC score
auc = roc_auc_score(y_test, y_score2)
print('AUC_tree: %.3f' % auc)


##Random Forest 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
rf = RandomForestClassifier(max_depth = 4, random_state = 0)
rf.fit(x_train, y_train)

fig = plt.figure(figsize=(25, 20))
plot_tree(rf.estimators_[0],
          feature_names=Features,
          class_names=['Apartment', 'House'],
          filled=True, impurity=True, proportion=(70), 
          rounded=True, fontsize=8)

y_score3 = rf.predict_proba(x_test)[:,1]
fpr, tpr, _  = roc_curve(y_test,y_score3, pos_label=rf.classes_[1])

roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

auc = roc_auc_score(y_test, y_score3)
print('AUC_random_forrest: %.3f' % auc)
print(' ')


#Logistic Model
df = df.replace({'Apartment':0, 'House': 1})
y_test = y_test.replace({'Apartment':0, 'House': 1})
y_train = y_train.replace({'Apartment':0, 'House': 1})
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

logit = LogisticRegression()
logit.fit(x_train_scaled, y_train)

y_pred = logit.predict(x_train_scaled)
logit.score(x_train_scaled, y_train)


#Plot the logistic regression
sns.regplot(x=x_train_scaled[:, 0], y=y_train, logistic=True)

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, logit.predict(x_test_scaled))
acc_score

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, logit.predict(x_test_scaled))
mse

#use model to predict probability that given y value is 1
y_score1 = logit.predict_proba(x_test_scaled)[:,1]

from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score

fpr, tpr, _  = roc_curve(y_test,y_score1, pos_label=logit.classes_[1])

roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

#//////////////////////////////////////////////////////////////////////////////
#Decision Tree confusion matrix
dt_matrix = metrics.confusion_matrix(y_test, np.round(abs(y_score2)))
dt_matrix


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(dt_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#Random Forest confusion matrix
rf_matrix = metrics.confusion_matrix(y_test, np.round(abs(y_score3)))
rf_matrix

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(rf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#Confusion matrix for logistic model
cnf_matrix = metrics.confusion_matrix(y_test, np.round(abs(y_score1)))
cnf_matrix
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


