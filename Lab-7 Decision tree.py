#!/usr/bin/env python
# coding: utf-8

# # Lab-7 Decision tree
# ### นายอธิศ สุนทโรดม
# ### Sec.2  65543206086-2

# In[4]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

df = pd.read_csv('student_major.csv')
df['GPA_old_group'] = df['GPA_old_group'].replace(['Bad', 'Normal', 'Good'],[0,1,2])
df['Age_group'] = df['Age_group'].replace(['<=20','21-25','26-30'],[0,1,2])

df['Rank_grade_major'] = df['Rank_grade_major'].replace(['Bad','Normal','Good'],[0,1,2])
df['Rank_grade_business'] = df['Rank_grade_business'].replace(['Bad','Normal','Good'],[0,1,2])
df['Rank_grade_computer'] = df['Rank_grade_computer'].replace(['Bad','Normal','Good'],[0,1,2])
df['Rank_grade_finance'] = df['Rank_grade_finance'].replace(['Bad','Normal','Good'],[0,1,2])
df['Rank_grade_total'] = df['Rank_grade_total'].replace(['Bad','Normal','Good'],[0,1,2])

study_onehot = pd.get_dummies(df['Study'],dummy_na=True)
df = pd.concat([df,study_onehot],axis=1)
df.rename(columns={"BANGKOK": "StudyBANGKOK", "UPCOUNTRY": "StudyUPCOUNTRY",np.nan:"StudyNaN"}, inplace=True)
df = df.drop(columns=['Study'])

Rank_study_group_onehot = pd.get_dummies(df['Rank_study_group'],dummy_na=True)
df = pd.concat([df,Rank_study_group_onehot],axis=1)
df.rename(columns={"GENERAL_EDU": "Rank_study_group_GENERAL_EDU", "VOCATIONAL_EDU": "Rank_study_groupVOCATIONAL_EDU",np.nan:"Rank_study_group_NaN"}, inplace=True)
df = df.drop(columns=['Rank_study_group'])

# TODO:
# 1. ทำ encode ให้ครบทุกคอลัมน์ที่จำเป็น
DT = DecisionTreeClassifier()
X = df.drop(columns=['Major'])
y = df['Major']

# 1.5 ทำ train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. train decision tree (ใช้ GridSearchCV ลองเปลี่ยน max_depth, critertion)
param_grid = {'max_depth': [3,5,7],
    'min_samples_split': [3,5,7],
    'criterion': ['gini', 'entropy']}
tree = GridSearchCV(DT, param_grid, cv=5,scoring='accuracy')
tree.fit(X_train, y_train)
tree = GridSearchCV(DT, param_grid, cv=5,scoring='accuracy')
tree.fit(X_train, y_train)
print("Best estimator :",tree.best_estimator_)
df


# In[8]:


from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

# 3. test กับ test set (หา acc, confusion matrix)
y_pred = tree.best_estimator_.predict(X_test)
acc = np.sum(y_pred == y_test)/len(X_test)
print("Acc : ",acc)
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(2,2))
plt.imshow(cm)

# 4. plot กราฟ acc vs max_depth 2, 3, 4, 5, ...., vs criterion (gini/entropy)
cv_results = tree.cv_results_
mean_test_scores = cv_results['mean_test_score']
param_max_depths = cv_results['param_max_depth']
param_criterions = cv_results['param_criterion']

unique_depths = np.unique(param_max_depths)
unique_criterions = np.unique(param_criterions)
mean_scores_per_depth = {}
mean_scores_per_criterion = {}

for depth in unique_depths:
    mean_scores_per_depth[depth] = []

for criterion in unique_criterions:
    mean_scores_per_criterion[criterion] = []

for i in range(len(mean_test_scores)):
    depth = param_max_depths[i]
    criterion = param_criterions[i]
    score = mean_test_scores[i]
    mean_scores_per_depth[depth].append(score)
    mean_scores_per_criterion[criterion].append(score)
# Plot accuracy vs. max_depth
plt.figure(figsize=(8, 5))
for depth, scores in mean_scores_per_depth.items():
    plt.plot(scores, label=f"max_depth={depth}")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. max_depth")
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy vs. criterion
plt.figure(figsize=(8, 5))
for criterion, scores in mean_scores_per_criterion.items():
    plt.plot(scores, label=f"criterion={criterion}")
plt.xlabel("criterion")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. criterion")
plt.legend()
plt.grid(True)
plt.show()

# 5. gen ภาพ dicision tree (webgrphviz)
best_dt_model = tree.best_estimator_
tree_dot = export_graphviz(
    best_dt_model,
    out_file=None,
    feature_names= df.columns[1:],
    class_names = df['Major'].unique().tolist(),
    rounded = True,
    filled = True
)
print(tree_dot)


# In[9]:


from sklearn.ensemble import RandomForestClassifier  # The model

#6 (extra) ลอง random forest
Rfc = RandomForestClassifier(random_state=42)
X_rfc = df.drop(columns=['Major'])
Y_rfc = df['Major']
x_train, x_test, Y_train, Y_test = train_test_split(X_rfc, Y_rfc, test_size=0.25, random_state=42)

param_grid = { 
    'n_estimators': [200, 500],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
clf_rfc = GridSearchCV(Rfc, param_grid, cv=5,scoring='accuracy')
clf_rfc.fit(x_train, Y_train)
print(tree.cv_results_)
print(tree.best_params_)
print(tree.best_estimator_)

y_predrfc = clf_rfc.best_estimator_.predict(x_test)
acc = np.sum(y_predrfc == Y_test)/len(x_test)
print(acc)
cm = confusion_matrix(Y_test,y_predrfc)
plt.figure(figsize=(8,8))
plt.imshow(cm)

BEST_RCF_MODEL = clf_rfc.best_estimator_

rcf_dot = export_graphviz(
    BEST_RCF_MODEL[0],
    out_file=None,
    feature_names= df.columns[1:],
    class_names = df['Major'].unique().tolist(),
    rounded = True,
    filled = True
)
print(rcf_dot)


# In[ ]:




