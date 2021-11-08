# -*- coding: utf-8 -*-
"""
@author: john

"""

# Import packages
from sklearn.naive_bayes import CategoricalNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

# Import data
values = pd.read_excel('Book1.xlsx')
#print(values)

# Split dataset 
train_data = values.sample(frac=0.8)

test_data = values
test_data = test_data[~test_data.isin(train_data)].dropna()

# Estimate the CPD for each variable based on a given data set.
model = BayesianModel([('A', 'C'), ('A', 'D'), ('A', 'B'), ('D', 'C'), ('D', 'B'), ('C', 'D2'), ('B', 'D2'), ('C', 'E'), ('D2', 'E'), ('B', 'E'), ('E', 'OUTPUT')])
model.fit(train_data)

print("\n")
for cpd in model.get_cpds():
    print("CPD of {variable}:".format(variable=cpd.variable))
    print(cpd)
    


# Initializing the VariableElimination class
infer = VariableElimination(model)


# Computing the joint probability of C and E given B=no.
q2 = infer.query(variables=['C', 'E'], evidence={'D2': 1})
print("joint probability of C and E given D2=YES",q2)

# Computing the joint probability of C and E given B=no.
q2 = infer.query(variables=['C', 'E'], evidence={'D': 1})
print("joint probability of C and E given D=YES",q2)

# Computing the joint probability of C and E given B=no.
q2 = infer.query(variables=['C', 'E'], evidence={'B': 1})
print("joint probability of C and E given B=YES",q2)




# Computing the probability of D2 given C=no.
q = infer.query(variables=['E'], evidence={'D2': 1})
print("probability of E given D2=yes",q)

q = infer.query(variables=['D2'], evidence={'B': 1})
print("probability of D2 given B=yes",q)


# Computing the probability of D2 given C=no.
q = infer.query(variables=['D2'], evidence={'E': 1})
print("probability of D2 given E=yes",q)


# Computing the probability of D2 given C=no.
q = infer.query(variables=['C'], evidence={'D': 1})
print("probability of C given D=yes",q)

# Computing the probability of D2 given C=no.
q = infer.query(variables=['C'], evidence={'A': 1})
print("probability of C given A=yes",q)

# Computing the probability of D2 given C=no.
q = infer.query(variables=['D2'], evidence={'C': 0})
print("probability of D2 given C=no",q)

# Computing the joint probability of C and E given B=no.
q2 = infer.query(variables=['C', 'E'], evidence={'D2': 1})
print("joint probability of C and E given D2=no",q2)

# Computing the probabilities (not joint) of C and E given B=no.
q3 = infer.query(variables=['C', 'E'], evidence={'B': 0}, joint=False)
print("probabilities (not joint) of C and E given B=no")
for factor in q3.values():
    print(factor)

# Computing the MAP of D2 given C=no.
q4 = infer.map_query(variables=['D2'], evidence={'C': 0})
print(" MAP of D2 given C=no",q4)

# Computing the MAP of D and D2 given B=yes
q5 = infer.map_query(variables=['D', 'D2'], evidence={'B': 1})
print("MAP of D and D2 given B=yes",q5)


# Create the X, Y, Training and Test
xtrain = train_data.drop('OUTPUT', axis=1)
ytrain = train_data.loc[:, 'OUTPUT']
xtest = test_data.drop('OUTPUT', axis=1)
ytest = test_data.loc[:, 'OUTPUT']


# Init the Categorical Classifier
model = CategoricalNB()

# Train the model 
model.fit(xtrain, ytrain)

# Predict Output 
pred = model.predict(xtest)

# Plot Confusion Matrix
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')


def show_confusion(a, b):
    print(confusion_matrix(a, b))

print("confusion matrix")
show_confusion(ytest, pred)
tn, fp, fn, tp = confusion_matrix(ytest, pred).ravel()

count_misclassified = (ytest != pred).sum()
print('Misclassified samples: {}/251'.format(count_misclassified))


print("\n")
print("tn: " ,tn, "fp: " ,fp,"fn: ",fn,"tp: ",tp)
print("accuracy score: ",accuracy_score(ytest, pred))
print("precision score : ",metrics.precision_score(ytest, pred))
print("recall score: ",metrics.recall_score(ytest, pred))
print("f1 score: ",metrics.f1_score(ytest, pred))

