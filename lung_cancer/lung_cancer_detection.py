import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

lung_cancer=pd.read_csv("../../pythonverivemakinaöğrenmesi/lung_cancer/lung_cancer_examples.csv")
print(lung_cancer.head(3))
lung_cancer.info()

from sklearn.model_selection import train_test_split

y=lung_cancer[["Result"]]
x=lung_cancer.drop(columns=["Name","Surname","Result"],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
tree=DecisionTreeClassifier()
model=tree.fit(x_train,y_train)

oran=model.score(x_test,y_test)
print(oran)

c=list(lung_cancer.iloc[60])
print(c)
print(c[2:-1])

sonuc=c[2:-1]

kanser_mi=model.predict([sonuc])

if(kanser_mi==0):
  print("Kanser değildir.")
else:
    print("Kanser")

print(len(lung_cancer))

