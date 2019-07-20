import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

dataset = pd.read_csv('kyphosis.csv')


sns.pairplot(dataset,hue='Kyphosis')
'''plt.show()'''

from sklearn.model_selection import train_test_split

x = dataset.drop('Kyphosis',axis=1)

y = dataset['Kyphosis']


x_train,x_test,y_trian,y_test = train_test_split(x,y,test_size=0.50)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy')

dtree.fit(x_train,y_trian)

prediction = dtree.predict(x_train)


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


con_mat = confusion_matrix(y_test,prediction)

accuracy = accuracy_score(y_test,prediction)

print(con_mat,accuracy)


print(classification_report(y_test,prediction))

from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot

features = list(dataset.columns[1:])

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())

Image(graph[0])

