#!/usr/bin/python3
from sklearn.datasets import load_iris
from sklearn import tree
import numpy
#loading data
iris=load_iris()

#printing features
print(iris.feature_names)

#printing target
print(iris.target_names)
x=[0,50,100]
only_target_training=numpy.delete(iris.target,x,axis=0)
print(only_target_training.size)

only_data_training=numpy.delete(iris.data,x,axis=0)
print(only_data_training)

#testing target
test_target=iris.target[x]
print(test_target)

test_data=iris.data[x]
print (test_data)

#calling algorithm
clf=tree.DecisionTreeClassifier()
trained=clf.fit(only_data_training,only_target_training)
print(trained)
output=trained.predict(test_data)
print(output)

print (test_target)
print(test_data)



#training data
#print(iris.data)
#setosa=iris.data[0:50]
#print(iris.target)
#print (setosa.size)
#s_data=iris.target[0:50]
#print(s_data.size)


