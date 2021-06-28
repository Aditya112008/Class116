from sys import modules
import pandas as pd 
import csv
import plotly.express as px
import plotly.graph_objects as go 

df = pd.read_csv("./logistics_data.csv")
salary = df["EstimatedSalary"].tolist()
purchased = df["Purchased"].tolist()

fig = px.scatter(x = salary, y = purchased)
fig.show()

#---------------------
#Let's Plot the data on the scatter plot to see how different variables effect the purchase 

salaries = df["EstimatedSalary"].tolist()
ages =  df["Age"].tolist()
purchased = df["Purchased"].tolist()

# Lets visualize the data in the scattered plot 
# Age , Salary can be axis 
#Red dots for people who refused to buy iphone (0)
#Green Dots Who Decide To Buy IPhone(1)

colors = []
for data in purchased : 
    if data == 1 :
        colors.append("green") 

    else :
        colors.append("red")

fig = go.Figure(data = go.Scatter(x = salaries , y = ages , mode = 'markers', marker = dict(color = colors)))
fig.show()

#We Observe that there is some relation in age , salary whether the people decide to buy IPhone or not 
#Also the plot is divided in 2 parts red dots(PEOPLE WHO DIDN'T BUY ARE CLUSTERED TOGETHER)
# Green Dots (PEOPLE WHO DID BUY ARE CLUSTERED TOGETHER)

#-------------------------------------------------------------

#Now Using the machine learning libraries we will build a model to predict that will the person buy the IPhone or not
#We will divide the data into 2 parts:
# 1st part - (75% of data) To Train this model to predict if the person will buy or not
# 2nd part - Testing of the model 


# train_test_split() is used to split the data (75% , 25%)
#this method is present in sklearn.model_selection library 
#It takes (data to split , test_size , random_state)

#Random_state = 0 (Means training data)
#Random_state = 1 (Means testing data)

#StandardsScaler - this will compare values (age , salary) it will give some score 
#StandardsScaler assumes your data is normally distributed and centered around 0 and with a standarf deviation of 1

#Accuracy = 1 (Means 100% likely to buy)
# Nearby to 1 (likely to buy)

#-----------------------------------------------

#Taking together Age and Salary of the person
factors = df[["EstimatedSalary", "Age"]]

#Purchases made
purchases = df["Purchased"]

from sklearn.model_selection import train_test_split 

salary_train, salary_test, purchase_train, purchase_test = train_test_split(factors, purchases, test_size = 0.25, random_state = 0)

print(salary_train[0:10])

from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 

salary_train = sc_x.fit_transform(salary_train)  
salary_test = sc_x.transform(salary_test) 
  
print (salary_train[0:10])

#Here, we can see that both the age and the salary of the person are given points. Now, we are sure that each and every feature will contribute equally to the decision making of our machine learning model, where we want to predict if the user will buy the product or not.
#Now, let's train a logistic regression model on the training data that we seperated out earlier. For this, we will use sklearn's pre-built class LogisticRegression.

from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) 
classifier.fit(salary_train, purchase_train)

#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
         #          intercept_scaling=1, l1_ratio=None, max_iter=100,
         #          multi_class='auto', n_jobs=None, penalty='l2',
         #          random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
         #          warm_start=False) 

#Here, the random_state = 0 means that we are providing the model with the Training data. If the value would have been 1, it would mean that we are providing with the testing data.
#Now that our model for Logistic Regression is trained, it's time for us to test this model. Is it predicting values well? Let's find out.

purchase_pred = classifier.predict(salary_test)

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(purchase_test, purchase_pred)) 

#Accuracy :  0.89

#An accuracy of 1 would have been perfectly accurate, 0.89 is an excellent accuracy.

#Let's try to predict if a user will buy the product or not, using this model.

user_age = int(input("Enter age of the customer -> "))
user_salary = int(input("Enter the salary of the customer -> "))

user_test = sc_x.transform([[user_salary, user_age]])

user_purchase_pred = classifier.predict(user_test)

if user_purchase_pred[0] == 1:
  print("This customer may purchase the product!")
else:
  print("This customer may not purchase the product!")