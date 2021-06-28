
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 

from sklearn.linear_model import LogisticRegression 

purchase_pred = classifier.predict(salary_test)

#Let's try to predict if a user will buy the product or not, using this model.

user_age = int(input("Enter age of the customer -> "))
user_salary = int(input("Enter the salary of the customer -> "))

user_test = sc_x.transform([[user_salary, user_age]])

user_purchase_pred = classifier.predict(user_test)

if user_purchase_pred[0] == 1:
  print("This customer may purchase the product!")
else:
  print("This customer may not purchase the product!")