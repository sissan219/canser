from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
file_path = (r"c:\Users\VIP\Downloads\data.csv")
df = pd.read_csv(r"c:\Users\VIP\Downloads\data.csv")
x , y =  load_breast_cancer(return_X_y=True)
x_train , x_test , y_train , y_test = train_test_split(x ,y ,test_size=0.20 , random_state=25)
model = LogisticRegression(max_iter=1000 , random_state=0)
model.fit(x_train , y_train)
acc = accuracy_score(y_test, model.predict(x_test))*100
print(acc)
