import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
import numpy as np

np.random.seed(31415) 

# Showing an image of the titanic
image = Image.open('Sea_Trials_of_RMS_Titanic,_2nd_of_April_1912.JPG')
st.image(image, caption='Picture of the Titanic',
         use_column_width=True)

# Reading in the titanic training dataset
train = pd.read_csv("./titanic/train.csv")

# Doing some data preprocessing so that the random forests classifier can use the input data
# It looks like there are a number of records where the age of the Passengers were not recorded 
# and there are a lot of records where the cabin wasn't recorded. We aren't interested in the cabin
# so we will exclude it as an input variable. With Age though, this might be an important predictor
# of survival. Therefore we have two choices, either we replace them via inputation (eg. give them the mean age)
# or we drop these records. For now we will drop these records.
# st.write(train.isnull().sum())
train = train.dropna(subset=['Age'])

# Converting Sex into a numeric variable
sex = {'female': 1, 'male': 0}
train['Sex'] = train['Sex'].map(sex)

# Creating the target variable from the training dataset which in this case is Survived 
# so this can be used to train the ml model.
Y = train.pop('Survived')

# Creating the input variables from the training dataset which in this case 
# we will use sex, age, and passenger class as these seem like variables which might 
# be predictive of survival.  
X = train[['Sex', 'Age', 'Pclass']]

# Splitting the Training Dataset into a test and training set. 
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, stratify=Y)

# Code used to enable the Random Forests Classifier to handle categorical variables (which in this case is Sex)
encoded_features = X.columns[X.dtypes==object].tolist()
col_trans = make_column_transformer(
                        (OneHotEncoder(),encoded_features),
                        remainder = "passthrough"
                        )

st.title('Titanic Survival Predictor')
st.write("This app allows users to change the Age, Sex and Passenger Class of a passenger in order to predict whether or they would survive the Titanic Disaster via a Machine Learning algorithm (Random Forests). ")
        
# Adding options so that the label in the selectbox shows male or female rather than 0 or 1.
display = ("male", "female")
options = list(range(len(display)))

# Creating the function which allows the users to play with the input
def user_input_features():
    chosen_sex = st.selectbox('What is the Passenger\'s Sex?', options, format_func=lambda x: display[x])
    chosen_age = st.slider('What is the Passenger\'s Age?', 0, 50, 100)
    chosen_passenger_class = st.slider('What is the Passenger\'s Ticket Class', 1, 2, 3)
    data = {'Sex': chosen_sex,
            'Age': chosen_age,
            'Pclass': chosen_passenger_class
            }
    features = pd.DataFrame(data, index=[0])
    return features

st.header("Input")
df = user_input_features()

# Creating the random forests classifer
rf_classifier = RandomForestClassifier()

# Fitting the model
pipe = make_pipeline(col_trans, rf_classifier)
pipe.fit(X_train, y_train)

# Getting prediction to determine the accuracy of the model
y_pred_test = pipe.predict(X_test)

# Predicting the outcome of survival based on the user input
prediction = pipe.predict(df)

if(pipe.predict(df)[0]) == 1:
    st.header("Prediction")
    st.write('Good news! The Passenger is predicted to survive!')
else:
    st.header("Prediction")
    st.write('Unfortunately, the Passenger is predicted to die')
   
# Creating a confusion matrix to assess the accuracy of the model
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float')*100 / matrix.sum(axis=1)[:, np.newaxis]
matrix = pd.DataFrame(matrix)
matrix = matrix.rename(columns={0:'Actual Died', 1:'Actual Survived'}, index={0: 'Predicted Dead', 1: 'Predicted Survival'})

# Calculating the accuracy score
accuracy_of_model = accuracy_score(y_test, y_pred_test)
accuracy_of_model_percent = accuracy_of_model *100

# Showing the accuracy of the model
st.header('Accuracy')
st.write('The Machine Learning Algorithm is {0:.2f}% accurate. Below is a confusion matrix showing the percentage of passengers predicted to survive with those who actually survived in order to further assess the accuracy of the model.'.format(accuracy_of_model_percent))
st.table(matrix)




