#Importing required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# define a Gaussain NB classifier
clf1 = GaussianNB()

#Added one more classifier
clf2 = KNeighborsClassifier(3)

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model for GaussianNB
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf1.fit(X_train, y_train)

    # do the test-train split and train the model for KNeighborsClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf2.fit(X_train, y_train)    

    # calculate the print the accuracy score GaussianNB
    acc1 = accuracy_score(y_test, clf1.predict(X_test))
    print(f"Model trained with accuracy1: {round(acc1, 3)}")


    # calculate the print the accuracy score KNeighborsClassifier
    acc2 = accuracy_score(y_test, clf2.predict(X_test))
    print(f"Model trained with accuracy2: {round(acc2, 3)}")
    # Checking the best acuracy and printing it
    if acc1>=acc2:
        print(f"Model trained with best accuracy of :" , acc1)
    else:
        print(f"Model trained with best accuracy of :" , acc2)


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction1 = clf1.predict([x])[0]
    print(f"Model prediction: {classes[prediction1]}")
    return classes[prediction1]  






# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained for GaussianNB
    clf1.fit(X, y)
  
      # fit the classifier again based on the new data obtained for KNeighborsClassifier
    clf2.fit(X, y)
    