import numpy as np
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix

#open the training set and the test set files
train = pd.read_csv('trainingSet.csv')
test = pd.read_csv('mnist_test.csv')

features = train.iloc[:,1:].values #extract features
labels = train.iloc[:,0].values #extract labels
testFeatures = test.iloc[:,1:].values
test_labels = test.iloc[:,0].values

np.unique(np.isnan(features)) #check for missing data

#splitting the data into training data and validation data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state = 0)

sns.countplot(y_train) #show a bar chart of the frequency of each class in our dataset
plt.show()

#Normalizing the data before implementing the SVM model (each column will have mean=0 and standard deviation=1)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_test = sc_X.transform(testFeatures)

print("Choose a kernel:\n1.Linear\n2.Polynomial\n3.rbf")
choice = input("Please enter your choice (1, 2 or 3): " )
if choice == '1':
        kernelType = 'linear'
elif choice == '2':
        kernelType = 'ploy'
elif choice == '3':
        kernelType = 'rbf'

gammaValueIn = input("Type in a gamma value: ")
cValueIn = input("Type in a C value: ")

gammaValue=float(gammaValueIn)
cValue = float(cValueIn)


start = timeit.default_timer() #start a timer to calculate the duration of the model
# print the parameters being used for the user to comprehend what the program is currently computing
print('SVM Classifier with gamma =',gammaValue, ', Kernel = ',kernelType, ', C= ',cValue)
#set the parameters to the actual classifier
classifier = SVC(gamma=gammaValue, kernel=kernelType, random_state = 0, C=cValue)
classifier.fit(X_train,y_train) #start training

label_pred = classifier.predict(X_test) #predict labels of the validation set

#compute accuracies and the confusion matrix
modelAccuracy = classifier.score(X_test, y_test)
validationAccuracy = accuracy_score(y_test, label_pred)
confusionMatrix = confusion_matrix(y_test,label_pred)

#display the accuracies and confusion matrix
print('\nAccuray of the trained classifier: ', modelAccuracy)
print('\nAccuracy of the classifier on the validation set: ',validationAccuracy)
print('\nConfusion Matrix: \n',confusionMatrix)

#calculate the duration from the timer and display result in minutes
duration = round((timeit.default_timer() - start)/60, 1)
print("Duration: ", duration, "mins")

#Plot the confusion matrix and the corresponding parameters and results
(fig, ax) = plt.subplots(1,1)
ax.matshow(confusionMatrix)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.text(1.0, 0.5, 'Accuracy: {:.2%}\nDuration: {}min\nGamma: {}\nC: {}\nKernel: {}'.format(modelAccuracy, duration, gammaValue, cValue, kernelType),
        dict(fontsize = 10, ha = 'left', va = 'center', transform = ax.transAxes))
fig.canvas.set_window_title('Confusion matrix')
fig.tight_layout()
fig.show()

#Test the model with the testing set
result = classifier.predict(sc_test)
#Compute accuracy on the test set and plot the confusion matrix
testAccuracy = accuracy_score(test_labels, result)
test_confusionMatrix = confusion_matrix(test_labels, result)
print('\nAccuracy of the classifier on the test set: ', testAccuracy)
print('\nTest Confusion Matrix: \n', test_confusionMatrix)
# Plot the confusion matrix and the corresponding parameters and results
(fig, ax) = plt.subplots(1, 1)

ax.matshow(confusionMatrix)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.text(1.0, 0.5,
        'Test Accuracy: {:.2%}\nDuration: {}min\nGamma: {}\nC: {}\nKernel: {}'.format(testAccuracy, duration,
                                                                                      gammaValue, cValue,
                                                                                      kernelType),
        dict(fontsize=10, ha='left', va='center', transform=ax.transAxes))
fig.canvas.set_window_title('Test Confusion matrix')
fig.tight_layout()
fig.show()

(fig, ax) = plt.subplots(1,1)
randoms = np.random.randint(low = 1, high = 400, size = 8)
for i in randoms: #loop through eight random images and output them together with their respective prediction
        two_d = (np.reshape(testFeatures[i], (28, 28)) * 255).astype(np.uint8)
        plt.title('Predicted Label: {0}'.format(result[i]))
        plt.imshow(two_d, interpolation='nearest',cmap='gray')
        plt.show()
print('\n\n')

input()