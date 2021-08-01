## https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2
## https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
## https://scikit-learn.org/0.17/auto_examples/calibration/plot_compare_calibration.html#example-calibration-plot-compare-calibration-py
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib
##matplotlib.use('PS')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_squared_error
import numpy as np

###############################################################
## Author: Rohan Gujral
## Project: Machine Learning Germline variant classifier
###############################################################

def plot_learning_curves(model, X1,  X2, y1, y2):
    X_train, X_val, y_train, y_val = X1, X2, y1, y2
    train_errors, val_errors = [], []
    ##for m in range(10000, len(X_train), 10000):
    for m in range(1000, len(X_train), 1000):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)

        counter = 0
        ##if (m == 92000):
        if (m == 105000):
            for i in range(1, len(y_val)):

                if (y_val[i] != y_val_predict[i]):
                    counter += 1
                    print( "" + str(i) + ": " + str(y_val[i]) + "\t" + str(y_val_proba[i])+ "\t" + str(counter), y_val_predict[i])
            print("Value of counter: ", counter)

        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
        print("Value of m: ", m)

        ##exit()
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training set")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Test set")
    plt.xlabel('Training size in Ks')
    plt.ylabel('RMSE')
    plt.show()

###############################################################
#### Reading training data
#### File name: cleaned_published_data_matrix_training_set
###############################################################
dataframe = pd.read_csv("Path to training data set cleaned_published_data_matrix_training_set", sep='\t')

print(dataframe.head(5))

print(dataframe.shape)
array = dataframe.values

dataframe.drop('truth', axis=1).plot(kind='box', subplots=True, layout=(6,4), sharex=False, sharey=False, figsize=(9, 9),
                                        title='Box Plot for each input variable')
plt.savefig('forestDNM')
plt.show()

X = array[:, 0:23]
Y = array[:, 23]
for i in range(5):
    print('X' + str(i) +' :', X[i])
    print('Y' + str(i) +' :', Y[i])

print("Describe: ", dataframe['QL'].describe())

plt.hist(X[:,10], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85, range=[0,3000])
plt.ylabel('Probability');
plt.show()
##X, Y = shuffle(X, Y)

test_size = 0.20
seed = 18
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# Fit the model on 80% & test on remaining 20%
###### LOGISTIC REGRESSION ######
modelLog = LogisticRegression()
knn = KNeighborsClassifier()
lda = LinearDiscriminantAnalysis()
gnb = GaussianNB()
dtc = DecisionTreeClassifier()
svm = SVC()
rfc = RandomForestClassifier(n_estimators=200)
adabst = AdaBoostClassifier(n_estimators=200)

##plot_learning_curves(rfc, X_train, X_test, y_train, y_test)

##modelLog.fit(X_train, y_train)

###### MAKING PREDICTIONS ######
print("Making Predictions:")
model = rfc.fit(X_train, y_train) ## training again with RFC
print("Done with training:")

print("feature importance: ", model.feature_importances_)

impParams = list(zip(X[1,:], model.feature_importances_))

y_test_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix: ", cm)

#################################
#### Path to validation data ####
#### Small validation data   ####
#### set is provided here    ####
#### for evaluation.         #### 
#### smallValidationData     ####
#################################
pred_dataframe = pd.read_csv("Path to sample validation data smallValidationData", sep='\t')
array_pred = pred_dataframe.values
Xpred = array_pred[:, 0:23]
Y_chr = array_pred[:, 23]

print("Describe: ", pred_dataframe['QL'].describe())
plt.hist(pred_dataframe['QL'], bins='auto', alpha=0.7, rwidth=0.85, range=[0,3000])
plt.ylabel('Probability');
plt.show()

probab = model.predict_proba(Xpred)[:,1]
##y_PRED = rfc.predict(Xpred)
##print("Pro: " + y_PRED);
counter = 0
print()
print("=====================================")
print("DNM Calls with probability > 0.5")
print("=====================================")
print()
print("Probability: Value" + "\t" + "chr-position" )
print("Probabilities: " + str(probab[i]) + "\t" + str(Y_chr[i]))
for i in range(0, len(probab), 1):

    if(probab[i] > 0.5):
        ##print("Probabilities: " + str(Y_chr[i]) + "\t" + str(Xpred[i]) + "\t" + str(probab[i]))
        ##print("Probabilities: " +  str(probab[i]) + "\t" + str(Xpred[i]) + "\t" + str(Y_chr[i]) )
        print("Probability: " + str(probab[i]) + "\t" + str(Y_chr[i]))
        counter = counter + 1

print("Counter: ", counter)

exit()



###############################################################################
# Plot calibration plots

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(modelLog, 'Logistic Regression'),
                  (knn, 'KNeighbors'),
                  (gnb, 'Gaussian Naive Bayes'),
                  (lda, 'Linear Discriminant'),
                  (dtc, 'DecisionTreeClassifier'),
                  ##(svm, 'Support Vector Classification'),
                  ##(adabst, 'AdaBoost'),
                  (rfc, 'Random Forest')]:

    clf.fit(X_train, y_train)

    print('Accuracy of ' + name + ' classifier on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
    print('Accuracy of ' + name + ' classifier on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))


    #####################################
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix: ", cm)
    print('Accuracy' + name +  ' classifier on training set: {:.2f}'
        .format(accuracy_score(y_test, y_pred)))
    #####################################

    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

    print('Probabiltity position for ' + name + ': ', prob_pos)
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_xlabel("Mean predicted value")
ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
##ax1.legend(loc="lower right")
ax1.legend(loc="upper left")
ax1.set_title('Calibration plots  (reliability curve)')


ax2.set_xlabel("Probability from classifier decision function")
ax2.set_ylabel("Count")
##ax2.legend(loc="upper center", ncol=2)
ax2.legend(loc="upper right", ncol=2)

plt.tight_layout()
plt.show()
