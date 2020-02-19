import argparse
import sys
import os
import numpy as np
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  
from sklearn.model_selection import KFold


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    print(C)
    accuracy = sum(C.diagonal())/sum(map(sum, C))
    return accuracy


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    return (list(C.diagonal() / [row.sum() for row in C]))


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    return (list(C.diagonal() / [row.sum() for row in C.T]))
    

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    classifier = ['SGDClassifier', 'GaussianNB', 'RandomForestClassifier', 'MLPClassifier', 'AdaBoostClassifier']
    accuracies = [0, 0, 0, 0, 0]
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        i = 0
        for classifier_name in classifier:
            if classifier_name == 'SGDClassifier':
                print('SGD')
                model = SGDClassifier(max_iter=1000, tol=1e-3)
            if classifier_name == 'GaussianNB':
                print('GaussianNB')
                model = GaussianNB()
            if classifier_name == 'RandomForestClassifier':
                print('Random')
                model = RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)
            if classifier_name == 'MLPClassifier':
                print('MLP')
                model = MLPClassifier(alpha=0.05, random_state=42)
            if classifier_name == 'AdaBoostClassifier':
                print('Ada')
                model = AdaBoostClassifier(random_state=42)

            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            conf_matrix = confusion_matrix(y_test, predicted)
            acc = accuracy(conf_matrix)
            accuracies[i] = acc
            recalls = recall(conf_matrix)
            precisions = precision(conf_matrix)

            i+=1

            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recalls]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precisions]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
    
    iBest = accuracies.index(max(accuracies))
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
  
    if iBest == 0:
        model = SGDClassifier(max_iter=1000, tol=1e-3)
    if iBest == 1:
        model = GaussianNB()
    if iBest == 2:
        model = RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)
    if iBest == 3:
        model = MLPClassifier(alpha=0.05, random_state=42)
    if iBest == 4:
        model = AdaBoostClassifier(random_state=42)

    data_sizes = [1000, 5000, 10000, 15000, 20000]
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        for num_train in data_sizes:
            train_x = X_train[:num_train, :]
            train_y = y_train[:num_train]
            model.fit(train_x, train_y)
            predicted = model.predict(X_test)
            conf_matrix = confusion_matrix(y_test, predicted)
            acc = accuracy(conf_matrix)
            outf.write(f'{num_train}: {acc:.4f}\n')
    
    X_1k = X_train[:1000, :]
    y_1k = y_train[:1000]

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''

    if i == 0:
        model = SGDClassifier(max_iter=1000, tol=1e-3)
    if i == 1:
        model = GaussianNB()
    if i == 2:
        model = RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)
    if i == 3:
        model = MLPClassifier(alpha=0.05, random_state=42)
    if i == 4:
        model = AdaBoostClassifier(random_state=42)

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        # 1. 
        X_32k = X_train[:32000, :]
        y_32k = y_train[:32000]
        # for each number of features k_feat, write the p-values for
        # that number of features:
        for k_feat in [5, 50]:
            selector = SelectKBest(f_classif, k_feat)
            X_new = selector.fit_transform(X_32k, y_32k)
            p_values = selector.pvalues_
            selector.fit(X_new, y_32k)
            
            print(p_values)
            outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')
        # 2. 
        # 1k
        selector = SelectKBest(f_classif, 5)
        X_new = selector.fit_transform(X_1k, y_1k)
        pp_1k = selector.pvalues_
        model.fit(X_new, y_1k)
        predicted = model.predict(selector.transform(X_test))
        conf_matrix = confusion_matrix(y_test, predicted)
        accuracy_1k = accuracy(conf_matrix)
        
        # 32k
        selector = SelectKBest(f_classif, 5)
        X_new = selector.fit_transform(X_32k, y_32k)
        pp_32k = selector.pvalues_
        model.fit(X_new, y_32k)
        predicted = model.predict(selector.transform(X_test))
        conf_matrix = confusion_matrix(y_test, predicted)
        accuracy_full = accuracy(conf_matrix)
        # 3.
        top5_1k = np.argpartition(pp_1k, 5)[:5]
        top5_32k = np.argpartition(pp_32k, 5)[:5]

        feature_intersection = np.intersect1d(top5_1k, top5_32k)
        # 4. 
        top_5 = top5_32k

        
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top_5}\n')
     


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
    '''

    classifiers = ['SGDClassifier', 'GaussianNB', 'RandomForestClassifier', 'MLPClassifier', 'AdaBoostClassifier']
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        kf = KFold(n_splits=5, shuffle = True)
        accuracies = np.zeros((5,5))
        kfold_accuracies = []
        k = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            j = 0
            for classifier_name in classifiers:
                if classifier_name == 'SGDClassifier':
                    model = SGDClassifier(max_iter=1000, tol=1e-3)
                if classifier_name == 'GaussianNB':
                    model = GaussianNB()
                if classifier_name == 'RandomForestClassifier':
                    model = RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)
                if classifier_name == 'MLPClassifier':
                    model = MLPClassifier(alpha=0.05, random_state=42)
                if classifier_name == 'AdaBoostClassifier':
                    model = AdaBoostClassifier(random_state=42)
                
                model.fit(X_train, y_train)
                predicted = model.predict(X_test)
                conf_matrix = confusion_matrix(y_test, predicted)
                kfold_accuracies.append(accuracy(conf_matrix))
                
                j+=1
            accuracies[k] = np.array(kfold_accuracies[:])
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
            kfold_accuracies = []
            k+=1
        index = [0, 1, 2, 3, 4]
        index.remove(i)
        p_values = []
        for num in index:
            p_values.append(ttest_rel(accuracies[num], accuracies[i])[1])
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
      


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    # load data and split into train and test.
    npzfile = np.load(args.input)
    
    for data in npzfile:
        info = npzfile[data]

    x = info[:,:-1]
    y = info[:,-1]
  
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
  
    # complete each classification experiment, in sequence.
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    (X_1k, y_1k) = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)
