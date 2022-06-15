import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.cluster import DBSCAN
import networkx as nx
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

def cross_validate_model(model, X, y, scorer = 'f1_weighted', k = 5, plot = True, verbose = True):
    '''
    A function that plots the learning curve of the model and calculates the cross validation 
    scores for the train and test set.
    
    model: An sklearn model that has been initialized with parameters.
    
    X: DataFrame of training features.
    
    y: Series of training labels.
    
    scorer: Sklearn (or handmade) scorer for the model. Default scorer is f1-weighted. Written as a string.
    
    k: the number of k-fold cross-validations to run. Default value is 5.
    
    plot: Default is True. If set to true, the function plots the learning curve.
    
    verbose: Default is True. If set to true, it outpluts the average train and validation scores.
    
    returns: The average train and validation scores.
    '''
    cross_val_dict = cross_validate(model, X, y, 
                                    scoring = scorer, 
                                    cv = k,
                                    return_train_score = True)
    if verbose:
        print(f"Average train {scorer} {cross_val_dict['train_score'].mean()}")
        print(f"Average validation {scorer} {cross_val_dict['test_score'].mean()}")
    if plot == True:
        train_sizes_abs, train_scores, test_scores = learning_curve(estimator = model, 
                                                                    X = X, y = y, 
                                                                    cv = k, 
                                                                    scoring = scorer)
        fig, axs = plt.subplots(figsize = (8, 6))
        axs.plot(train_sizes_abs, np.mean(train_scores, axis = 1),  color = 'red')
        axs.plot(train_sizes_abs, np.mean(test_scores, axis = 1), color = 'blue')
        axs.set_xlabel('Training sample size')
        axs.set_ylabel(scorer)
        axs.legend(['Train', 'Validation'])
        axs.grid(color = 'grey')
        axs.tick_params(rotation = 0)
        axs.set_facecolor('whitesmoke')
        plt.show()
    return cross_val_dict['train_score'].mean(), cross_val_dict['test_score'].mean()

def grid_search_lr(X_train, y_train, c_list, solver_list, plot = False, verbose = True):
    '''
    A function that runs a grid search specifically on a random forest model with the above hyperparameters.
    
    X_train: A DataFrame that contains all of the training features. These are skills column as well as total percentage score on the placement test.
    
    y_train: A Series that contains the labels of the test set.
    
    c_list: A list of float values (0.0 to 1.0) for the parameter "C" in logistic regression.
    
    solver_list: A list of string values for the parameter "solver" in logistic regression.
    
    plot: If set to true, it plots a learning curve for the training of the model. Default is False.
    
    verbose: If set to true, the model parameters are printed as well as the average train and validation scores for the model parameters.
    
    returns: A DataFrame with the performance (avg train and validation scores) of each combination of hyperparameters.
    '''
    num_testings = len(c_list) * len(solver_list)
    cl = []
    sl = []
    train_mean_lst = []
    val_mean_lst = []
    print(f'Testing {num_testings} combinations \n')
    for c in c_list:
        for s in solver_list:
            cl.append(c)
            sl.append(s)
            if verbose:
                print(f'Model parameters => C: {c}, solver: {s}')
                print('-'*105)
            lr_model = LogisticRegression(class_weight = 'balanced', 
                                          C = c, 
                                          solver = s,
                                          random_state = 1234)
            train_mean, val_mean = cross_validate_model(lr_model, X_train, y_train,
                                                        plot = plot, verbose = verbose)
            train_mean_lst.append(train_mean)
            val_mean_lst.append(val_mean)
            if verbose:
                print('='*105, '\n\n')
            else:
                print('=', end = '')
    results_df = pd.DataFrame({'C': cl, 'solver': sl,
                               'train_mean': train_mean_lst, 'val_mean': val_mean_lst})
    results_df['overfit'] = results_df['train_mean'] - results_df['val_mean']
    return results_df

def grid_search_rf(X_train, y_train, min_samples_split, min_samples_leaf, criterion, max_features, plot = False, verbose = True):
    '''
    A function that runs a grid search specifically on a random forest model with the above hyperparameters.
    
    X_train: A DataFrame that contains all of the training features. These are skills column as well as total percentage score on the placement test.
    
    y_train: A Series that contains the labels of the test set.
    
    min_samples_split: A list of positive integer values for min_samples_split.
    
    min_samples_leaf: A list of positive integer values to test for min_samples_leaf
    
    criterion: A list of string values to test for the evaluation criterion of the split of a decision tree. Options are 'gini', 'entropy', 'logloss'
    
    max_features: A list of values to test for max_features. Options are 'sqrt', 'log2', None, integer or float values.
    
    plot: If set to true, it plots a learning curve for the training of the model. Default is False.
    
    verbose: If set to true, the model parameters are printed as well as the average train and validation scores for the model parameters.
    
    returns: A DataFrame with the performance (avg train and validation scores) of each combination of hyperparameters.
    '''
    num_testings = len(min_samples_split) * len(min_samples_leaf) * len(criterion) * len(max_features)
    mss = []
    msl = []
    cr = []
    mf = []
    train_mean_lst = []
    val_mean_lst = []
    print(f'Testing {num_testings} combinations \n')
    for s in min_samples_split:
        for l in min_samples_leaf:
            for c in criterion:
                for f in max_features:
                        mss.append(s)
                        msl.append(l)
                        cr.append(c)
                        mf.append(f)
                        if verbose:
                            print(f'Model parameters => min_split: {s}, min_leaf: {l}, split_criterion: {c}, max_features: {f}')
                            print('-'*105)
                        rf_model = RandomForestClassifier(n_estimators = 100,
                                                          min_samples_split = s,
                                                          min_samples_leaf = l,
                                                          criterion = c,
                                                          max_features = f,
                                                          class_weight = 'balanced',
                                                          random_state = 1234)
                        train_mean, val_mean = cross_validate_model(rf_model, X_train, y_train,
                                                                    plot = plot, verbose = verbose)
                        train_mean_lst.append(train_mean)
                        val_mean_lst.append(val_mean)
                        if verbose:
                            print('='*105, '\n\n')
                        else:
                            print('=', end = '')
    results_df = pd.DataFrame({'min_split': mss, 'min_leaf': msl,
                              'split_criterion':cr, 'max_features': mf,
                               'train_mean': train_mean_lst, 'val_mean': val_mean_lst})
    results_df['overfit'] = results_df['train_mean'] - results_df['val_mean']
    return results_df

def feature_importance(model, model_type):
    '''
    A function that plots the features in descending order with their feature importance. If the model is
    RandomForest, feature importance is calculated for the model as a whole. If the model is 
    LogisticRegression, feature importance is calculated for each class.
    
    model: The trained classification model.
    
    model_type: Can be either "LogisticRegression" or "RandomForest"
    
    returns: Nothing but the plot of the graph of feature importance.
    '''
    if model_type == 'LogisticRegression':
        fig, axs = plt.subplots(1,len(model.classes_), figsize = (16, 4))
        for i, label in enumerate(model.classes_):
            feat_importance = np.sort(model.coef_[i])[::-1]
            feature_names = model.feature_names_in_[np.argsort(model.coef_[i])[::-1]]
            axs[i].bar(x = feature_names,height = feat_importance, edgecolor = 'Black')
            axs[i].set_title(label)
            axs[i].set_xlabel('Feature')
            axs[i].set_ylabel('Feature Importance')
            axs[i].set_xticklabels(feature_names, rotation = 60)
        plt.show()
        plt.tight_layout()
    if model_type == 'RandomForest':
        feat_importance = np.sort(model.feature_importances_)[::-1]
        feature_names = model.feature_names_in_[np.argsort(model.feature_importances_)[::-1]]
        plt.bar(x = feature_names,
                height = feat_importance, edgecolor = 'Black')
        plt.xlabel('Feature')
        plt.ylabel('Feature Importance')
        plt.xticks(rotation = 45)
        plt.show()
        plt.tight_layout()

def create_error_df(all_df, X, y_true, y_pred, model_type = 'classification', model = None):
    '''
    A function that creates a DataFrame of the errors of the model. The DataFrame contains columns of student name, skill scores, graduation year, true_class, Grade, cluster labels, predicted labels, and predicted probabilities.
    
    all_df: A DataFrame that contains the following columns: student name, skill scores, graduation year, true_class, Grade, cluster labels. 
    
    X: A DataFrame that contains all of the testing features. These are skills column as well as total percentage score on the placement test.
    
    y_true: A Series that contains the labels of the test set.
    
    y_pred: A Numpy array containing all of the predicted labels of the test set.
    
    model_type: Could be 'classification' of 'clustering'. The default is classification.
    
    model: A classification model that has already been trained.
    
    returns: An error DataFrame. Description is above.
    '''
    #Get the total dataframe with all the original values
    data_idxs = y_true[y_pred != y_true].index
    errors = all_df.loc[data_idxs]
    if model_type == 'clustering':
        y_pred = pd.Series(y_pred, index = y_true.index)
        errors['pred_label'] = y_pred.loc[data_idxs]
        return errors
    else:
        errors['pred_label'] = model.predict(X.loc[data_idxs])
        errors['pred_prob'] = model.predict_proba(X.loc[data_idxs]).max(axis = 1)
        return errors.sort_values(by = 'pred_prob', ascending = False)

def create_confusion_matrix(y_true, y_pred, axis):
    '''
    A function that takes the true and predicted labels and outputs a confusion matrix.
    
    y_true: A Series that contains the labels of the test set.
    
    y_pred: A Numpy array containing all of the predicted labels of the test set.
    
    axis: An integer or tuple of integers that gives the index of the axes to plot the graph.
    
    returns: Plotted confusion matrix with labels.
    '''
    cm=confusion_matrix(y_true, y_pred, labels = y_true.unique())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=y_true.unique())
    disp.plot(xticks_rotation='vertical',cmap='Purples', ax = axs[axis])
    axs[axis].set_xticks(rotation = 45)

def analyze_errors_by_class(all_df, errors_df, X_train, y_true, y_pred, which_class = None):
    '''A function that outputs a confusion matrix of a given model and graphs the average class scores compared to the average scores of the errors within each class.
    
    all_df: A DataFrame that contains the following columns student name, skill scores, graduation year, true_class, Grade, cluster labels. 
    
    errors_df: A DataFrame that is the output of the function create_error_df.
    
    X_train: A DataFrame that contains all of the training features. These are skills column as well as total percentage score on the placement test.
    
    y_true: A Series that contains the labels of the test set.
    
    y_pred: A Numpy array containing all of the predicted labels of the test set.
    
    which_class: Either None or a specified class. If it is not None, instead of outputting the average scores of the errors within each class, it gives the scores of the errors of a specific class.
    
    returns: Confusion matrix of the model and a barplot of the average class scores vs. average class scores of the errors.
    '''
    print('-'*20+'F1-score: {:.3f}, Precision score: {:.3f}, Recall score {:.3f}'
          .format(f1_score(y_true, y_pred, average = "weighted"),
                  precision_score(y_true, y_pred, average = "weighted"), 
                  recall_score(y_true, y_pred, average = 'weighted'))
          + '-'*20 + '\n')
    
    fig, axs = plt.subplots(1, 2, 
                            figsize = (12, 6), 
                            gridspec_kw={'width_ratios': [1, 1.7]},
                            facecolor="aliceblue",
                            linewidth=4, edgecolor="Black")
    
    cm=confusion_matrix(y_true, y_pred, labels = y_true.unique())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=y_true.unique())
    disp.plot(xticks_rotation='vertical',
              cmap='Purples', 
              ax = axs[0])
    axs[0].tick_params(labelrotation=45)
    axs[0].set_title('Confusion Matrix')
    axs[0].set_facecolor('Lavender')
    
    rel_cols = X_train.columns
    all_train = all_df.loc[X_train.index]
    math_aa_hl_train_avgs = all_train[all_train.true_class == 'Math AA HL'][rel_cols].mean()
    math_aa_sl_train_avgs = all_train[all_train.true_class == 'Math AA SL'][rel_cols].mean()
    math_ai_sl_train_avgs = all_train[all_train.true_class == 'Math AI SL'][rel_cols].mean()
    
    if which_class is not None:
        errors_avg = errors_df[errors_df.true_class == which_class][rel_cols].mean()
        avgs_df = pd.concat([math_aa_hl_train_avgs, math_aa_sl_train_avgs, math_ai_sl_train_avgs,
                             errors_avg], axis = 1)
        avgs_df.columns = ['math_aa_hl_avg', 'math_aa_sl_avg', 'math_ai_sl_avg',
                           which_class + ' class_error_avg']
    else:
        math_aa_hl_error_avg = errors_df[errors_df.true_class == 'Math AA HL'][rel_cols].mean()
        math_aa_sl_error_avg = errors_df[errors_df.true_class == 'Math AA SL'][rel_cols].mean()
        math_ai_sl_error_avg = errors_df[errors_df.true_class == 'Math AI SL'][rel_cols].mean()
        avgs_df = pd.concat([math_aa_hl_train_avgs, math_aa_sl_train_avgs, 
                             math_ai_sl_train_avgs,
                             math_aa_hl_error_avg, 
                             math_aa_sl_error_avg,
                             math_ai_sl_error_avg], axis = 1)
        avgs_df.columns = ['math_aa_hl_train_avg', 'math_aa_sl_train_avg', 
                           'math_ai_sl_train_avg',
                           'math_aa_hl_error_avg', 'math_aa_sl_error_avg',
                          'math_ai_sl_error_avg']
    avgs_df.plot(kind = 'bar', 
                 colormap = 'Blues',
                 edgecolor = 'Black', ax = axs[1])
    axs[1].set_title('Average score vs. error score', fontsize = 12)
    # axs[1].set_xticks(rotation = 45, fontsize = 10)
    axs[1].set_ylabel('Percentage score', fontsize = 11)
    axs[1].tick_params(labelrotation=45)
    axs[1].set_facecolor('Lavender')
    plt.suptitle('Model Performance By Class', fontsize = 14)
    plt.tight_layout()
    plt.show()
    
def node_sketch(y_true, y_pred, cluster, axs):
    '''
    A function that plots a diagram of the true, predicted, and cluster labels.
    
    y_true: A Series that contains the labels of the test set.
    
    y_pred: A Numpy array containing all of the predicted labels of the test set.
    
    cluster: A Series that contains the cluster labels of the test set.
    
    axs: An integer or tuple of integers that specifies the index of an axis on which to plot the graph.
    
    returns: A diagram sketch of notes that connect true, predicted, and cluster labels to their values.
    '''
    #fig, axs = plt.subplots(figsize = (6,4))
    
    From = ['True\nlabel', 'Cluster\nlabel','Predicted\nlabel']
    To = [y_true, cluster, y_pred]

    df = pd.DataFrame({'from':From,
                       'to':To})
    
    if len(np.unique(np.array([y_true, cluster, y_pred]))) == 2:
        # Define Node Positions
        pos = {'True\nlabel':(1,1),
            'Cluster\nlabel':(1,2.2),
           'Predicted\nlabel':(1,3.4),
            y_true:(3,1.5),
            y_pred:(3,2.7)}

        # Define Node Colors
        NodeColors = {'True\nlabel':'royalblue',
                      'Cluster\nlabel':'paleturquoise',
                      'Predicted\nlabel':'lightcyan',
                      y_true:'deepskyblue',
                      y_pred:'skyblue'}
    elif len(np.unique(np.array([y_true, cluster, y_pred]))) == 3:
        # Define Node Positions
        pos = {'True\nlabel':(1,1),
               'Cluster\nlabel':(1,2.2),
               'Predicted\nlabel':(1,3.4),
               y_true:(3,1),
               cluster: (3, 2.2),
            y_pred:(3,3.4)}

        # Define Node Colors
        NodeColors = {'True\nlabel':'royalblue',
                      'Cluster\nlabel':'paleturquoise',
                      'Predicted\nlabel':'lightcyan',
                      y_true:'deepskyblue',
                      cluster: 'powderblue',
                      y_pred:'skyblue'}
        
    Labels = {}
    i = 0
    for a in From:
        Labels[a]=a
    for i in To:
        Labels[i]=i


    # Build your graph. Note that we use the DiGraph function to create the graph! This adds arrows
    G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph(ax = axs) )

    # Define the colormap and set nodes to circles, but the last one to a triangle
    Circles = []
    Colors_Circles = []
    for n in G.nodes:
        Circles.append(n)
        Colors_Circles.append(NodeColors[n])

    # By making a white node that is larger, I can make the arrow "start" beyond the node
    nodes = nx.draw_networkx_nodes(G, pos, 
                           nodelist = Circles,
                           node_size=3e3,
                           node_shape='o',
                           node_color='white',
                           alpha=1, ax = axs)

    nodes = nx.draw_networkx_nodes(G, pos, 
                           nodelist = Circles,
                           node_size=3e3,
                           node_shape='o',
                           node_color=Colors_Circles,
                           edgecolors='black',
                           alpha=0.5, ax = axs)


    nx.draw_networkx_labels(G, pos, Labels, font_size=12, ax = axs)

    # Again by making the node_size larer, I can have the arrows end before they actually hit the node
    edges = nx.draw_networkx_edges(G, pos, node_size=9e3,
                                   arrowstyle='->',width=2, ax = axs)

    axs.set_xlim(0,4)
    axs.set_ylim(0,4)
    axs.set_facecolor('Lavender')
    axs.set_title('Predictions')

def analyze_errors_idx(all_df, error_df, idx):
    '''
    A function that outputs the following:
        1. Printout of the predicted label and its corresponding probability.
        2. Printout of the subject/grade of the exam that the student took.
        3. A node sketch of the predicted, cluster, and true labels.
        4. A bargraph that compares the students scores in the skills compared to the class averages in those skills.
    
    all_df: A DataFrame that contains the following columns student name, skill scores, graduation year, true_class, Grade, cluster labels. 
    
    errors_df: A DataFrame that is the output of the function create_error_df.
    
    idx: The integer index of an error.
    
    returns: Nothing other than the above stated graphs.
    '''
    student_data = error_df.loc[idx]
    print('\nThe model predicted {} with probability {:.3f}.\n'
          .format(student_data.pred_label, student_data.pred_prob))
    actual_class_grades = all_df.loc[idx][['Subject', 'Level', 'Grade']]
    print('The student took their exam in {} {} and received a grade of {}.\n'
          .format(actual_class_grades.Subject, 
                  actual_class_grades.Level, 
                  int(actual_class_grades.Grade)))
    #Visualize the cluster labels vs. the predicted label vs. the true label.
    fig, axs = plt.subplots(1, 2, figsize = (12, 6), 
                            gridspec_kw={'width_ratios': [1, 1.7]},
                            facecolor="Ivory",
                            linewidth=4, edgecolor="Black")
    node_sketch(student_data.true_class,
                student_data.pred_label, 
                student_data.kmeans_label, axs[0])
    #Show the confidence. Make a diagram.
    
    #Sketch a graph of the errors values compared to the the mean value of each class.
    rel_cols = ['num_ops', 'alg_exp', 'expn', 'linear', 
                'geom', 'func', 'quad', 'total_weighted_perc']
    true_label = student_data.true_class
    math_aa_hl_df = all_df[all_df.true_class == 'Math AA HL']
    math_aa_sl_df = all_df[all_df.true_class == 'Math AA SL']
    math_ai_sl_df = all_df[all_df.true_class == 'Math AI SL']
    df = pd.DataFrame([math_aa_hl_df[rel_cols].mean().values,
                       math_aa_sl_df[rel_cols].mean().values,
                       math_ai_sl_df[rel_cols].mean().values,
                       student_data[rel_cols].values], 
                      index = ['math_aa_hl_avg', 'math_aa_sl_avg', 
                               'math_ai_sl_df', 'student_score'], 
                      columns = rel_cols)
    df.transpose().plot(kind = 'bar', 
                        colormap = 'Blues',
                       edgecolor = 'Black', ax = axs[1])
    axs[1].set_ylabel('Percentage', fontsize = 12)
    axs[1].set_facecolor('Lavender')
    axs[1].tick_params(labelrotation=45)
    axs[1].set_title('Student average vs. class averages')
    plt.suptitle(f'Errors for Student IDX: {idx}', fontsize = 14)
    plt.tight_layout()


class predict_confidence_intervals(BaseEstimator, TransformerMixin):
    '''
    A class that creates confidence intervals for each prediction of the model. This is done through bootstrapping and finding quantiles of the bootstrapped predictions.
    '''
    def __init__(self, estimator, num_bootstraps = 100):
        '''
        estimator: An initialized sklearn classifier that has an attribute of predict_proba.
        
        num_bootstraps: The number of times to train the model and collect predictions. Default is set to 100. Must be an integer value.
        '''
        self.estimator = estimator
        self.num_bootstraps = num_bootstraps
        
    def fit(self, X, y, boot_size = None):
        '''
        A function that trains a base model as well as num_bootstraps number of models on samples of the training data.
        
        X: DataFrame of training features.
    
        y: Series of training labels.
        
        boot_size: The number of samples to train on for each model.
        '''
        self.main_model = clone(self.estimator)
        self.main_model.fit(X, y)
        # Fit number of models according to number of bootstrapped models.
        self.bootstrapped_models = []
        if boot_size == None:
            boot_size = len(X)
        for i in range(self.num_bootstraps):
            boot_strapped_model = clone(self.estimator)
            sample_idxs = np.random.choice(np.arange(len(X)), boot_size)
            X_sampled = X.iloc[sample_idxs]
            y_sampled = y.iloc[sample_idxs]
            self.bootstrapped_models.append(boot_strapped_model.fit(X_sampled, y_sampled))
        return self
    
    def predict(self, X, c_i = 95, plot = False):
        '''
        A function that takes the training data and confidence intervals and returns a plot of the confidence intervals as well as a DataFrame that contains the pred_proba, lower and upper confidence intervals, lower confidence interval labels, predicted labels, and true labels.
        
        X: DataFrame of training features.
    
        y: Series of training labels.
        
        c_i: The confidence interval. Can be an integer or float value. The number represents the percentagel.
        
        plot: If set to true, the function plots the predicted probabilities of each sample with their confidence intervals. Default is set to False.
        
        returns: DataFrame of confidence interval information as stated above.
        '''
        predicted_label = self.main_model.predict(X)
        predicted_probabilities = self.main_model.predict_proba(X)
        predicted_label_idxs = predicted_probabilities.argmax(axis = 1)
        predicted_label_prob = predicted_probabilities.max(axis = 1)
        boot_strapped_predictions = np.zeros((self.num_bootstraps, len(X), 3))
        boot_max_class_predictions = np.zeros((self.num_bootstraps, len(X)))
        
        for i in range(self.num_bootstraps):
            probabilities = self.bootstrapped_models[i].predict_proba(X)
            boot_strapped_predictions[i] = probabilities
            boot_max_class_predictions[i]= probabilities[np.arange(0, len(X)), predicted_label_idxs]
        
        quant = c_i/100
        lower_bound_all_class = np.quantile(boot_strapped_predictions, ((1-quant)/2), axis = 0)
        upper_bound_all_class = np.quantile(boot_strapped_predictions, ((1-quant)/2)+quant, axis = 0)
        confidence_intervals_all_class = np.array((lower_bound_all_class,upper_bound_all_class)).T
        lower_bound_max_class = np.quantile(boot_max_class_predictions, ((1-quant)/2), axis = 0)
        upper_bound_max_class = np.quantile(boot_max_class_predictions, ((1-quant)/2)+quant, axis = 0)
        confidence_intervals_max_class = np.array((lower_bound_max_class,upper_bound_max_class)).T
       
        lower_conf_max_label_idxs = self.find_nearest_idx(boot_max_class_predictions,
                                                         lower_bound_max_class)
        lower_conf_label = boot_strapped_predictions[lower_conf_max_label_idxs, 
                                                     np.arange(boot_strapped_predictions.shape[1]), :].argmax(axis = 1)
        lower_conf_named_label = self.main_model.classes_[lower_conf_label.reshape(len(X), 1)]
        confidence_int_df = pd.DataFrame({'pred_prob':predicted_label_prob,
                                         'lower_ci_val':confidence_intervals_max_class[:, 0],
                                          'upper_ci_val': confidence_intervals_max_class[:, 1],
                                         'pred_label': predicted_label,
                                         'lower_conf_label': lower_conf_named_label.reshape(-1)}, 
                                         index = X.index)
        if plot:
            plt.figure(figsize = (8, 6), facecolor = 'Beige', edgecolor = 'Black')
            print('{:.2%} of the predictions were predicted with probability above 0.5.\n\n'
                  .format((predicted_label_prob>0.5).mean()))
            print('{:.2%} of the lower boundaries of the confidence intervals had probabilities above 0.5.\n\n'
                  .format((confidence_int_df.lower_ci_val>0.5).mean()))
            lower_error = predicted_label_prob - lower_bound_max_class
            upper_error = upper_bound_max_class - predicted_label_prob
            plt.errorbar(np.arange(len(predicted_label_prob)), 
                         predicted_label_prob[predicted_label_prob.argsort()], 
                         yerr = np.array((lower_error[predicted_label_prob.argsort()],
                                          upper_error[predicted_label_prob.argsort()])),
                         fmt = 'bo',
                         ecolor = 'tab:green', capsize = 2,
                         barsabove = True)
            plt.axhline(y=0.5, color='Black', linestyle='--')
            plt.xlabel('Sample')
            plt.ylabel('Probability')
            plt.title('Confidence Intervals of Predictions')
            plt.grid()
            plt.show()
        return confidence_int_df    
    
    def find_nearest_idx(self, boot_max_class, max_class_lb):
        '''
        Function that finds the index of the bootstrapped sample that is nearest to the lower confidence interval.
        
        boot_max_class: A numpy array of all the bootstrapped predictions of the predicted class.
        
        max_class_lb: A float value that represents the lower confidence interval probability value of the predicted class.
        
        returns: Index of the nearest bootstrapped sample to the lower confidence interval.
        '''
        lb_nearest_idx = np.abs(boot_max_class.T - max_class_lb.reshape(len(max_class_lb), 1)).argmin(axis = 1)
        return lb_nearest_idx

def confidence_interval_info_by_idx(idx, confidence_int_df, true_label, c_i):
    '''
    A function that prints out the predicted probability, confidene interval, predicted label, and lower confidence interval label.
    
    idx: An integer index value of a sample.
    
    confidence_int_df: A DataFrame that is the output of the predict function of the class predict_confidence_intervals.
    
    true_label: The string true label of the sample with index idx.
    
    c_i: The confidence interval value. Can be an int or float. Must represent a percentage.
    
    returns: Nothing. Just the printout as described above.
    '''
    data = confidence_int_df.loc[idx]
    print('\nPredicted probability: {:.3}\n{}% Confidence Interval: {:.3} - {:.3}\n'
          .format(data.pred_prob,
                  c_i,
                  data.lower_ci_val,
                  data.upper_ci_val))
    print('Predicted label: {}\nLower confidence interval label: {}\nTrue label: {}\n'
          .format(data.pred_label, data.lower_conf_label, true_label.loc[idx]))