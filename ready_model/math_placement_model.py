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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import networkx as nx
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import clone

class prepare_clean_df(BaseEstimator, TransformerMixin):
    '''
    A class that prepares both training and testing data for the model.
    
    Initialized with original train data as a sample so that columns can be set according to the train data that was used for the model.
    
    Data is transformed via the "transform" method.
    
    '''
    def __init__(self, X_train_df, y_train):
        '''
        X_train_df: A DataFrame that has been imported as a pickle file, that was used to train the model.
        y_train: A Series that has the class labels for 
        '''
        self.X_train_df = X_train_df
        self.y_train = y_train
        self.skills = [x for x in X_train_df.columns if 'total' not in x.split('_') and 'level' not in x.split('_')]
        
        
    def create_col_list(self, skill, df):
        '''
        This function creates a list of columns that correspond to a particular 
        math skill.

        skill: String name for a math skill.

        df: DataFrame of math placement test questions and results.

        returns: Column names that correspond to a skill.
        '''
        skill_list = [x for x in df.columns if skill in x]
        return skill_list

    
    def create_weighted_skill(self, row, skill_col_list):
        '''
        This function takes a weighted average of the points earned within a 
        given skill. 
        If the skill was level one difficulty, it is given a weight of 1. If the skill was a level 
        two, it is given a weight of 2, etc.

        row: A row from the math placement results DataFrame.

        skill_col_list: A list of column names that correspond to a particular sill.

        returns: A weighted average score of questions on the skill.
        '''
        total_weights = 0
        weighted_sum = 0
        for i, skill in enumerate(skill_col_list):
            weight = int(re.split('_', skill)[-1])
            total_weights += weight
            weighted_sum += row[skill] * weight
        return weighted_sum/total_weights

    
    def create_total_perc(self, row, columns):
        '''
        This function calculates a total percentage of the placement test.
        Weights are given to each question based on the difficulty level. Thus a 
        weighted average is calculated.

        row: A row from the math placement results DataFrame.

        columns: The column names of the questions.

        returns: Total weighted percentage for the placement test.
        '''
        total_weights = 0
        weighted_sum = 0
        for col in columns:
            weight = int(re.split('_', col)[-1])
            total_weights += weight
            weighted_sum += row[col] * weight
        return weighted_sum / total_weights

    
    def create_list_level_cols(self, question_cols, level_num):
        '''
        This function creates a list of column names corresponding to a particular level.

        question_cols: A list of column names that are broken down by skill, question number, and weight.

        level_num: An integer from 1-3 representing the difficulty level of the question.

        returns: A list of column names corresponding to a particular question level.
        '''
        level_list = [col for col in question_cols 
                      if int(col.split('_')[-1]) == level_num]
        return level_list

    
    def average_by_level(self,row, question_cols, level_num):
        '''
        This function that takes a row from the math placement test df and a level number, and returns the
        average grade for that level.

        row: A row from the math placement data, DataFrame.

        question_cols: A list of column names that are broken down by skill, question number, and weight.

        level_num: An integer from 1-3 representing the difficulty level of the question.

        returns: Average score for the given level.
        '''
        level_list = self.create_list_level_cols(question_cols, level_num)
        level_sum = 0
        for col in level_list:
            level_sum+=row[col]
        return level_sum/len(level_list)


    def create_clean_math_df(self, year_df):
        '''
        This function cleans the math placement dataframe. It combines skills into one, and gives
        each skill a weighted average. It also calculates a total weighted average for the exam.

        year_df: The math placement results DataFrame from a particular year.

        returns: A 'clean' dataframe that calculates weighted averages for each skill and calculates a total weighted average for the placement test. It drops columns such as 'Student' and 'Currently in. It also calculates an average score by level.'    
        '''
        question_cols = year_df.drop(['Currently in'], axis = 1).columns
        df_clean = pd.DataFrame(year_df.apply(self.create_total_perc, axis = 1, args = (question_cols,)),
                                index = year_df.index, columns = ['total_weighted_perc'])
        for skill in self.skills:
            col_list= self.create_col_list(skill, year_df)
            df_clean[skill] = year_df.apply(self.create_weighted_skill, 
                                            axis = 1, 
                                            args = (col_list,))
        for level in [1, 2, 3]:
            df_clean['level_' + str(level) + '_avg'] = year_df.apply(self.average_by_level, args = (question_cols, level, ), axis = 1)
        return df_clean
    
    
    def label_class(self, row):
        '''
        A function that assigns a true class to each sample based on the subject that they took and their grade on their final exams.

        row: A row from the ib_results DataFrame.

        returns: True class label. Could be either Math HL, Math SL, or Math Studies.
        '''
        if row.Level == 'HL':
            if int(row.Grade) >=5:
                return 'Math AA HL'
            else:
                return 'Math AA SL'
        elif 'MATHEMATICS ANALYSIS' in row.Subject:
            if int(row.Grade) >=4:
                return 'Math AA SL'
            else:
                return 'Math AI SL' 
        else:
            if int(row.Grade) ==7:
                return 'Math AA SL'
            else:
                return 'Math AI SL'
    
    
    def create_clean_ib_label(self, y_file_path):
        '''
        Function that turns IB results into a Series containing the "True Label".
        
        y_file_path: Absolute path of the IB results file.
        
        returns: Series of the true label.
        '''
        ib_results = pd.read_csv(y_file_path)
        ib_results.set_index('Name', inplace = True)
        ib_results.index.rename('Student', inplace = True)
        ib_results = ib_results[['Year', 'Subject', 'Level', 'Grade']].copy()
        ib_results.dropna(axis = 0, how = 'any', inplace = True)
        ib_results = ib_results[ib_results.Subject.str.contains('MATH')]
        ib_results = ib_results[ib_results.Level != 'EE']
        ib_results = ib_results[ib_results.Grade != 'N']
        ib_results['Grade'] = ib_results['Grade'].astype('int')
        clean_y = pd.Series(ib_results.apply(self.label_class, axis = 1), 
                            index = ib_results.index,
                           name = 'true_label')
        return clean_y
    
    def transform(self, x_file_path, y_file_path = None, train_test = 'test'):
        '''
        A function that transforms placement data and IB results data into data that can be run through the model.
        
        x_file_path: Absolute file path (as a string) for the math placement data to be used.
        y_file_path: Absolute file path (as a string) for the IB results to be used in trainig. 
        Default is None, since this is only needed if it is transforming training data.
        
        train_test: Either "train" or "test". If "train", the function will add the data to the existing training data. If "test", the function will prepare the math placement data for the model.
        
        returns: Clean X_train, y_train data if train_test is set to true. Otherwise, it outputs a clean X_test df.
        '''
        df = pd.read_csv(x_file_path).dropna(axis = 0, how = 'any')
        df.set_index('Student', inplace = True)
        
        # Clean the X data.
        df_clean = self.create_clean_math_df(df)
        if train_test == 'train':
            if y_file_path is None:
                return 'ERROR: Cannot update train data without a file path for labels.'
            else:
                #Clean the y_data.
                y_clean = pd.DataFrame(self.create_clean_ib_label(y_file_path))
                total_new_train = df_clean.merge(y_clean, 
                                           left_index = True,
                                           right_index = True,
                                           how = 'inner')
                new_y = pd.Series(total_new_train['true_label'])
                new_x = total_new_train.drop(['true_label'], axis = 1)
                
                #Combine pre-existing train data with new train data.
                y_train = pd.concat([self.y_train, new_y], axis = 0)
                X_train = pd.concat([self.X_train_df, new_x], axis = 0)
            return X_train, y_train
        elif train_test == 'test':
            return df_clean
        else:
            return 'ERROR: train_test must be specified as either "train" or "test".'

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
        
class math_placement_prediction_model(BaseEstimator, TransformerMixin):
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
    
    
    def predict(self, X, c_i = 95, plot = True):
        '''
        A function that takes the training data and confidence intervals and returns a plot of the confidence intervals as well as a DataFrame that contains the pred_proba, lower and upper confidence intervals, lower confidence interval labels, predicted labels, and true labels.
        
        X: DataFrame of test features.
        
        c_i: The confidence interval. Can be an integer or float value. The number represents the percentagel.
        
        plot: If set to true, the function plots the predicted probabilities of each sample with their confidence intervals. Default is set to False.
        
        returns: DataFrame of confidence interval information as stated above.
        '''
        self.c_i = c_i
        self.X_test = X
        
        #Gather the model predictions.
        predicted_label = self.main_model.predict(X)
        predicted_probabilities = self.main_model.predict_proba(X)
        
        #Finding the index and values of the maximum predicted probability.
        predicted_label_idxs = predicted_probabilities.argmax(axis = 1)
        predicted_label_prob = predicted_probabilities.max(axis = 1)
        
        #Saving the results of each of the n_bootstraps models.
        boot_strapped_predictions = np.zeros((self.num_bootstraps, len(X), 3))
        boot_max_class_predictions = np.zeros((self.num_bootstraps, len(X)))
        
        for i in range(self.num_bootstraps):
            probabilities = self.bootstrapped_models[i].predict_proba(X)
            boot_strapped_predictions[i] = probabilities
            #Saving the maximum predicted probabilities of each sample for each bootstrapped model.
            boot_max_class_predictions[i]= probabilities[np.arange(0, len(X)), predicted_label_idxs]
        
        quant = c_i/100
        
        #Gathering the confidence intervals for each sample for each class.
        lower_bound_all_class = np.quantile(boot_strapped_predictions, ((1-quant)/2), axis = 0)
        upper_bound_all_class = np.quantile(boot_strapped_predictions, ((1-quant)/2)+quant, axis = 0)
        confidence_intervals_all_class = np.array((lower_bound_all_class,upper_bound_all_class)).T
        
        #Gathering the confidence intervals for only the maximum predicted class for each sample.
        lower_bound_max_class = np.quantile(boot_max_class_predictions, ((1-quant)/2), axis = 0)
        upper_bound_max_class = np.quantile(boot_max_class_predictions, ((1-quant)/2)+quant, axis = 0)
        confidence_intervals_max_class = np.array((lower_bound_max_class,upper_bound_max_class)).T
       
        #Finding the index of the models that brought the lower confidence probability.
        lower_conf_max_label_idxs = self.find_nearest_idx(boot_max_class_predictions,
                                                         lower_bound_max_class)
        
        #Finding the label with the maximum probability when the predicted class had its lower confidence probability.
        lower_conf_label = boot_strapped_predictions[lower_conf_max_label_idxs, 
                                                     np.arange(boot_strapped_predictions.shape[1]), :].argmax(axis = 1)
        lower_conf_named_label = self.main_model.classes_[lower_conf_label.reshape(len(X), 1)]
        
        #Creating a DataFrame the provides for each sample the predicted probability, probability confidence intervals, 
        #and labels of predicted and lower confidence probabilities.
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
        self.confidence_int_df = confidence_int_df
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

    
    def confidence_interval_info_by_idx(self, idx, confidence_int_df , true_label = None):
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
                      self.c_i,
                      data.lower_ci_val,
                      data.upper_ci_val))
        if true_label is None:
            print('Predicted label: {}\nLower confidence label: {}\n'
                  .format(data.pred_label, data.lower_conf_label))
        else:
            print('Predicted label: {}\nLower confidence label: {}\nTrue label: {}\n'
                  .format(data.pred_label, data.lower_conf_label, true_label.loc[idx]))
            
            
    def c_i_widget(self, y_test = None):
        '''
        A function that creates a widget that provides predictions, predicted probabilities, and lower confidence labels for each sample by index.
        
        y_test: A Series of true labels for the data.
        
        returns: A widget with the output from "confidence_interval_info_by_idx".
        '''
        interact(self.confidence_interval_info_by_idx,
                 idx = self.X_test.index,
                 confidence_int_df = fixed(self.confidence_int_df), 
                 true_label = fixed(y_test))


def change_currently_in_label(label):
    '''
    Function that changes the 'Currently in' label that teachers entered, into the standard way of 
    referring to the label.
    
    label: A string with the 'Currently in' label.
    '''
    label = label.lower()
    if 'hl' in label:
        return 'Math AA HL'
    elif 'app' in label or 'ai' in label:
        return 'Math AI SL'
    elif 'an' or 'aa' in label:
        return 'Math AA SL'
    else:
        return 'Cannot interpret current class choice.'
    
def assign_num_to_class(clss):
    '''
    A function that assigns numerical value to each class to help determine the differences in their class
    choices and actual predicted class.
    
    clss: The class label.
    '''
    if clss == 'Math AA HL':
        return 3
    elif clss == 'Math AA SL':
        return 2
    else:
        return 1

def student_emails_strength_idx(curr, pred, lower):
    '''
    A function that determines the type of letter that needs to be sent to students, based on their
    choices and their predicted class.
    
    curr: A string that represents the student's chosen class.
    
    pred: A string that represents the student's predicted class.
    
    lower: A string that represents the student's lower confidence class.
    '''
    curr_num = assign_num_to_class(curr)
    pred_num = assign_num_to_class(pred)
    lower_label_num = assign_num_to_class(lower)
    
    if curr_num - pred_num ==2:
        if curr_num - lower_label_num == 2:
            return 'Strong'
        else:
            return 'Strong Conditional'
    elif curr_num - pred_num == 1:
        if curr_num - lower_label_num >=1:
            return 'Strong'
        else:
            return 'Moderate'
    else:
        return 'None'

def student_email_by_idx(confidence_int_df, placement_df, idx):
    '''
    A function that prints a letter for the student based on their class choices and predicted class.
    
    confidence_int_df: The predictions DF from the confidence interval model.
    
    placement_df: The cleaned placement DataFrame.
    
    idx: Student name (which is the index of the column).
    
    returns: Nothing. Prints out the chosen course, predicted course, lower confidence course, and a letter to the students.
    '''
    
    curr = placement_df.loc[idx, 'Currently in']
    pred = confidence_int_df.loc[idx, 'pred_label']
    lower_label = confidence_int_df.loc[idx, 'lower_conf_label']
    
    print('Chosen course: {}\n\nPredicted course: {}\n\nLower Confidence: {}\n'.format(curr, pred, lower_label))
    
    strength = student_emails_strength_idx(curr, pred, lower_label)
        
    print('\033[1m' + 'Email:\n' + '\033[0m')
    
    if strength == 'Strong':
        print('Dear {},\n\n\
Based on the results of your diagnostic test, we are placing you in: {}.\n\n\
Wishing you a great start to the year and much success in the Diploma Programme.\n\n\
Sincerely,\n\n\
EMIS Math Department'.format(idx, pred))
    
    elif strength == 'Strong Conditional':
        print('Dear {},\n\n\
After taking the mathematics placement exam, you have tested into {}, instead of {}. \
While this is our current recommendation, this is a significant change. \
The only alternative to {} for you, is a conditional placement in {}. \
This means that by the start of the Sukkot Holidays, you will need to have a 55% average in \
assessments in {} to stay in the course. \n\n\
Please email [IB COORDINATOR] at [EMAIL] (cc: [MATH COORDINATOR EMAIL]) \
to let her know whether you plan to take the recommendation of {} \
or whether you plan to try the conditional placement in {} until the holidays.\n\n\
Wishing you a great start to the year and much success in the Diploma Programme.\n\n\
Sincerely,\n\n\
EMIS Math Department'
              .format(idx,
                     pred,
                     curr,
                     pred,
                     lower_label,
                     lower_label,
                     pred,
                     lower_label))
    elif strength == 'Moderate':
        print('Dear {},\n\n\
After taking the mathematics placement test, you are strongly recommended to take {}.\
If you choose to stay in {}, this will be a conditional placement. \
You will need to have an average of a 55% in assessments in this by the start of theSukkot Holidays, \
in order to maintain your position in the class. Otherwise you will be moved to {}.\n\n\
Please email [IB COORDINATOR] at [EMAIL] (cc: [MATH COORDINATOR EMAIL]) \
to let her know whether you plan to take the recommendation of {} or whether you plan to try \
the conditional placement in {} until the holidays.\n\n\
Wishing you a great start to the year and much success in the Diploma Programme.\n\n\
Sincerely,\n\n\
EMIS Math Department'
              .format(idx,
                     pred,
                     curr,
                     pred,
                     pred,
                     curr))
    else:
        print('No email necessary')

def email_widget(raw_test_path, ci_predictions):
    '''
    A function that takes the raw test data and confidence interval predicts, and outputs a widget that composes emails for each student.
    
    raw_test_path: Absolute path to the test data file.
    
    ci_predictions: The predictions DF from the confidence interval model.
    '''
    placement_results = pd.read_csv(raw_test_path)
    placement_results = placement_results.set_index('Student')
    placement_results['Currently in'] = placement_results['Currently in'].apply(change_currently_in_label)
    interact(student_email_by_idx,
         confidence_int_df = fixed(ci_predictions),
         placement_df = fixed(placement_results), 
         idx = placement_results.index)