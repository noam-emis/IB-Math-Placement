import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

def create_level(subject):
    '''
    A function that takes IB results of 2021 and extracts the level of the class from the subject.
    
    subject: String of subject name.
    
    returns: String of subject level. Can be either 'SL' or 'HL'.
    '''
    if 'HL' in subject:
        return 'HL'
    else:
        return 'SL'
    

def label_class(row):
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
    elif row.Subject == 'Math AA':
        if int(row.Grade) >=4:
            return 'Math AA SL'
        else:
            return 'Math AI SL' 
    else:
        if int(row.Grade) ==7:
            return 'Math AA SL'
        else:
            return 'Math AI SL'
        
def create_col_list(skill, df):
    '''
    This function creates a list of columns that correspond to a particular 
    math skill.
    
    skill: String name for a math skill.
    
    df: DataFrame of math placement test questions and results.
    
    returns: Column names that correspond to a skill.
    '''
    skill_list = [x for x in df.columns if skill in x]
    return skill_list

def create_weighted_skill(row, skill_col_list):
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

def create_total_perc(row, columns):
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

def create_list_level_cols(question_cols, level_num):
    '''
    This function creates a list of column names corresponding to a particular level.
    
    question_cols: A list of column names that are broken down by skill, question number, and weight.
    
    level_num: An integer from 1-3 representing the difficulty level of the question.
    
    returns: A list of column names corresponding to a particular question level.
    '''
    level_list = [col for col in question_cols 
                  if int(col.split('_')[-1]) == level_num]
    return level_list

def average_by_level(row, question_cols, level_num):
    '''
    This function that takes a row from the math placement test df and a level number, and returns the
    average grade for that level.
    
    row: A row from the math placement data, DataFrame.
    
    question_cols: A list of column names that are broken down by skill, question number, and weight.
    
    level_num: An integer from 1-3 representing the difficulty level of the question.
    
    returns: Average score for the given level.
    '''
    level_list = create_list_level_cols(question_cols, level_num)
    level_sum = 0
    for col in level_list:
        level_sum+=row[col]
    return level_sum/len(level_list)


def create_clean_math_df(year_df):
    '''
    This function cleans the math placement dataframe. It combines skills into one, and gives
    each skill a weighted average. It also calculates a total weighted average for the exam.
    
    year_df: The math placement results DataFrame from a particular year.
    
    returns: A 'clean' dataframe that calculates weighted averages for each skill and calculates a total weighted average for the placement test. It drops columns such as 'Student' and 'Currently in. It also calculates an average score by level.'    
    '''
    question_cols = year_df.drop(['Currently in', 'Recommended Course'], axis = 1).columns
    df_clean = pd.DataFrame(year_df.apply(create_total_perc, axis = 1, args = (question_cols,)),
                            index = year_df.index, columns = ['total_weighted_perc'])
    skills = np.unique(np.array(['_'.join(col.split('_')[:-2]) for col in question_cols]))
    for skill in skills:
        col_list= create_col_list(skill, year_df)
        df_clean[skill] = year_df.apply(create_weighted_skill, 
                                        axis = 1, 
                                        args = (col_list,))
    for level in [1, 2, 3]:
        df_clean['level_' + str(level) + '_avg'] = year_df.apply(average_by_level, args = (question_cols, level, ), axis = 1)
    return df_clean
