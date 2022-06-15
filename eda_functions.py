import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

def plot_value_counts(df, col, color, rotation = 0):
    '''
    A function that plots a bar graph of value counts for a particular column.
    
    df: DataFrame
    
    col: A column name given as a string.
    
    color: A string for the name of a matplotlib color
    
    rotation: Angle of rotation of the ticks on the graph.
    
    returns: Nothing. Plots a barplot of value counts.
    '''
    fig, axs = plt.subplots()
    (df[col].value_counts(normalize = True)
     .plot(kind = 'bar', color = color, edgecolor = 'Black'))
    axs.set_title(f'Percentage of students by {col}')
    axs.set_ylabel('Percentage of students')
    axs.grid(color = 'grey')
    axs.tick_params(rotation = rotation)
    axs.set_facecolor('lightgrey')
    plt.show()

def percentage_scores_by_question_subplts(df, cols, level_number, figsize, number_cols):
    '''
    This function graphs subplots of score distributions for selected
    questions
    
    df: A DataFrame that contains the scores of each student on their math placement test.
    
    level_number: An integer that indicates the difficulty level of the question. THe higher, the more difficult the question.
    
    figsize: A tuple of integer or float values for the size of a matplotlib figure.
    
    number_cols: An integer value that denotes the number of columns to have in a subplot.
    
    returns: Nothing. Output is a series of subplots of barplots of the scores on each question on the math placement test.
    '''
    
    if len(cols) % number_cols == 0:
        number_rows = len(cols)//number_cols
    else:
        number_rows = len(cols)//number_cols + 1
    fig, axs = plt.subplots(number_rows, number_cols, figsize = figsize)
    for idx, col in enumerate(cols):
        axs_idx = (idx // number_cols, idx % number_cols)
        (df[col].value_counts(normalize=True)
         .rename('percent').reset_index()
         .rename(columns = {'index':'score'})
         .pipe((sns.barplot,'data'), x = 'score', y='percent', edgecolor = 'black', ax = axs[axs_idx]))
        axs[axs_idx].set_title(col)
        axs[axs_idx].set_xlabel('score')
        axs[axs_idx].set_ylabel('Percentage')
        axs[axs_idx].grid(color = 'grey')
        axs[axs_idx].tick_params(rotation = 0)
        axs[axs_idx].set_facecolor('whitesmoke')
    fig.suptitle(f'Level {level_number} Questions', fontsize = 16) 
    plt.tight_layout()
    plt.show()

def display_question_distributions(df, figsize, number_cols):
    '''
    This function plots subplots of score distributions for questions of each
    level
    
    df: A DataFrame that contains the scores of each student on their math placement test.
    
    figsize: A tuple of integer or float values for the size of a matplotlib figure.
    
    number_cols: An integer value that denotes the number of columns to have in a subplot.
    
    returns: Nothing. Output is 3 subplots of barplots of scores on each question on the math placement test.
    '''
    numerical_cols = df.columns[df.dtypes != 'object']
    level_1_cols = [x for x in numerical_cols if int(x.split('_')[-1]) == 1]
    level_2_cols = [x for x in numerical_cols if int(x.split('_')[-1]) == 2]
    level_3_cols = [x for x in numerical_cols if int(x.split('_')[-1]) == 3]
    for idx, col_level in enumerate([level_1_cols, level_2_cols, level_3_cols]):
        percentage_scores_by_question_subplts(df, 
                                              col_level,
                                              idx + 1, 
                                              figsize, 
                                              number_cols)