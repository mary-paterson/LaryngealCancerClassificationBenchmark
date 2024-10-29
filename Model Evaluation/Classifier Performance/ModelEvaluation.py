import numpy as np

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from confidence_intervals import evaluate_with_conf_int

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
import numpy as np

import pandas as pd

def calculate_sensitivity(y_test, y_pred):
    #Calculates sensitivity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['Benign', 'Malignant']).ravel()
    sensitivity = tp/(tp+fn)
    return sensitivity

def calculate_specificity(y_test, y_pred):
    #Calculates specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['Benign', 'Malignant']).ravel()
    specificity = tn / (tn+fp)
    return specificity

def plot_confusion_matrix(y_test, y_pred, model_name=None, save_file=None):
    #Plots a confusion matrix and saves it to the desired file
    labels=y_test.unique()
    cnf = confusion_matrix(y_test, y_pred, labels=labels)
    cnf_percent = (cnf / cnf.sum(axis=1)[:, np.newaxis])*100
    plt.figure()
    sns.heatmap(cnf_percent, annot=cnf, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels, vmin=0, vmax=100)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if model_name!= None:
        plt.title(f'Confusion Matrix\n{model_name}')
    if save_file!=None:
        plt.savefig(save_file)
    plt.show()
    return cnf

def evaluation_report(y_test, y_pred, y_pred_prob=None, report_save_file=None, model_name=None, confusion_matrix_save_file=None):
    # Initialize an empty string to hold the report text
    txt = ''  

    # Add model name to the report if provided
    if model_name != None:
        txt += f'Results for {model_name}:\n'
    
    # Set alpha for confidence interval calculation and compute the number of bootstraps
    alpha = 5  # Confidence level parameter, 5% significance
    num_bootstraps = int(50 / alpha * 100)  # Number of bootstrap samples based on alpha
    
    # Calculate balanced accuracy, sensitivity, and specificity with confidence intervals
    balanced_acc = evaluate_with_conf_int(y_pred, balanced_accuracy_score, y_test, num_bootstraps=num_bootstraps, alpha=alpha)
    sensitivity = evaluate_with_conf_int(y_pred, calculate_sensitivity, y_test, num_bootstraps=num_bootstraps, alpha=alpha)
    specificity = evaluate_with_conf_int(y_pred, calculate_specificity, y_test, num_bootstraps=num_bootstraps, alpha=alpha)

    # Append metrics and their confidence intervals to the report
    txt += f'Balanced Accuracy: {balanced_acc[0]:.3f} ({balanced_acc[1][0]:.3f}, {balanced_acc[1][1]:.3f})\n'
    txt += f'Sensitivity: {sensitivity[0]:.3f} ({sensitivity[1][0]:.3f}, {sensitivity[1][1]:.3f})\n'
    txt += f'Specificity: {specificity[0]:.3f} ({specificity[1][0]:.3f}, {specificity[1][1]:.3f})\n'

    # Calculate AUROC with confidence intervals if probability predictions are available
    if len(y_pred_prob) != 0:
        auc = evaluate_with_conf_int(y_pred_prob, roc_auc_score, y_test, num_bootstraps=num_bootstraps, alpha=alpha)
        txt += f'AUROC: {auc[0]:.3f} ({auc[1][0]:.3f}, {auc[1][1]:.3f})\n'
        
    # Print the report to the console
    print(txt)

    # Generate and save a confusion matrix plot if a save file path is provided
    plot_confusion_matrix(y_test, y_pred, save_file=confusion_matrix_save_file, model_name=model_name)

    # Save the report to a text file if a file path is specified
    if report_save_file != None:
        with open(report_save_file, "w") as text_file:
            text_file.write(txt)


def get_results_df(results_dfs, dataset_names, metric, algorithms, audio_features, input_features, 
                   ground_truth_col='ground truth', column_format='{algorithm}_{audio_feature}_{input_feature} prediction', 
                   alpha=None, num_bootstraps=None):
    
    # Initialize an empty DataFrame to store the formatted results
    results = pd.DataFrame()

    # Check if results_dfs is a list (containing multiple DataFrames to process)
    if type(results_dfs) == list:
        # Iterate over each DataFrame and corresponding dataset name in results_dfs
        for i in range(len(results_dfs)):
            df = results_dfs[i]  # Current DataFrame to format
            dataset = dataset_names[i]  # Name of the current dataset for reference
            print(f'Formatting {dataset}...')
            
            # If results is not empty, concatenate the new formatted DataFrame
            if len(results) != 0:
                # Format the current DataFrame using `format_dataframe` and temporary storage
                results_temp = format_dataframe(df, dataset, metric, algorithms, audio_features, 
                                                input_features, ground_truth_col, column_format, 
                                                alpha, num_bootstraps)
                # Concatenate the temporary formatted DataFrame with results
                results = pd.concat([results, results_temp])
            else:
                # If results is still empty, simply assign the formatted DataFrame to it
                results = format_dataframe(df, dataset, metric, algorithms, audio_features, 
                                           input_features, ground_truth_col, column_format, 
                                           alpha, num_bootstraps)
                
    # If results_dfs is a single DataFrame, format it directly without iteration
    elif type(results_dfs) == pd.core.frame.DataFrame:
        results = format_dataframe(results_dfs, dataset, metric, algorithms, audio_features, 
                                   input_features, ground_truth_col, column_format, alpha, num_bootstraps)
    else:
        raise TypeError("results_dfs should be a dataframe or a list of dataframes")
    
    return results


def format_dataframe(df, dataset, metric, algorithms, audio_features, input_features, 
                     ground_truth_col='ground truth', column_format='{algorithm}_{audio_feature}_{input_feature} prediction', 
                     alpha=None, num_bootstraps=None):
    
    # Set default alpha and num_bootstraps if not provided
    if alpha == None:
        alpha = 5
    if num_bootstraps == None:
        num_bootstraps = int(50 / alpha * 100)

    # Initialize an empty DataFrame to store formatted results
    results = pd.DataFrame(columns=audio_features)

    # Iterate over each combination of algorithm, input feature, and audio feature
    for algorithm in algorithms:
        for input_feature in input_features:
            index_value = len(results)  # Track the index for the current row in results
            
            for audio_feature in audio_features:
                # Format column name based on provided template
                column_name = column_format.format(
                    algorithm=algorithm,
                    audio_feature=audio_feature,
                    input_feature=input_feature
                )
                print(f'Processing {column_name}...') 

                # Set Input and Dataset values for the current row in results
                results.at[index_value, 'Input'] = input_feature
                results.at[index_value, 'Dataset'] = dataset
                
                # Calculate metric with confidence interval and store result under the audio feature column
                results.at[index_value, audio_feature] = evaluate_with_conf_int(
                    df[column_name], metric, df[ground_truth_col], 
                    num_bootstraps=num_bootstraps, alpha=alpha
                )

    # Set 'Dataset' and 'Input' as index columns for easier lookup and organization
    results = results.set_index(['Dataset', 'Input'])

    return results


def get_x(start, step, count):
    return [start + step * i for i in range(count)]

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def insert_breaks(s, max_length):
    result = []
    words = s.split(" ")  # Split the string into words

    current_line = ""
    
    for word in words:
        # If the word itself is longer than max_length, start a new line
        if len(word) > max_length:
            # If current_line has content, append it to the result
            if current_line:
                result.append(current_line)
                current_line = ""
            result.append(word)  # Place the long word on its own line
        else:
            # Check if adding this word would exceed the max_length
            if len(current_line) + len(word) + 1 > max_length:
                result.append(current_line)  # Append the current line to the result
                current_line = word  # Start a new line with the current word
            else:
                # If not, add the word to the current line
                if current_line:  # If not the first word in the line, add a space
                    current_line += " "
                current_line += word

    # Append any remaining content in current_line to the result
    if current_line:
        result.append(current_line)

    return "\n".join(result)


def plot_results(results, colour_map='tab10', title='', ylabel='', input_map=None, 
                 max_label_length=15, save_file=None):
    
    # Extract unique dataset names and input values from the DataFrame's multi-level index
    datasets = list(results.index.get_level_values(0).unique())  
    input_values = list(results.index.get_level_values(1).unique())  
    
    # Get a list of feature names (columns) from the results DataFrame
    features = list(results.columns)

    # Calculate parameters for layout
    num_datasets = len(datasets)
    small_gap = num_datasets / 4  
    small_step_length = num_datasets + 2 * small_gap  
    big_gap = num_datasets / 3  

    # Set up colormap
    cmap = mpl.colormaps[colour_map]
    if cmap.N > 200:
        colours = cmap(np.linspace(0, 1, num_datasets))
    else:
        colours = cmap.colors

    # Initialize variables to track x positions and tick labels
    x_start = 0
    dataset_num = 0
    x_tick_list = []  # Holds positions for x-axis tick labels
    
    fig, ax = plt.subplots()

    # Loop through each unique input value group to plot bars for each dataset
    for input_value in input_values:
        index_start = x_start  # Track the start of each input feature section
        dataset_xticks = []  # Store x-tick positions for this dataset group
        
        # Plot data for each dataset within the current input feature
        for dataset in datasets:
            
            dataset_num = datasets.index(dataset) 
            
            values = list(results.loc[dataset, input_value])  # Get metric values and CIs
            
            # Extract mean and confidence interval values
            mean_val = [x[0] for x in values] 
            lower_ci = [x[1][0] for x in values] 
            upper_ci = [x[1][1] for x in values] 

            # Calculate x positions for bars within this dataset group
            x = get_x(x_start, small_step_length, len(values))

            # Plot bars for mean values
            plt.bar(x, mean_val, width=1, color=colours[dataset_num], label=dataset)
            # Add vertical lines for confidence intervals
            plt.vlines(x=x, ymin=lower_ci, ymax=upper_ci, colors='k')
            
            # Update starting x position for next dataset
            x_start += 1
            
            # Adjust tick positioning for display
            if len(dataset_xticks) == 0:
                dataset_xticks = x
            else:
                dataset_xticks = [(x+y)/2 for x, y in zip(*[dataset_xticks, x])]
    
        x_tick_list += dataset_xticks
        
        # Calculate x position for the next input feature group
        x_start = x[-1] + big_gap + small_step_length
        
        # Center label position for input feature
        label_x = (index_start + x[-1]) / 2
        
        # Get appropriate label for input feature, inserting line breaks if necessary
        if input_map == None:
            input_label = input_value
        else:
            input_label = input_map[input_value]
        input_label = insert_breaks(input_label, max_label_length)
        
        # Display the input feature label centered below its bars
        plt.text(label_x, -0.35, input_label, ha='center')
        
    # Add legend, avoiding duplicates
    legend_without_duplicate_labels(ax)
    
    # Set x-axis ticks to feature names, repeated for each input group
    plt.xticks(x_tick_list, features * len(input_values), rotation=90)

    plt.title(title)
    plt.ylabel(ylabel)

    # Save plot to file if save path is specified
    if save_file != None:
        plt.savefig(save_file, bbox_inches='tight')
    
    # Display the plot
    plt.show()