import numpy as np

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from confidence_intervals import evaluate_with_conf_int

import matplotlib.pyplot as plt
import seaborn as sns

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
    txt = ''
    if model_name!=None:
        txt+=f'Results for {model_name}:\n'
    
    alpha = 5 
    num_bootstraps = int(50/alpha*100)
    
    balanced_acc = evaluate_with_conf_int(y_pred, balanced_accuracy_score, y_test, num_bootstraps=num_bootstraps, alpha=alpha)
    sensitivity = evaluate_with_conf_int(y_pred, calculate_sensitivity, y_test, num_bootstraps=num_bootstraps, alpha=alpha)
    specificity = evaluate_with_conf_int(y_pred, calculate_specificity, y_test, num_bootstraps=num_bootstraps, alpha=alpha)

    txt+=f'Balanced Accuracy: {balanced_acc[0]:.3f} ({balanced_acc[1][0]:.3f}, {balanced_acc[1][1]:.3f})\n'
    txt+=f'Sensitivity: {sensitivity[0]:.3f} ({sensitivity[1][0]:.3f}, {sensitivity[1][1]:.3f})\n'
    txt+=f'Specificity: {specificity[0]:.3f} ({specificity[1][0]:.3f}, {specificity[1][1]:.3f})\n'

    if len(y_pred_prob)!=0:
        auc = evaluate_with_conf_int(y_pred_prob, roc_auc_score, y_test, num_bootstraps=num_bootstraps, alpha=alpha)
        txt+=f'AUROC: {auc[0]:.3f} ({auc[1][0]:.3f}, {auc[1][1]:.3f})\n'
        
    print(txt)
    plot_confusion_matrix(y_test, y_pred, save_file=confusion_matrix_save_file, model_name=model_name)

    if report_save_file!=None:
        with open(report_save_file, "w") as text_file:
            text_file.write(txt)

