import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

#Defines the demographic and symptom columns required for each of the model types
non_audio_features = {'Rec_Only':['filename'],
                     'Age_Sex':['filename', 'Age', 'Sex'],
                     'Symptoms':['filename', 'Narrow pitch range', 'Decreased volume', 'Fatigue', 'Dryness', 'Lumping', 'Heartburn', 'Choking', 
                                 'Eye dryness',  'PND', 'Smoking', 'PPD', 'Drinking', 'frequency', 'Diurnal pattern', 'Onset of dysphonia ', 
                                 'Noise at work',  'Occupational vocal demand', 'Diabetes', 'Hypertension', 'CAD', 'Head and Neck Cancer', 
                                 'Head injury', 'CVA', 'Voice handicap index - 10'],
                     'Age_Sex_Symptoms': ['filename', 'Narrow pitch range', 'Decreased volume', 'Fatigue', 'Dryness', 'Lumping', 'Heartburn', 
                                          'Choking', 'Eye dryness', 'PND', 'Smoking', 'PPD', 'Drinking', 'frequency', 'Diurnal pattern', 
                                          'Onset of dysphonia ', 'Noise at work',  'Occupational vocal demand', 'Diabetes', 'Hypertension', 'CAD', 
                                          'Head and Neck Cancer', 'Head injury', 'CVA',  'Voice handicap index - 10', 'Age', 'Sex']}

malignant_pathologies = ['Laryngeal cancer', 'Dysplasia']

SVM_param_grid ={'classifier__C': [0.1, 1, 10, 100, 1000],  
              'classifier__gamma': ['scale', 'auto', 1e-4, 1e-3, 1e-2, 1e-1, 1], 
              'classifier__degree': [2, 3, 4],
              'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid']} 

MLP_param_grid = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100), (50, 50, 50)],
    'classifier__activation': ['relu', 'tanh'],
    'classifier__solver': ['adam', 'sgd', 'lbfgs'],
    'classifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
}

LR_param_grid = {
    'classifier__penalty': ['l1', 'l2', 'elasticnet', None],        
    'classifier__C': [0.01, 0.1, 1, 10, 100],                       
    'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],  
    'classifier__max_iter': [100, 200, 300, 500],                   
    'classifier__l1_ratio': [0, 0.25, 0.5, 0.75, 1]                 
}

def format_input_dataframe(audio_features, demographics, train_files, input_type='Rec_Only', merge_col='filename', non_audio_features=non_audio_features, malignant_pathologies=malignant_pathologies):
    demographics = demographics[non_audio_features[input_type]]

    df = pd.merge(audio_features, demographics, on=merge_col, how='inner')

    train_df = df[df['filename'].isin(train_files)]
    train_df = train_df.reset_index(drop=True)

    train_df['pathology'] = train_df['pathology'].apply(lambda x: 'Malignant' if x in malignant_pathologies else 'Benign')

    return train_df

def create_preprocessor(audio_features, symptom_features, audio_preprocessing=None, symptom_preprocessing=None):

    if audio_preprocessing==None:
        # Define the preprocessing and feature selection for audio features
        audio_preprocessing = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Imputer step
            ('scaler', StandardScaler()),  # Scaling step
            ('feature_selection', SelectFromModel(DecisionTreeClassifier(random_state=42)))  # Feature Selection
        ])
    
    if symptom_preprocessing==None:
        # Define the preprocessing for symptoms and demographics if any
        symptom_preprocessing = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Imputer step
            ('scaler', StandardScaler())  # Scaling step
        ])

    # Combine the preprocessing for audio features with symptom preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('audio', audio_preprocessing, audio_features),
            ('symptoms', symptom_preprocessing, symptom_features)
        ]
    )

    return preprocessor

def create_classification_pipeline(preprocessor, class_weight_dict, model_type='SVM'):
    #Create a pipeline for the preprocessing and classifier
    if model_type=='SVM':
        pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(random_state=42, class_weight=class_weight_dict, probability=True)) 
        ])
    elif model_type=='MLP':
        pipeline = ImPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),  # SMOTE for handling class imbalance
        ('classifier', MLPClassifier(random_state=42, early_stopping=True)) 
        ])
    elif model_type=='LR':
        pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, class_weight=class_weight_dict))
        ])
    else:
        raise Exception("model_type not recognised. Please choose from one of SVM, MLP, or LR")

    return pipeline


def train_model(pipeline, X_train, y_train, param_grid=None, model_type='SVM', cv=5, verbose=3, return_train_score=True, scoring='balanced_accuracy'):
    if param_grid==None:
        if model_type=='SVM':
            param_grid=SVM_param_grid
        elif model_type=='MLP':
            param_grid=MLP_param_grid
        elif model_type=='LR':
            param_grid=LR_param_grid
        else:
            raise Exception("model_type not recognised. Please provide a parameter grid or choose a model_type from one of SVM, MLP, or LR")

    print(f'Training {model_type} \n parameter grid: {param_grid}')
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring=scoring, verbose=verbose, return_train_score=return_train_score)
    
    # Fit the classifier using the training data
    grid_search.fit(X_train, y_train)

    # Print the best parameters found 
    print(f'Best parameters: {grid_search.best_params_}')

    # Print the mean test and train score for the best parameters 
    cv_results = grid_search.cv_results_
    best_index = grid_search.best_index_
    print("Cross-validation results for the best parameters:")
    
    print(f"Mean test score: {cv_results['mean_test_score'][best_index]}")
    print(f"Mean train score: {cv_results['mean_train_score'][best_index]}")

    return grid_search
    
    