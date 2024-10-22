import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer

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

    
        