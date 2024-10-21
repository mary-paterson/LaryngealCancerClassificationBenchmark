# Imports
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import joblib
import torch
import os

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import opensmile
import librosa

format_rules = {
    'Sex': [1,2],
    'Age': 'integer',
    ('Narrow pitch range','Decreased volume','Fatigue','Dryness','Lumping','Heartburn','Choking','Eye dryness','PND','Diabetes','Hypertension','CAD','Head and Neck Cancer','Head injury','CVA'): [0,1],
    'Smoking': [0,1,2,3],
    'PPD': 'float',
    'Drinking': [0,1,2],
    'Frequency': [1,2,3],
    'Onset of dysphonia': [1,2,3,4,5],
    'Noise at work': [1,2,3],
    'Diurnal pattern': [1,2,3,4],
    'Occupational vocal demand': [1,2,3,4],
    'Voice handicap index - 10': list(range(0,41))
}

# Custom transformer for extracting audio features from file names
class Wav2Vec2BatchFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53", batch_size=32, audio_dir=''):
        self.model_name = model_name
        self.batch_size = batch_size
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.audio_dir = audio_dir  # Directory containing the audio files

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X contains only file names, so convert to full paths
        file_paths = [os.path.join(self.audio_dir, file_name) for file_name in X]

        num_samples = len(file_paths)
        feature_states = torch.empty(size=(num_samples, 512), device=self.device)

        for i in range(0, num_samples, self.batch_size):
            batch_paths = file_paths[i:i + self.batch_size]

            # Load audio files
            batch_signals = []
            for file_path in batch_paths:
                signal, _ = librosa.load(file_path, sr=16000)
                batch_signals.append(signal)

            # Process signals with Wav2Vec2
            inputs = self.feature_extractor(batch_signals, return_tensors="pt", sampling_rate=16000, padding=True)
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values=input_values)

            sig_feature_state = torch.mean(outputs.extract_features, axis=1)
            feature_states[i:i + self.batch_size] = sig_feature_state

        # Convert to DataFrame
        columns = list(map(str, range(512)))
        feature_df = pd.DataFrame(feature_states.cpu().numpy(), dtype=np.float64, columns=columns)

        return feature_df

class MFCCFeatureExtractor(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, file_paths):
        features_df = pd.DataFrame(columns=['mfcc'])
        for file in file_paths:
            y, sr = librosa.load(file, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            features_df.loc[len(features_df)] = [mfcc]
        
        return features_df

class OpenSmileFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Extract features for each audio file and convert to DataFrame
        features_list = [self.smile.process_file(file_path) for file_path in X]
        # Concatenate all extracted features into a single DataFrame
        features_df = pd.concat(features_list, axis=0, ignore_index=True)
        return features_df

def check_features(demographics, format_rules):

    invalid_columns = {}
    
    nan_columns = []

    flattened_rules = {}
    # Expand format_rules to handle tuples of columns
    for columns, expected_format in format_rules.items():
        if isinstance(columns, (tuple, list)):  # If the key is a tuple or list, iterate over it
            for column in columns:
                flattened_rules[column] = expected_format
        else:
            flattened_rules[columns] = expected_format

    demographic_columns = list(flattened_rules.keys())
    required_columns = list(set(demographic_columns) & set(demographics.columns))
    
    flattened_rules = {key: flattened_rules[key] for key in required_columns if key in flattened_rules}
    
    # Check each column
    for column, expected_format in flattened_rules.items():
        #Check for Nans
        if demographics[column].isna().any():
            nan_columns.append(column)
            
        if type(expected_format) == str:
            if expected_format == "integer":
                if not pd.api.types.is_integer_dtype(demographics[column]):
                    invalid_columns[column] = "Expected integer"
            elif expected_format == "float":
                if not pd.api.types.is_float_dtype(demographics[column]):
                    invalid_columns[column] = "Expected float"
        else:
            if not demographics[column].isin(expected_format).all():
                if len(expected_format) > 2:
                    invalid_columns[column] = f'Expected values: {min(expected_format)}-{max(expected_format)}'
                else:
                    invalid_columns[column] = f'Expected values: {expected_format[0]}/{expected_format[1]}'

    # Return results: invalid formats and columns with NaN
    result = {}
    if invalid_columns:
        result['invalid_format_columns'] = invalid_columns
    if nan_columns:
        result['The following columns contain NaN which will be replaced with 0'] = nan_columns

    print(result) if result else print('All features are as expected')

class CombinedFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, audio_column='filename', include_columns=None, audio_dir='', format_rules=None, feature=None):
        self.audio_column = audio_column
        self.include_columns = include_columns  # Columns to include from the DataFrame
        self.audio_dir = audio_dir
        self.format_rules = format_rules  # Rules for validating the columns
        self.feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if the included columns follow the format rules before proceeding
        print('Checking data format...')
        symptom_cols = list(set(X.columns)&set(self.include_columns))
        check_features(X[symptom_cols], self.format_rules)  # Call the validation function
        
        # Extract filenames from the DataFrame
        file_paths = X[self.audio_column].tolist()
        file_paths = [f'{self.audio_dir}/{file}.wav' for file in file_paths]
        
        num_samples = len(file_paths)

        # Extract the demographic and symptom features that need to be included
        included_features = X.loc[:, X.columns.isin(self.include_columns)]

        # Extract audio features
        print('Extracting features...')
        if self.feature == 'FeatureStates':
            audio_feature_extractor = Wav2Vec2BatchFeatureExtractor(audio_dir=self.audio_dir)
        elif self.feature == 'MFCC':
            audio_feature_extractor = MFCCFeatureExtractor()
        elif self.feature == 'OpenSmile':
            audio_feature_extractor = OpenSmileFeatureExtractor()
        else:
            print('Feature extractor not found. Ensure feature is one of FeatureStates, MFCC, or OpenSmile')
            
        audio_features = audio_feature_extractor.transform(file_paths)

        # Combine audio features with the included demographic and symptom features
        combined_features = pd.concat([audio_features, included_features.reset_index(drop=True)], axis=1)
        
        print('Making predictions...')
        return combined_features


def predict(test_files, trained_pipeline, audio_dir, audio_col, feature, format_rules=format_rules):
    input_features = trained_pipeline.feature_names_in_
    
    combined_feature_extractor = CombinedFeatureExtractor(
    audio_dir=audio_dir,
    audio_column=audio_col,
    include_columns=input_features,
    format_rules=format_rules,
    feature=feature 
    )

    # Define the pipeline
    new_pipeline = Pipeline([
        ('features', combined_feature_extractor),
        ('existing', trained_pipeline)  # This is your classifier pipeline
    ])

    return new_pipeline.predict(test_files)