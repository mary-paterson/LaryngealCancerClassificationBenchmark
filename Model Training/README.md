# Model Training

The notebook TrainModels.ipynb contains the code needed to train the models. 

Variables:

input_type - the input to the model
| Option        | Meaning       |
| ------------- |:-------------:|
| Rec_Only           | Only use the audio features as input to the model |
| Age_Sex      | Use the audio features as well as the age and sex of the patient as input to the model |
| Symptoms | Use the audio features as well as the patient's symptoms as input to the model |
| Age_Sex_Symptoms | Use the audio features as well as the patient's symptoms, age, and sex as input to the model |

audio_feature - the audio feature used as input
| Option        | Meaning       |
| ------------- |:-------------:|
| FeatureStates           | Feature vector extracted from Wav2Vec2 |
| OpenSmile      | GeMAPSv02 feature set extracted using OpenSMILE      |
| MFCC | Mel frequency cepstral coefficients       |

model_type - the algorithm used
| Option        | Meaning       |
| ------------- |:-------------:|
| SVM           | Support Vector Machine |
| MLP      | Multilayed Perceptron      |
| LR | Logistic Regression      |