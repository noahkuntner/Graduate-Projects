from src.preprocessing import *

parameters = load_parameters('parameters.yml')
#%% Text Preprocessing
train_path = parameters['preprocessing_parameters']['train_path']
test_path = parameters['preprocessing_parameters']['test_path']
data_path = parameters['preprocessing_parameters']['data_path']

preprocess_text(data_path, train_path)
preprocess_text(data_path, test_path)

print("All text preprocessed! Run run.py next!")