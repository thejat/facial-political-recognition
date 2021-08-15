import glob
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense

def get_sampled_data(dataset_path, n = 1000):
  '''
  Returns the sampled data.

  Parameters:
          dataset_path (str): A string where the data in csv is saved.
          n (int): Number of samples 

  Returns:
          data (DataFrame): A sampled dataframe.
  '''

  data = pd.read_csv(dataset_path)  

  if data.shape[0] > n:
    return data.head(n)
  else:
    return data

def get_clean_data(data):
  '''
    Returns the processed data.

    The function takes in a dataframe and processes : 
      - replaces columns with NaN values with mean values 
      - Creates dummies for "object" data type columns

    Parameters:
            data (pd.DataFrame): A pandas DataFrame

    Returns:
            data (DataFrame): A processed DataFrame
    '''

  # TODO: Add docstring and example
  
  if data.shape[0] > 0:

    # TODO: is this needed? gender replacing with <unk> 

    # TODO: replacing other columns with mean values of the age and 5 big personality traits 
    for col in data.loc[:, data.isna().any()].columns:
      data[col] = data[col].fillna(data[col].mean())

    # data = data.drop(data.columns[data.isna().any()].tolist(), axis =1)

    temp_df = data.drop("pol",axis =1)
    cat_columns = temp_df.select_dtypes("object").columns
    if len(cat_columns) > 0 :
      dummy_df = pd.get_dummies(temp_df[list(cat_columns)])
      data = pd.concat([data.drop(cat_columns,axis =1 ), dummy_df], axis = 1)

      return data
    else:
      
      return data


def get_model(dimension_input):
  '''
  The function takes in input dimension to create the model architecture and 
  Returns the Neural Network Model

  Parameters:
          dimension_input (int): input dimension of the neural network

  Returns:
          model (Keras Model): Keras neural network 
  '''

  model = Sequential()
  model.add(Dense(1024, input_dim=dimension_input, activation='relu'))
  model.add(Dense(512, input_dim=1024, activation='relu'))
  model.add(Dense(256, input_dim=512, activation='relu'))
  model.add(Dense(128, input_dim=256, activation='relu'))
  model.add(Dense(60, input_dim=128, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])

  return model

def save_model(model, model_name , model_path):
  '''
  Function to save the trained models

  Parameters:
    model (Keras Model) : Fitted Model 
    model_name (str)    : Either of the models [LR, NN] for logistic models or Neural Network.
    model_path (str)    : Location where the models are saved. 

  '''
  if model_name == "LR":
    joblib.dump(model, model_path)
  elif model_name == "NN":
    model.save(model_path)
  else:
    return NotImplementedError

def fit_and_get_metrics(model_name, X_train, y_train, X_test, y_test, model_params = None, dry_run = False):

  '''
  
  '''

  if model_name == "LR":
    # TODO: LR is the same as NN, remove sklearn version and use a keras version

    if model_params is not None:
      lr = LogisticRegression(penalty = 'l1', solver = "saga", n_jobs = -1, **model_params)  
    else:
      lr = LogisticRegression(penalty = 'l1', solver = "saga", n_jobs = -1, max_iter = 100)

    if dry_run:
      return 0, 0, lr

    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_test)
    auc = round(metrics.roc_auc_score(y_test, y_pred[:,1])*100,2)
    acc = round(metrics.accuracy_score(y_test, lr.predict(X_test))*100,2)
    return auc, acc, lr
  
  elif model_name == "NN":

    model = get_model(X_train.shape[1])

    if dry_run:
      return 0, 0, model

    if model_params is not None:
      model.fit(x = X_train, y = y_train, batch_size = 1000, verbose = 1, validation_split = 0.2, **model_params)
    else:
      model.fit(epochs = 25, x = X_train, y = y_train, batch_size = 1000, verbose = 1, validation_split = 0.2)
    y_pred = model.predict(X_test)
    auc = round(metrics.roc_auc_score(y_test, y_pred)*100,2)
    _, acc = model.evaluate(X_test, y_test,batch_size=1000, verbose=0)
    return auc, round(acc*100, 2), model
  
  else:
    logger.debug(" Enter model name string as LR or NN (for Logistic Regression or Neural Network respectively).")
    return NotImplementedError
  
def get_dataframe_name(dataset_path):
  # TODO: Add docstring and example
  return "_".join(dataset_path.split("/")[-1:][0][:-4].split("_")[1:])



def get_segment_dataframe(data_dir, segment_to_run = "Canada_0_dating"):
  """ 
  Function to concat the dataframe from training. 

  # TODO: Add example

  @param : 
    - segment_to_run : The segment which needs to be trained. 

  @ returns :
    - a dataframe with data points with Country _ gender _ database.
  """

  path = data_dir + "/" + segment_to_run # use your path
  
  all_files = glob.glob(path + "/*.csv")

  li = []

  for filename in all_files:
      df = pd.read_csv(filename, index_col=None, header=0)
      li.append(df)

  return pd.concat(li, axis=0, ignore_index=True)


def save_results(results_array, location, columns):
  results_df = pd.DataFrame(results_array, columns = columns)
  results_df.to_csv(location, index=False)
  logger.debug("Results Saved.")

