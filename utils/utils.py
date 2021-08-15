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
  
  # TODO: Add docstring and example

  data = pd.read_csv(dataset_path)  

  if data.shape[0] > n:
    return data.head(n)
  else:
    return data

def get_clean_data(data):

  # TODO: Add docstring and example
  
  if data.shape[0] > 0:

    # TODO: is this needed? gender replacing with <unk> 

    # TODO: replacing other columns with mean values of the age and 5 big personality traits 
    for col in data.loc[:, data.isna().any()].columns:
      data[col] = data[col].fillna(data[col].mean())

    data = data.drop(data.columns[data.isna().any()].tolist(), axis =1)

    temp_df = data.drop("pol",axis =1)
    cat_columns = temp_df.select_dtypes("object").columns
    if len(cat_columns) > 0 :
      dummy_df = pd.get_dummies(temp_df[list(cat_columns)])
      data = pd.concat([data.drop(cat_columns,axis =1 ), dummy_df], axis = 1)

      return data
    else:
      
      return data


def get_model(dimension_input):

  # TODO: Add docstring and example


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
  if model_name == "LR":
    joblib.dump(model, model_path)
  elif model_name == "NN":
    model.save(model_path)
  else:
    return NotImplementedError

def fit_and_get_metrics(data, model_name, dry_run = False):

  # TODO: Add docstring and example


  y = data['pol'].replace({"liberal": 1, "conservative": 0})
  X = data.drop('pol', axis = 1)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
  # TODO: check reproducibility of splits so that additional metrics computed post-hoc
  # are consistent. See https://stackoverflow.com/questions/53182821/scikit-learn-train-test-split-not-reproducible

  if model_name == "LR":
    # TODO: LR is the same as NN, remove sklearn version and use a keras version
    # assert 'pol' in data.columns
    lr = LogisticRegression(penalty='l1', solver="saga")

    if dry_run:
      return 0, 0, lr

    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_test)
    auc = round(metrics.roc_auc_score(y_test, y_pred[:,1])*100,2)
    acc = round(metrics.accuracy_score(y_test, lr.predict(X_test))*100,2)
    return auc, acc, model
  
  elif model_name == "NN":
    # assert 'pol' in data.columns

    model = get_model(X_train.shape[1])

    if dry_run:
      return 0, 0, model

    model.fit(epochs=25, x = X_train, y = y_train, batch_size = 1000, verbose = 0, validation_split = 0.2)
    y_pred = model.predict(X_test)
    auc = round(metrics.roc_auc_score(y_test,y_pred)*100,2)
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

  path = DATA_DIR + "/" + segment_to_run # use your path
  
  all_files = glob.glob(path + "/*.csv")

  li = []

  for filename in all_files:
      df = pd.read_csv(filename, index_col=None, header=0)
      li.append(df)

  return pd.concat(li, axis=0, ignore_index=True)


def save_results(results_array, location):
  results_df = pd.DataFrame(results_array, columns = ["Group Name", "Model", "Feature Set", "Test AUC", "Test Accuracy"])
  results_df.to_csv(location, index=False)
  logger.debug("Results Saved.")

