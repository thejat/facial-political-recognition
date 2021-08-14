from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split

import glob
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense

def get_sampled_data(dataset_path, n = 1000):
  
  data = pd.read_csv(dataset_path)  

  if data.shape[0] > n:
    return data.head(n)
  else:
    return data

# def temp():
#   #TODO
#   for folder in tqdm(folders):
#   csv_files = os.listdir(DATA_DIR + folder)
#   for csv in csv_files:
#     if '.csv' in csv:
#       if DEBUG: print(DATA_DIR + "full/" + folder + "/" + csv)
#       data_sample = get_sampled_data(DATA_DIR + folder + "/" + csv)
#       data_sample.to_csv(DATA_DIR + "sample/" + folder + csv)


def get_clean_data(dataset_path, drop_col):
  
  data = pd.read_csv(dataset_path)  
  
  if data.shape[0] > 0:

    data = data.drop(drop_col, axis = 1)

    # TODO: is this needed? gender replacing with <unk> 

    # TODO: replacing other columns with mean values of the age and 5 big personality traits 
    for col in data.loc[:, data.isna().any()].columns:
      data[col] = data[col].fillna(data[col].mean())

    data = data.drop(data.columns[data.isna().any()].tolist(), axis =1)

  return data



def get_model(dimension_input):
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

def fit_and_get_metrics(data, model_name):

  y = data['pol'].replace({"liberal":1,"conservative":0})
  X = data.drop('pol', axis = 1)

  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2) 

  if model_name == "LR":
    # assert 'pol' in data.columns
    lr = LogisticRegression(penalty='l1',solver="saga")

    lr.fit(X_train, y_train)

    y_pred = lr.predict_proba(X_test)

    auc = round(metrics.roc_auc_score(y_test,y_pred[:,1])*100,2)
    acc = round(metrics.accuracy_score(y_test, lr.predict(X_test))*100,2)

    return auc, acc
  
  elif model_name == "NN":
    # assert 'pol' in data.columns

    model = get_model(X_train.shape[1])
    model.fit(epochs=25,x=X_train,y=y_train,batch_size=1000, verbose=0, validation_split=0.2)

    y_pred = model.predict(X_test)
  
    auc = round(metrics.roc_auc_score(y_test,y_pred)*100,2)

    _, acc = model.evaluate(X_test, y_test,batch_size=1000, verbose=0)

    return auc, round(acc*100, 2)
  
  else:
    print(" Enter Model name LR or NN for Logistic regression or Neural Network respectively.")

  
def get_dataframe_name(dataset_path):
  return "_".join(dataset_path.split("/")[-1:][0][:-4].split("_")[1:])

def get_segment_dataframe(segment_to_run = "Canada_0_dating"):
  """ 
  Function to concat the dataframe from training. 

  @param : 
    - segment_to_run : The segment which needs to be trained. 

  @ returns :
    - a dataframe with data points with Country _ gender _ database.
  """

  path = r'data/sample/' + segment_to_run # use your path
  
  all_files = glob.glob(path + "/*.csv")

  li = []

  for filename in all_files:
      df = pd.read_csv(filename, index_col=None, header=0)
      li.append(df)

  return pd.concat(li, axis=0, ignore_index=True)

#--------------------------------------------- raw

# Funtion to return the joined data 

def get_dataframe(segment_to_run = "Canada_0_dating"):
  """ 
  Function to concat the dataframe from training. 

  @param : 
    - segment_to_run : The segment which needs to be trained. 

  @ returns :
    - a dataframe with data points with Country _ gender _ database.
  """

  path = r'/content/drive/Shareddrives/Facial Recognition/data/' + segment_to_run # use your path
  
  all_files = glob.glob(path + "/*.csv")

  li = []

  for filename in all_files:
      df = pd.read_csv(filename, index_col=None, header=0)
      li.append(df)

  return pd.concat(li, axis=0, ignore_index=True)

def get_experiment_data(dataframe, column_list):
  return dataframe[column_list]

def save_eda(arr):
  # saving the results 
  results_df = pd.DataFrame(arr, columns = ["Segment","Samples","Distribution"])
  results_file_loc = "/content/drive/Shareddrives/Facial Recognition/eda/results/class_distribution.csv"
  results_df.to_csv(results_file_loc, index=False)
  print("Distribution Saved !!")

def save_segment_results(arr):
  # saving the results 
  results_df = pd.DataFrame(arr, columns = ["Features","Test AUC","Test Accuracy","Segment"])
  results_file_loc = "/content/drive/Shareddrives/Facial Recognition/exp_variation_ethinicity_vs_segments/results/LR_ethinicity_vs_segments.csv"
  results_df.to_csv(results_file_loc, index=False)
  print(" Segment Results Saved !!")

#--------------------------------------------- Not useful
