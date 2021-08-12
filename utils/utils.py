from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import numpy as np


def get_sampled_data(dataset_path, n = 1000):
  
  data = pd.read_csv(dataset_path)  

  if data.shape[0] > n:
    return data.head(n)
  else:
    return data


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


def fit_and_get_metrics(data, model):

  assert 'pol' in data.columns

  y = data['pol']
  X = data.drop('pol', axis = 1)

  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2) 

  model.fit(X_train, y_train)

  y_pred = model.predict_proba(X_test)

  auc = round(metrics.roc_auc_score(y_test,y_pred[:,1]),2)*100
  acc = round(metrics.accuracy_score(y_test, model.predict(X_test)),2)*100

  return auc, acc, data.shape[0]

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
