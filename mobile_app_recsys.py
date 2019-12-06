###############         Anastasios Kachrimanis            ######################
###############        Student number: u792289            ######################
###############    a.kachrimanis@tilburguniversity.edu    ######################

# Thesis Tilburg University
# CF Recommender System using an RFD-TP model


#import the libraries
import pandas as pd
import numpy as np
from scipy import sparse
import random 
from sklearn import preprocessing
import tensorrec



#read the data
data = pd.read_csv("phone_use_data.csv")
data.head()

n_users = len(data.user_id.value_counts())
n_items = len(data.application.value_counts())

index = np.unique(data.user_id)
columns = np.unique(data.application)

################ FEATURE ENGINEERING #####################
#------------------- Dowload history approach  ---------------------------#

dh = pd.DataFrame(index=index, columns=columns)
dh = dh.fillna(0) 

dh.index = dh.index.map(str)
dh.index.names = ['users']
dh = dh.reset_index()

look_up_table_users = []
for index, value in zip(range(len(dh.users)), dh.users): 
     look_up_table_users.append((index, value))
        
     
    
for i in data.groupby(['user_id']):
    idx = dh.index[dh['users'] == str(i[0])].tolist()
    
    apps = (np.unique(i[1]['application']))
    for app in apps:    
        dh.at[idx, app]=1
        
dh = dh.drop(['users'], axis=1)

dh_interactions = pd.DataFrame(columns=['UserID', 'AppID', 'Installed'])

for i in range(len(dh)):
    for j in range(dh.shape[1]):
        if dh.iloc[i][j]!=0:
            dh_interactions = dh_interactions.append({'UserID': i, 'AppID': j, 'Installed': 1}, ignore_index=True)
            
look_up_table_apps = []
for index, value in zip(range(dh.shape[1]), dh.keys()): 
     look_up_table_apps.append((index, value))
     
     
dh_ratings = np.array(dh_interactions)


#------------------- RFD approach  ---------------------------#

look_up_table_users = {}

users = np.unique(data.user_id)

for i in range(len(users)):
    look_up_table_users[str(users[i])] = str(i)
    
lu_table_apps_inverse = {}

apps = np.unique(data.application)

for i in range(len(apps)):
    lu_table_apps_inverse[str(i)] = str(apps[i])
    
#### Recency
index = np.unique(data.user_id)
columns = np.unique(data.application)
recency = pd.DataFrame(index=index, columns=columns)
recency = recency.fillna(0) 

recency.index = recency.index.map(str)
recency.index.names = ['users']

for i in range(data.shape[0]):
    
    label = str(data.iloc[i]["application"])
    user_id = str(data.iloc[i]["user_id"])
    user_index = look_up_table_users[user_id]

    recency.iloc[int(user_index)][label] = data.iloc[i]["endTimeMillis"]
    
min_max_scaler = preprocessing.MinMaxScaler()
recency_scaled = min_max_scaler.fit_transform(recency)

#### Frequency
index = np.unique(data.user_id)
columns = np.unique(data.application)
frequency = pd.DataFrame(index=index, columns=columns)
frequency = frequency.fillna(0) 

frequency.index = frequency.index.map(str)
frequency.index.names = ['users']

for i in range(data.shape[0]):
    
    label = str(data.iloc[i]["application"])
    user_id = str(data.iloc[i]["user_id"])
    user_index = look_up_table_users[user_id]

    frequency.iloc[int(user_index)][label] += 1

frequency_scaled = min_max_scaler.fit_transform(frequency)

####Duration
index = np.unique(data.user_id)
columns = np.unique(data.application)
duration = pd.DataFrame(index=index, columns=columns)
duration = duration.fillna(0) 

duration.index = duration.index.map(str)
duration.index.names = ['users']

for i in range(data.shape[0]):
    
    label = str(data.iloc[i]["application"])
    user_id = str(data.iloc[i]["user_id"])
    user_index = look_up_table_users[user_id]

    duration.iloc[int(user_index)][label] += data.iloc[i]["endTimeMillis"] - data.iloc[i]["startTimeMillis"]
    
duration_scaled = min_max_scaler.fit_transform(duration)

### RFD usage score
index = np.unique(data.user_id)
columns = np.unique(data.application)

usage_score = pd.DataFrame(index=index, columns=columns, dtype=np.float32)
usage_score = usage_score.fillna(0) 
usage_score.index = usage_score.index.map(str)
usage_score.index.names = ['users']

use_score_table = recency_scaled + frequency_scaled + duration_scaled

for row in range(use_score_table.shape[0]):
    for column in range(use_score_table.shape[1]):
        app = lu_table_apps_inverse[str(column)]
        usage_score.iloc[row][app] = use_score_table[row][column]
        
rfd_interactions = pd.DataFrame(columns=['UserID', 'AppID', 'Rating'])

for i in range(usage_score.shape[0]):
    for j in range(usage_score.shape[1]):
        if usage_score.iloc[i][j]!=0:
            rfd_interactions = rfd_interactions.append({'UserID': int(i), 'AppID': int(j), 'Rating': usage_score.iloc[i][j]}, ignore_index=True)


convert_dict = {'UserID': int, 
                'AppID': int
               } 
  
rfd_interactions = rfd_interactions.astype(convert_dict) 


rfd_ratings = np.array(rfd_interactions)

#------------------- TPFD approach  ---------------------------#
index = np.unique(data.user_id)
columns = np.unique(data.application)
temporal_pattern = pd.DataFrame(index=index, columns=columns)
temporal_pattern = temporal_pattern.fillna(0) 

temporal_pattern.index = temporal_pattern.index.map(str)
temporal_pattern.index.names = ['users']

#sth is wrong it drops accuracy
day_millisecs = 86400000
last_day = max(data["endTimeMillis"])
total_days = (max(data["endTimeMillis"]) - min(data["startTimeMillis"]))/day_millisecs

for i in data.groupby(['user_id', 'application']):
    sample = i[1]
    sample = sample.sort_values(by=['startTime'])

    inactive_days = 0
    if len(sample) > 1:
        for times in range(len(sample)-1):
            diff = sample.iloc[times+1]["startTimeMillis"] - sample.iloc[times]["endTimeMillis"]
            if (diff) >= day_millisecs:
                inactive_days += diff//day_millisecs

        last_day_diff = last_day - sample.iloc[-1]["endTimeMillis"]
        if last_day_diff >= day_millisecs:
            inactive_days += last_day_diff//day_millisecs
        
    else:
        diff = last_day - sample.iloc[0]["endTimeMillis"]
        inactive_days = diff//day_millisecs
    
            
    label = str(sample.iloc[0]["application"])
    user_id = str(sample.iloc[0]["user_id"])
    user_index = look_up_table_users[user_id]
    temporal_pattern.iloc[int(user_index)][label] = total_days - inactive_days
 
    
temporal_pattern_scaled = min_max_scaler.fit_transform(temporal_pattern)
 
### TPFD usage score
index = np.unique(data.user_id)
columns = np.unique(data.application)

tpfd_usage_score = pd.DataFrame(index=index, columns=columns, dtype=np.float32)
tpfd_usage_score = tpfd_usage_score.fillna(0) 

tpfd_usage_score.index = tpfd_usage_score.index.map(str)
tpfd_usage_score.index.names = ['users']

tpfd_use_score_table = recency_scaled + frequency_scaled + duration_scaled + temporal_pattern_scaled

for row in range(tpfd_use_score_table.shape[0]):
    for column in range(tpfd_use_score_table.shape[1]):
        app = lu_table_apps_inverse[str(column)]
        tpfd_usage_score.iloc[row][app] = tpfd_use_score_table[row][column]
        
tpfd_interactions = pd.DataFrame(columns=['UserID', 'AppID', 'Rating'])

for i in range(tpfd_usage_score.shape[0]):
    for j in range(tpfd_usage_score.shape[1]):
        if tpfd_usage_score.iloc[i][j]!=0:
            tpfd_interactions = tpfd_interactions.append({'UserID': int(i), 'AppID': int(j), 'Rating': tpfd_usage_score.iloc[i][j]}, ignore_index=True)
            
convert_dict = {'UserID': int, 
                'AppID': int
               } 
  
tpfd_interactions = tpfd_interactions.astype(convert_dict) 
  
tpfd_ratings = np.array(tpfd_interactions)





########## Split the data for all 3 approaches

train_size = int(.8 * len(dh_ratings))

random.shuffle(dh_ratings)
dh_train_ratings = dh_ratings[:train_size]
dh_test_ratings = dh_ratings[train_size:]

random.shuffle(rfd_ratings)
rfd_train_ratings = rfd_ratings[:train_size]
rfd_test_ratings = rfd_ratings[train_size:]

random.shuffle(tpfd_ratings)
tpfd_train_ratings = tpfd_ratings[:train_size]
tpfd_test_ratings = tpfd_ratings[train_size:]

######################### FUNCTION ############################################
################# Interactions to sparse matrix ###############################

def int_to_sparse(ratings_list):
    users, apps, ratings = zip(*ratings_list)
    
    return sparse.coo_matrix((ratings, (users, apps)),shape=(n_users, n_items))


###############################################################################

sp_dh_train_ratings = int_to_sparse(dh_train_ratings)
sp_dh_test_ratings = int_to_sparse(dh_test_ratings)

sp_rfd_train_ratings = int_to_sparse(rfd_train_ratings)
sp_rfd_test_ratings = int_to_sparse(rfd_test_ratings)

sp_tpfd_train_ratings = int_to_sparse(tpfd_train_ratings)
sp_tpfd_test_ratings = int_to_sparse(tpfd_test_ratings)

user_factors = sparse.identity(n_users)
app_factors = sparse.identity(n_items)

######################### FUNCTION ############################################
################### Recall at k for the whole dataset  ########################

def rec_at_k(predictions, k, test_ratings):
    result = tensorrec.eval.recall_at_k(test_interactions=test_ratings,
        predicted_ranks=predictions, k=k).mean()

    return result

###############################################################################
    

####### Define the model
cf_model = tensorrec.TensorRec(n_components=100,
                              loss_graph=tensorrec.loss_graphs.WMRBLossGraph())



############################# Generate Predictions and Evaluate  ##############


###### Baseline model
random_recs_5 = tensorrec.eval.eval_random_ranks_on_dataset(sp_dh_test_ratings, recall_k=5)
random_recall_at_5, _, _ = random_recs_5
print("Baseline model recall@5:", random_recall_at_5)


random_recs_10 = tensorrec.eval.eval_random_ranks_on_dataset(sp_dh_test_ratings, recall_k=10)
random_recall_at_10, _, _ = random_recs_10
print("Baseline model recall@10:", random_recall_at_10)


random_recs_20 = tensorrec.eval.eval_random_ranks_on_dataset(sp_dh_test_ratings, recall_k=20)
random_recall_at_20, _, _ = random_recs_20
print("Baseline model recall@20:", random_recall_at_20)


#################### Download history approach   ##################
#### fit the model
cf_model.fit(interactions=sp_dh_train_ratings,
                     user_features=user_factors,
                     item_features=app_factors,
                     n_sampled_items=int(n_items * .01))


#### predict ratings
dh_predictions = cf_model.predict_rank(user_features=user_factors,
                                                item_features=app_factors)


dh_recall_at_5 = rec_at_k(dh_predictions, 5, sp_dh_test_ratings)
print("Download history approach recall@5:", dh_recall_at_5)

dh_recall_at_10 = rec_at_k(dh_predictions, 10, sp_dh_test_ratings)
print("Download history approach recall@10:", dh_recall_at_10)

dh_recall_at_20 = rec_at_k(dh_predictions, 20, sp_dh_test_ratings)
print("Download history approach recall@5:", dh_recall_at_20)


#################### RFD approach   ##################
#### fit the model
cf_model.fit(interactions=sp_rfd_train_ratings,
                     user_features=user_factors,
                     item_features=app_factors,
                     n_sampled_items=int(n_items * .01))


#### predict ratings
rfd_predictions = cf_model.predict_rank(user_features=user_factors,
                                                item_features=app_factors)


rfd_recall_at_5 = rec_at_k(rfd_predictions, 5, sp_rfd_test_ratings)
print("RFD approach recall@5:", rfd_recall_at_5)

rfd_recall_at_10 = rec_at_k(rfd_predictions, 10, sp_rfd_test_ratings)
print("RFD approach recall@10:", rfd_recall_at_10)

rfd_recall_at_20 = rec_at_k(rfd_predictions, 20, sp_rfd_test_ratings)
print("RFD approach recall@20:", rfd_recall_at_20)



#################### TPFD approach   ##################
#### fit the model
cf_model.fit(interactions=sp_tpfd_train_ratings,
                     user_features=user_factors,
                     item_features=app_factors,
                     n_sampled_items=int(n_items * .01))


#### predict ratings
tpfd_predictions = cf_model.predict_rank(user_features=user_factors,
                                                item_features=app_factors)


tpfd_recall_at_5 = rec_at_k(tpfd_predictions, 5, sp_tpfd_test_ratings)
print("TPFD approach recall@5:", tpfd_recall_at_5)

tpfd_recall_at_10 = rec_at_k(tpfd_predictions, 10, sp_tpfd_test_ratings)
print("TPFD approach recall@10:", tpfd_recall_at_10)

tpfd_recall_at_20 = rec_at_k(tpfd_predictions, 20, sp_tpfd_test_ratings)
print("TPFD approach recall@20:", tpfd_recall_at_20)

