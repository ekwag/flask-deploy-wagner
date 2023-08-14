import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from pybaseball import statcast_batter
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def get_player_id(pitcher_name, batter_name):
    first_name_pitcher, last_name_pitcher = pitcher_name.split(' ',1)
    first_name_pitcher = first_name_pitcher
    
    last_name_pitcher = last_name_pitcher
   
    get_id_pitcher = playerid_lookup(last_name_pitcher,first_name_pitcher)
    first_name_pitcher, last_name_pitcher = pitcher_name.split(' ',1)
    first_name_pitcher = first_name_pitcher
    
    last_name_pitcher = last_name_pitcher
   
    get_id_pitcher = playerid_lookup(last_name_pitcher,first_name_pitcher)
    
    first_name_batter, last_name_batter = batter_name.split(' ',1)
    first_name_batter = first_name_batter
    
    last_name_batter = last_name_batter
   
    get_id_batter = playerid_lookup(last_name_batter,first_name_batter)
    first_name_batter, last_name_batter = batter_name.split(' ',1)
    first_name_batter = first_name_batter
    
    last_name_batter = last_name_batter
   
    get_id_batter = playerid_lookup(last_name_batter,first_name_batter)
    
    pitcher_id = get_id_pitcher['key_mlbam'].to_string(index=False)
    batter_id = get_id_batter['key_mlbam'].to_string(index=False)
    
    #print(get_id_pitcher['key_mlbam'].to_string(index=False))
    #print(get_id_batter['key_mlbam'].to_string(index=False))
    #sorted_pitcher_data = get_pitcher_data(pitcher_id)
    #sorted_batter_data = get_batter_data(batter_id)
    return pitcher_id, batter_id #, sorted_pitcher_data, sorted_batter_data

def get_pitcher_data(pitcher_id):
   
    statcast_data_pitcher = statcast_pitcher('2019-07-01', '2019-07-31', player_id = pitcher_id)
    pitcher_data = statcast_data_pitcher[['pitch_type', 'description', 'p_throws', 'on_3b','on_2b','on_1b', 'zone', 'pitcher', 'balls', 'strikes', 'outs_when_up', 'delta_run_exp']].copy()
    newpredictiondatapitcher = pitcher_data.copy()
    
    groupbypitcher = newpredictiondatapitcher.groupby(['zone','pitch_type','balls','strikes'],as_index= False).delta_run_exp.sum()
    
    sorted_pitcher_data = pd.merge(newpredictiondatapitcher,groupbypitcher, how='left', on=['zone','pitch_type','balls','strikes'])
    
    return sorted_pitcher_data

def get_batter_data(batter_id):
    statcast_data_batter = statcast_batter('2019-07-01', '2023-07-31', player_id = batter_id)
    batter_data = statcast_data_batter[['batter','description','zone','pitch_type', 'p_throws', 'on_3b','on_2b','on_1b','outs_when_up','inning','balls','strikes','delta_run_exp']].copy()
    newpredictiondatabatter = batter_data.copy()
    groupbybatter = newpredictiondatabatter.groupby(['zone','pitch_type','balls','strikes'],as_index= False).delta_run_exp.sum()
    sorted_batter_data = pd.merge(newpredictiondatabatter,groupbybatter, how='left', on=['zone','pitch_type','balls','strikes'])
    return sorted_batter_data

def data_prep(sorted_pitcher_data, sorted_batter_data):
    # Prepare the data and absolutely use concat because delta_run is the same for pitcher and hitter
    #combined_data = pd.merge(sorted_pitcher_data, sorted_batter_data, on = ['pitch_type', 'zone', 'balls', 'strikes'], how = 'left')
    sorted_pitcher_data = sorted_pitcher_data.dropna(subset=['pitch_type'])
    sorted_batter_data = sorted_batter_data.dropna(subset=['pitch_type'])
    i = set(sorted_pitcher_data['pitch_type']) & set(sorted_batter_data['pitch_type'])
    combined_data = pd.concat([sorted_pitcher_data[sorted_pitcher_data['pitch_type'].isin(i)], sorted_batter_data[sorted_batter_data['pitch_type'].isin(i)]]).sort_values('pitch_type')
    
    #combined_data = pd.concat([sorted_pitcher_data[sorted_pitcher_data['pitch_type'].isin(sorted_batter_data['pitch_type'])],sorted_batter_data[sorted_batter_data['pitch_type'].isin(sorted_pitcher_data['pitch_type'])]])
    #combined_data = combined_data.dropna(subset=['zone'])
    
 
    
    combined_data.replace(np.nan,0)
    combined_data = combined_data.fillna(0)
    combined_data = combined_data.reset_index(drop=True)
    combined_data.drop(columns=['description', 'p_throws'], inplace=True)
    combined_data.pitch_type = pd.Categorical(combined_data.pitch_type)
    combined_data['pitch_code'] = combined_data.pitch_type.cat.codes

    combined_data['on_1b'] = combined_data['on_1b'].where(combined_data['on_1b'] <= 1, 1)
    combined_data['on_2b'] = combined_data['on_2b'].where(combined_data['on_2b'] <= 1, 1)
    combined_data['on_3b'] = combined_data['on_3b'].where(combined_data['on_3b'] <= 1, 1)
    combined_data.drop(columns=['pitch_type'], inplace=True)
    combined_data['batter'] = combined_data['batter'].astype(int)
    combined_data['zone'] = combined_data['zone'].astype(int)
    combined_data['on_3b'] = combined_data['on_3b'].astype(int)
    combined_data['on_2b'] = combined_data['on_2b'].astype(int)
    combined_data['on_1b'] = combined_data['on_1b'].astype(int)
    combined_data['inning'] = combined_data['inning'].astype(int)
    combined_data['pitcher'] = combined_data['pitcher'].astype(int)
    combined_data = combined_data.multiply(1000)
    combined_data['delta_run_exp_x'] = combined_data['delta_run_exp_x'].astype(int)
    combined_data['delta_run_exp_y'] = combined_data['delta_run_exp_y'].astype(int)
    return combined_data

def data_preprocessing(combined_data):
    # Data Preprocessing
    # Encode categorical variables using Label Encoding
    label_encoder = LabelEncoder()
    ordinal_encoder = OrdinalEncoder()
    combined_data['pitch_type_encoded'] = label_encoder.fit_transform(combined_data['pitch_code'])
    combined_data['zone_encoded'] = label_encoder.fit_transform(combined_data['zone'])
    combined_data['on_3b_encoded'] = label_encoder.fit_transform(combined_data['on_3b'])
    combined_data['on_2b_encoded'] = label_encoder.fit_transform(combined_data['on_2b'])
    combined_data['on_1b_encoded'] = label_encoder.fit_transform(combined_data['on_1b'])
    combined_data['inning_encoded'] = ordinal_encoder.fit_transform(combined_data[['inning']])
    combined_data['balls_encoded'] = ordinal_encoder.fit_transform(combined_data[['balls']])
    combined_data['strikes_encoded'] = ordinal_encoder.fit_transform(combined_data[['strikes']])
    combined_data['outs_encoded'] = ordinal_encoder.fit_transform(combined_data[['outs_when_up']])

    #combined_data['description_encoded'] = label_encoder.fit_transform(combined_data['description'])

    # Select relevant features for training the model #, 'description_encoded' #'on_3b', 'on_2b','on_1b','inning',
    features = ['on_3b_encoded', 'on_2b_encoded','on_1b_encoded','inning_encoded','outs_encoded', 'balls_encoded', 'strikes_encoded', 'zone_encoded', 'pitch_type_encoded']
    #features = ['batter', 'pitcher', 'balls', 'strikes', 'zone', 'pitch_type_encoded']
    target = 'delta_run_exp_y'
    return features, target, combined_data, ordinal_encoder

def model_train(features, target, combined_data):
    # Train-Test Split
    X = combined_data[features]
    y = combined_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the Model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    #pickle.dump(model, open("model.pkl", "wb")) ##pick up here tomorrow. Is this where I pickle
    return features, target, combined_data, model

# Predicting the Zone and Pitch Type for the Best delta_run_exp #batter_id, pitcher_id,
def get_best_zone_and_pitch_type(model, features, ordinal_encoder, combined_data, balls_encoded, strikes_encoded, on_3b_encoded, on_2b_encoded, on_1b_encoded, inning_encoded, outs_encoded):
    # Create a dataframe with all possible combinations of zone and pitch_type
    
    count = combined_data.pitch_type_encoded.nunique()
    
    possible_combinations = pd.DataFrame({
        #'batter': [batter_id] * 13 * count,  # Replace 9 and 8 with the number of zones and pitch_types respectively
        #'pitcher': [pitcher_id] * 13 * count,
        'pitch_type_encoded': np.tile(np.arange(0, count), 13),
        'balls_encoded': [balls_encoded] * 13 * count,
        'strikes_encoded': [strikes_encoded] * 13 * count,
        'zone_encoded': np.repeat(np.arange(1, 14), count),
        
        'on_3b_encoded': [on_3b_encoded] * 13 * count,
        'on_2b_encoded': [on_2b_encoded] * 13 * count,
        'on_1b_encoded': [on_1b_encoded] * 13 * count,
        'inning_encoded': [inning_encoded] * 13 * count,
        'outs_encoded': [outs_encoded] * 13 * count
        
        
       
        #need to figure out what this really is so that I can add the things that are errors below
        #'description_encoded': np.tile(np.arange(0, 8), 9)  # Assuming 8 unique descriptions for pitch_types
    })
    #print(possible_combinations)
    # Make predictions for each combination
    predictions = model.predict(possible_combinations[features])
    
    # Get the index of the combination with the lowest predicted delta_run_exp
    best_idx = np.argmin(predictions)
    
    # Get the zone and pitch_type corresponding to the best index
    best_zone = possible_combinations.loc[best_idx, 'zone_encoded']
    best_pitch_type_encoded = possible_combinations.loc[best_idx, 'pitch_type_encoded']
    #print(count)
    
    #print(combined_data)
    #print(possible_combinations)
    #print(best_pitch_type_encoded)
    #combined_data['on_1b_encoded'] = label_encoder.fit_transform(combined_data['on_1b'])
    #combined_data['inning_encoded'] = ordinal_encoder.fit_transform(combined_data[['inning']])
    best_pitch_type = ordinal_encoder.inverse_transform(best_pitch_type_encoded.reshape(-1, 1))[0,0] #ok, so this could be the issue, not all are label encoded
    
    return predictions, best_zone, best_pitch_type, possible_combinations

def main(pitcher_name, batter_name, balls_encoded, strikes_encoded, outs_encoded, on_3b_encoded, on_2b_encoded, on_1b_encoded, inning_encoded):
    pitcher_id, batter_id = get_player_id(pitcher_name, batter_name)
    sorted_pitcher_data = get_pitcher_data(pitcher_id)
    sorted_batter_data = get_batter_data(batter_id)
    combined_data = data_prep(sorted_pitcher_data, sorted_batter_data)
    features, target, combined_data, ordinal_encoder = data_preprocessing(combined_data)
    features, target, combined_data, model = model_train(features, target, combined_data)
    predictions, best_zone, best_pitch_type, possible_combinations = get_best_zone_and_pitch_type(model, features, ordinal_encoder, combined_data, balls_encoded, strikes_encoded, on_3b_encoded, on_2b_encoded, on_1b_encoded, inning_encoded, outs_encoded)
    return predictions, best_zone, best_pitch_type, possible_combinations




