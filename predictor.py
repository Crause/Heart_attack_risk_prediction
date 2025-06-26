from joblib import load
import pandas as pd
import numpy as np
import re

class Predictor:

    features = pd.DataFrame()
    features_proc = pd.DataFrame()
    predictions = pd.DataFrame()

    def __init__(self):
        self.model = load('models/dtc_0521.joblib')

    def predict(self, file):
        self.features = pd.read_csv(file)
        self.features_proc = self.__proc(pd.read_csv(file))
        self.predictions = pd.DataFrame({'prediction': self.model.predict(self.features_proc)}, index=self.features_proc.index)

    def __proc(self, df):
        df = df.drop(columns=['Unnamed: 0'])
        df.set_index('id', inplace=True)
        df.rename(columns=lambda x: re.sub(r'[\(\)]', '', re.sub(r'[-\s]', '_', x.lower())), inplace=True)
 
        df['gender'] = df['gender'].replace({'Male': '1.0', 'Female': '0.0'}).astype(float)
        for col in ['stress_level', 'physical_activity_days_per_week']:
            median_dic = df.groupby('gender')[col].median()
            df = df.apply(lambda x: self.__fill_missing(x, col, 'gender', median_dic), axis=1)
        for col in ['alcohol_consumption', 'smoking']:
            median_dic = df.groupby('stress_level')[col].median()
            df = df.apply(lambda x: self.__fill_missing(x, col, 'stress_level', median_dic), axis=1)
        for col in ['diabetes', 'obesity', 'medication_use']:
            important_features = ['stress_level', 'physical_activity_days_per_week']
            median_dic = df.groupby(important_features)[col].median()
            df = df.apply(lambda x: self.__fill_missing_2d(x, col, important_features, median_dic), axis=1)
            
        df['bad_health'] = df['triglycerides'] * \
                            (0.1 + df['alcohol_consumption'])  * \
                            (0.1 + df['smoking'])  * \
                            (0.1 + df['obesity']) * \
                            (0.1 + df['diabetes']) * \
                            (0.1 + df['sedentary_hours_per_day']) * \
                            df['bmi'] * \
                            df['cholesterol'] * \
                            df['age'] 
        
        df['good_health'] = df['exercise_hours_per_week'] * df['physical_activity_days_per_week'] * df['sleep_hours_per_day']

        #bad_health
        df.drop(columns=['triglycerides'], inplace=True)
        df.drop(columns=['alcohol_consumption'], inplace=True)
        df.drop(columns=['smoking'], inplace=True)
        df.drop(columns=['obesity'], inplace=True)
        df.drop(columns=['diabetes'], inplace=True)
        df.drop(columns=['sedentary_hours_per_day'], inplace=True)
        df.drop(columns=['bmi'], inplace=True)
        df.drop(columns=['cholesterol'], inplace=True)
        df.drop(columns=['age'], inplace=True)
        #good_health
        df.drop(columns=['exercise_hours_per_week'], inplace=True)
        df.drop(columns=['physical_activity_days_per_week'], inplace=True)
        df.drop(columns=['sleep_hours_per_day'], inplace=True)
        #drop
        df.drop(columns=['family_history'], inplace=True)
        df.drop(columns=['diet'], inplace=True)
        df.drop(columns=['previous_heart_problems'], inplace=True)
        df.drop(columns=['troponin'], inplace=True)
        df.drop(columns=['income'], inplace=True)
        df.drop(columns=['heart_rate'], inplace=True)

        return df

    def __fill_missing(self, row, missing_feature, important_feature, median_dic):
        result_row = row.copy()
        
        if np.isnan(result_row[missing_feature]):
            result_row[missing_feature] = median_dic[result_row[important_feature]]
            
        return result_row
    
    def __fill_missing_2d(self, row, missing_feature, important_features, median_dic):
        result_row = row.copy()
        
        if np.isnan(result_row[missing_feature]):
            f1 = result_row[important_features[0]]
            f2 = result_row[important_features[1]]
            result_row[missing_feature] = median_dic[f1][f2]
            
        return result_row