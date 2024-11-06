import numpy as np
import pandas as pd
import joblib
import json
import gradio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier




from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load dataset
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')


# Handing outliers
outlier_colms = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
df1 = df.copy()
outlier_handling = {}

def handle_outliers(df, colm, out_dict):
    '''Change the values of outlier to upper and lower whisker values '''
    q1 = df.describe()[colm].loc["25%"]
    q3 = df.describe()[colm].loc["75%"]
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    out_dict['lower_bound'] = lower_bound
    out_dict['upper_bound'] = upper_bound

    for i in range(len(df)):
        if df.loc[i,colm] > upper_bound:
            df.loc[i,colm]= upper_bound
        if df.loc[i,colm] < lower_bound:
            df.loc[i,colm]= lower_bound
    return df, out_dict

for colm in outlier_colms:
    out_dict = {}
    df1, outlier_dict = handle_outliers(df1, colm, out_dict)
    outlier_handling[colm] = out_dict





json_string = json.dumps(outlier_handling, indent=4)
print(json_string)

# Export to a JSON file
with open('outlier_handling.json', 'w') as json_file:
    json.dump(outlier_handling, json_file, indent=4)



model = joblib.load('xgboost-model.pkl')


with open('outlier_handling.json', 'r') as json_file:
    loaded_outlier_handling = json.load(json_file)
loaded_outlier_handling


# Function for prediction

def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                        high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):

    yes_no_map = {'Yes':1, 'No':0}
    gender_map = {'M':1, 'F':0}

    inputs = {
                'age': age,
                'anaemia': yes_no_map[anaemia],
                'creatinine_phosphokinase': creatinine_phosphokinase,
                'diabetes': yes_no_map[diabetes],
                'ejection_fraction': ejection_fraction,
                'high_blood_pressure': yes_no_map[high_blood_pressure],
                'platelets': platelets,
                'serum_creatinine': serum_creatinine,
                'serum_sodium': serum_sodium,
                'sex': gender_map[sex],
                'smoking': yes_no_map[smoking],
                'time': time
                }


    outlier_colms = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
    for col in outlier_colms:

        if inputs[col] > loaded_outlier_handling[col]['upper_bound']:
            inputs[col] = loaded_outlier_handling[col]['upper_bound']
        elif inputs[col] < loaded_outlier_handling[col]['lower_bound']:
            inputs[col] = loaded_outlier_handling[col]['lower_bound']

    values = list(inputs.values())
    inputs_to_model = np.array(values).reshape(1, -1)

    prediction = model.predict(inputs_to_model)[0]
    prob = model.predict_proba(inputs_to_model)[0]*100

    if prediction==1:
        return f"Subject patient will not survive with a probability of {prob[1]:.1f} %"
    elif prediction==0:
        return f"Subject patient will survive with a probability of {prob[0]:.1f} %"
    else:
        return f"Error observed while making prediction"
    

inputs = [gradio.Slider(df['age'].min(), df['age'].max(), label="Enter the age of the patient:"),
          gradio.Radio(["Yes", "No"], label="Whether patient is Anaemic or not?:"),
          gradio.Slider(round(df['creatinine_phosphokinase'].min(),2), round(df['creatinine_phosphokinase'].max(),2), label="Enter the level of CPK enzyme in the patient's blood (mcg/L):"),
          gradio.Radio(["Yes", "No"], label="Whether patient is diabetic or not?:"),
          gradio.Slider(df['ejection_fraction'].min(), df['ejection_fraction'].max(), label="Enter the % of blood leaving the patient's heart at each contraction:"),
          gradio.Radio(["Yes", "No"], label="Whether patient is Hypertensive or not?:"),
          gradio.Slider(df['platelets'].min(), df['platelets'].max(), label="Enter the No. of platelets in the patient's blood (kiloplatelets/mL):"),
          gradio.Slider(round(df['serum_creatinine'].min(),2), round(df['serum_creatinine'].max(),2), label="Enter the level of serum creatinine in the patient's blood (mg/dL):"),
          gradio.Slider(round(df['serum_sodium'].min(),2), round(df['serum_sodium'].max(),2), label="Enter the level of serum sodium in the patient's blood (mEq/L): "),
          gradio.Radio(["M", "F"], label="Choose the sex of the patient:"),
          gradio.Radio(["Yes", "No"], label="Whether the patient smokes or not?:"),
          gradio.Slider(df['time'].min(), df['time'].max(), label="Enter the follow-up period (days):"),
          ]

# Output response
outputs = gradio.Textbox(type="text", label='Will the patient survive?')


title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,inputs = inputs,outputs = outputs,title = title,description = description)

iface.launch(server_name="127.0.0.0", server_port = 8001, share = True)
