# %% setup


import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


# %% data extraction


path = ''

filename = path + "data_test2018.txt"
with open(filename, 'w') as f:
    f.write("")


pids = [559, 563, 570, 575, 588, 591]
for n, pid in enumerate(pids):
    xmlfile = path + "Data/2018/test/" + str(pid) + "-ws-testing.xml"
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    
    timestamps = []
    for item in root.findall("glucose_level")[0]:
        timestamps.append(item.attrib['ts'])
    timestamps = pd.to_datetime(timestamps, dayfirst=True)
    min_date = min(timestamps).date()
    max_date = max(timestamps + pd.DateOffset(1)).date()
    
    
    ## create dataframe
    info = pd.DataFrame({
        'pid': pid,
        'time': pd.date_range(min_date, max_date, freq='5min')
    })
    info.drop(index=0, inplace=True)
    info.reset_index(drop=True, inplace=True)
    
    
    ## glucose level
    glucose_level = []
    for item in root.findall("glucose_level")[0]:
        glucose_level.append([item.attrib['ts'], item.attrib['value']])
    glucose_level = pd.DataFrame(glucose_level)
    glucose_level.columns = ['ts', 'value']
    glucose_level['ts'] = pd.to_datetime(glucose_level['ts'], dayfirst=True)
    glucose_level['value'] = pd.to_numeric(glucose_level['value'])
    
    info['glucose_level'] = [0.0] * info.shape[0]
    j = 0
    for i in range(info.shape[0]):
        curr_values = []
        while (j < glucose_level.shape[0]) and (glucose_level.loc[j, 'ts'] <= info.loc[i, 'time']):
            curr_values.append(glucose_level.loc[j, 'value'])
            j += 1
        info.loc[i, 'glucose_level'] = np.mean(curr_values) if curr_values else np.nan
    
    
    ## basal insulin
    basal = []
    for item in root.findall("basal")[0]:
        basal.append([item.attrib['ts'], item.attrib['value']])
    basal = pd.DataFrame(basal)
    basal.columns = ['ts', 'value']
    basal['ts'] = pd.to_datetime(basal['ts'], dayfirst=True)
    basal['value'] = pd.to_numeric(basal['value'])
    
    info['basal'] = [0.0] * info.shape[0]
    j = 0
    for i in range(info.shape[0]):
        curr_values = []
        while (j < basal.shape[0]) and (basal.loc[j, 'ts'] <= info.loc[i, 'time']):
            curr_values.append(basal.loc[j, 'value'])
            j += 1
        info.loc[i, 'basal'] = np.mean(curr_values) if curr_values else np.nan
    ## fill missing data with the last valid value
    info['basal'] = info['basal'].ffill(axis=0)
    
    
    ## temporary basal insulin rate
    temp_basal = []
    for item in root.findall("temp_basal")[0]:
        temp_basal.append([item.attrib['ts_begin'], item.attrib['ts_end'], item.attrib['value']])

    if len(temp_basal) > 0:
        temp_basal = pd.DataFrame(temp_basal)
        temp_basal.columns = ['ts_begin', 'ts_end', 'value']
        temp_basal['ts_begin'] = pd.to_datetime(temp_basal['ts_begin'], dayfirst=True)
        temp_basal['ts_end'] = pd.to_datetime(temp_basal['ts_end'], dayfirst=True)
        temp_basal['value'] = pd.to_numeric(temp_basal['value'])
        
        j = 0
        for i in range(info.shape[0]):
            curr_values = []
            while (j < temp_basal.shape[0]) and (temp_basal.loc[j, 'ts_begin'] <= info.loc[i, 'time']):
                curr_values.append(temp_basal.loc[j, 'value'])
                if temp_basal.loc[j, 'ts_end'] <= info.loc[i, 'time']:
                    j += 1
                else:
                    break
            ## temporary basal insulin rate that supersedes the patient's normal basal rate
            if curr_values: info.loc[i, 'basal'] = np.mean(curr_values)
    
    
    ## bolus insulin
    bolus = []
    for item in root.findall("bolus")[0]:
        bolus.append([item.attrib['ts_begin'], item.attrib['dose']])
    bolus = pd.DataFrame(bolus)
    bolus.columns = ['ts', 'value']
    bolus['ts'] = pd.to_datetime(bolus['ts'], dayfirst=True)
    bolus['value'] = pd.to_numeric(bolus['value'])
    
    info['bolus'] = [0.0] * info.shape[0]
    j = 0
    for i in range(info.shape[0]):
        curr_values = []
        while (j < bolus.shape[0]) and (bolus.loc[j, 'ts'] <= info.loc[i, 'time']):
            curr_values.append(bolus.loc[j, 'value'])
            j += 1
        info.loc[i, 'bolus'] = np.mean(curr_values) if curr_values else 0
    
    
    ## self-reported time and type of a meal, carbohydrate estimate
    meal = []
    for item in root.findall("meal")[0]:
        meal.append([item.attrib['ts'], item.attrib['carbs']])
    meal = pd.DataFrame(meal)
    meal.columns = ['ts', 'value']
    meal['ts'] = pd.to_datetime(meal['ts'], dayfirst=True)
    meal['value'] = pd.to_numeric(meal['value'])
    
    info['meal'] = [0.0] * info.shape[0]
    j = 0
    for i in range(info.shape[0]):
        curr_values = []
        while (j < meal.shape[0]) and (meal.loc[j, 'ts'] <= info.loc[i, 'time']):
            curr_values.append(meal.loc[j, 'value'])
            j += 1
        info.loc[i, 'meal'] = np.mean(curr_values) if curr_values else 0
    ## delayed effect of the meal
    delay = int(60 / 5)
    meals = info['meal'].shift(periods=list(range(delay)), axis=0, fill_value=0)
    meals = meals * np.array([0.9**a for a in range(delay)])
    info['meal'] = meals.sum(axis=1)
    
    
    ## self-reported sleep, plus the patient’s subjective assessment of sleep quality
    sleep = []
    for item in root.findall("sleep")[0]:
        ## in the original dataset, ts_begin > ts_end, so we switched the order here
        sleep.append([item.attrib['ts_end'], item.attrib['ts_begin'], item.attrib['quality']])
    sleep = pd.DataFrame(sleep)
    sleep.columns = ['ts_begin', 'ts_end', 'value']
    sleep['ts_begin'] = pd.to_datetime(sleep['ts_begin'], dayfirst=True)
    sleep['ts_end'] = pd.to_datetime(sleep['ts_end'], dayfirst=True)
    sleep['value'] = pd.to_numeric(sleep['value'])
    
    info['sleep'] = [0.0] * info.shape[0]
    j = 0
    for i in range(info.shape[0]):
        curr_values = []
        while (j < sleep.shape[0]) and (sleep.loc[j, 'ts_begin'] <= info.loc[i, 'time']):
            curr_values.append(sleep.loc[j, 'value'])
            if sleep.loc[j, 'ts_end'] <= info.loc[i, 'time']:
                j += 1
            else:
                break
        info.loc[i, 'sleep'] = np.mean(curr_values) if curr_values else 0
    
    
    ## self-reported times of going to and from work, subjective assessment of physical exertion
    work = []
    for item in root.findall("work")[0]:
        work.append([item.attrib['ts_begin'], item.attrib['ts_end'], item.attrib['intensity']])

    info['work'] = [0.0] * info.shape[0]
    if len(work) > 0:
        work = pd.DataFrame(work)
        work.columns = ['ts_begin', 'ts_end', 'value']
        work['ts_begin'] = pd.to_datetime(work['ts_begin'], dayfirst=True)
        work['ts_end'] = pd.to_datetime(work['ts_end'], dayfirst=True)
        work['value'] = pd.to_numeric(work['value'])
        
        j = 0
        for i in range(info.shape[0]):
            curr_values = []
            while (j < work.shape[0]) and (work.loc[j, 'ts_begin'] <= info.loc[i, 'time']):
                curr_values.append(work.loc[j, 'value'])
                if work.loc[j, 'ts_end'] <= info.loc[i, 'time']:
                    j += 1
                else:
                    break
            info.loc[i, 'work'] = np.mean(curr_values) if curr_values else 0
    
    
    ## self-reported stress
    stressors = []
    for item in root.findall("stressors")[0]:
        stressors.append([item.attrib['ts']])

    info['stressors'] = [0.0] * info.shape[0]
    if len(stressors) > 0:
        stressors = pd.DataFrame(stressors)
        stressors.columns = ['ts']
        stressors['ts'] = pd.to_datetime(stressors['ts'], dayfirst=True)
        
        j = 0
        for i in range(info.shape[0]):
            curr_values = []
            while (j < stressors.shape[0]) and (stressors.loc[j, 'ts'] <= info.loc[i, 'time']):
                curr_values.append(1)
                j += 1
            info.loc[i, 'stressors'] = np.mean(curr_values) if curr_values else 0
    
    
    ## self-reported hypoglycemic episode
    hypo_event = []
    for item in root.findall("hypo_event")[0]:
        hypo_event.append([item.attrib['ts']])

    info['hypo_event'] = [0.0] * info.shape[0]
    if len(hypo_event) > 0:
        hypo_event = pd.DataFrame(hypo_event)
        hypo_event.columns = ['ts']
        hypo_event['ts'] = pd.to_datetime(hypo_event['ts'], dayfirst=True)
        
        j = 0
        for i in range(info.shape[0]):
            curr_values = []
            while (j < hypo_event.shape[0]) and (hypo_event.loc[j, 'ts'] <= info.loc[i, 'time']):
                curr_values.append(1)
                j += 1
            info.loc[i, 'hypo_event'] = np.mean(curr_values) if curr_values else 0
    ## delayed effect of the meal
    delay = int(15 / 5)
    hypos = info['hypo_event'].shift(periods=list(range(delay)), axis=0, fill_value=0)
    hypos = hypos * np.array([0.8**a for a in range(delay)])
    info['hypo_event'] = hypos.sum(axis=1)
    
    
    ## self-reported illness
    illness = []
    for item in root.findall("illness")[0]:
        illness.append([item.attrib['ts_begin']])

    info['illness'] = [0.0] * info.shape[0]
    if len(illness) > 0:
        illness = pd.DataFrame(illness)
        illness.columns = ['ts']
        illness['ts'] = pd.to_datetime(illness['ts'], dayfirst=True)
        
        j = 0
        for i in range(info.shape[0]):
            curr_values = []
            while (j < illness.shape[0]) and (illness.loc[j, 'ts'] <= info.loc[i, 'time']):
                curr_values.append(1)
                j += 1
            info.loc[i, 'illness'] = np.mean(curr_values) if curr_values else 0
    
    
    ## time and duration, in minutes, of selfreported exercise, subjective assessment of physical exertion
    exercise = []
    for item in root.findall("exercise")[0]:
        exercise.append([item.attrib['ts'], item.attrib['duration'], item.attrib['intensity']])

    info['exercise'] = [0.0] * info.shape[0]
    if len(exercise) > 0:
        exercise = pd.DataFrame(exercise)
        exercise.columns = ['ts_begin', 'ts_end', 'value']
        exercise['ts_begin'] = pd.to_datetime(exercise['ts_begin'], dayfirst=True)
        exercise['ts_end'] = pd.to_numeric(exercise['ts_end'])
        exercise['ts_end'] = pd.to_timedelta(exercise['ts_end'], unit='m')
        exercise['ts_end'] = exercise['ts_begin'] + exercise['ts_end']
        exercise['value'] = pd.to_numeric(exercise['value'])
        
        j = 0
        for i in range(info.shape[0]):
            curr_values = []
            while (j < exercise.shape[0]) and (exercise.loc[j, 'ts_begin'] <= info.loc[i, 'time']):
                curr_values.append(exercise.loc[j, 'value'])
                if exercise.loc[j, 'ts_end'] <= info.loc[i, 'time']:
                    j += 1
                else:
                    break
            info.loc[i, 'exercise'] = np.mean(curr_values) if curr_values else 0
    
    
    ## heart rate
    basis_heart_rate = []
    for item in root.findall("basis_heart_rate")[0]:
        basis_heart_rate.append([item.attrib['ts'], item.attrib['value']])
    basis_heart_rate = pd.DataFrame(basis_heart_rate)
    basis_heart_rate.columns = ['ts', 'value']
    basis_heart_rate['ts'] = pd.to_datetime(basis_heart_rate['ts'], dayfirst=True)
    basis_heart_rate['value'] = pd.to_numeric(basis_heart_rate['value'])
    
    info['basis_heart_rate'] = [0.0] * info.shape[0]
    j = 0
    for i in range(info.shape[0]):
        curr_values = []
        while (j < basis_heart_rate.shape[0]) and (basis_heart_rate.loc[j, 'ts'] <= info.loc[i, 'time']):
            curr_values.append(basis_heart_rate.loc[j, 'value'])
            j += 1
        info.loc[i, 'basis_heart_rate'] = np.mean(curr_values) if curr_values else np.nan

    
    ## galvanic skin response
    basis_gsr = []
    for item in root.findall("basis_gsr")[0]:
        basis_gsr.append([item.attrib['ts'], item.attrib['value']])
    basis_gsr = pd.DataFrame(basis_gsr)
    basis_gsr.columns = ['ts', 'value']
    basis_gsr['ts'] = pd.to_datetime(basis_gsr['ts'], dayfirst=True)
    basis_gsr['value'] = pd.to_numeric(basis_gsr['value'])
    
    info['basis_gsr'] = [0.0] * info.shape[0]
    j = 0
    for i in range(info.shape[0]):
        curr_values = []
        while (j < basis_gsr.shape[0]) and (basis_gsr.loc[j, 'ts'] <= info.loc[i, 'time']):
            curr_values.append(basis_gsr.loc[j, 'value'])
            j += 1
        info.loc[i, 'basis_gsr'] = np.mean(curr_values) if curr_values else np.nan
    
    
    ## skin temperature
    basis_skin_temperature = []
    for item in root.findall("basis_skin_temperature")[0]:
        basis_skin_temperature.append([item.attrib['ts'], item.attrib['value']])
    basis_skin_temperature = pd.DataFrame(basis_skin_temperature)
    basis_skin_temperature.columns = ['ts', 'value']
    basis_skin_temperature['ts'] = pd.to_datetime(basis_skin_temperature['ts'], dayfirst=True)
    basis_skin_temperature['value'] = pd.to_numeric(basis_skin_temperature['value'])
    
    info['basis_skin_temperature'] = [0.0] * info.shape[0]
    j = 0
    for i in range(info.shape[0]):
        curr_values = []
        while (j < basis_skin_temperature.shape[0]) and (basis_skin_temperature.loc[j, 'ts'] <= info.loc[i, 'time']):
            curr_values.append(basis_skin_temperature.loc[j, 'value'])
            j += 1
        info.loc[i, 'basis_skin_temperature'] = np.mean(curr_values) if curr_values else np.nan
    
    
    ## air temperature
    basis_air_temperature = []
    for item in root.findall("basis_air_temperature")[0]:
        basis_air_temperature.append([item.attrib['ts'], item.attrib['value']])
    basis_air_temperature = pd.DataFrame(basis_air_temperature)
    basis_air_temperature.columns = ['ts', 'value']
    basis_air_temperature['ts'] = pd.to_datetime(basis_air_temperature['ts'], dayfirst=True)
    basis_air_temperature['value'] = pd.to_numeric(basis_air_temperature['value'])
    
    info['basis_air_temperature'] = [0.0] * info.shape[0]
    j = 0
    for i in range(info.shape[0]):
        curr_values = []
        while (j < basis_air_temperature.shape[0]) and (basis_air_temperature.loc[j, 'ts'] <= info.loc[i, 'time']):
            curr_values.append(basis_air_temperature.loc[j, 'value'])
            j += 1
        info.loc[i, 'basis_air_temperature'] = np.mean(curr_values) if curr_values else np.nan
    
    
    ## step count
    basis_steps = []
    for item in root.findall("basis_steps")[0]:
        basis_steps.append([item.attrib['ts'], item.attrib['value']])
    basis_steps = pd.DataFrame(basis_steps)
    basis_steps.columns = ['ts', 'value']
    basis_steps['ts'] = pd.to_datetime(basis_steps['ts'], dayfirst=True)
    basis_steps['value'] = pd.to_numeric(basis_steps['value'])
    
    info['basis_steps'] = [0.0] * info.shape[0]
    j = 0
    for i in range(info.shape[0]):
        curr_values = []
        while (j < basis_steps.shape[0]) and (basis_steps.loc[j, 'ts'] <= info.loc[i, 'time']):
            curr_values.append(basis_steps.loc[j, 'value'])
            j += 1
        info.loc[i, 'basis_steps'] = np.mean(curr_values) if curr_values else np.nan
    
    
    ## times when the Basis band reported that the subject was asleep, sleep quality
    basis_sleep = []
    for item in root.findall("basis_sleep")[0]:
        basis_sleep.append([item.attrib['tbegin'], item.attrib['tend'], item.attrib['quality']])
    basis_sleep = pd.DataFrame(basis_sleep)
    basis_sleep.columns = ['ts_begin', 'ts_end', 'value']
    basis_sleep['ts_begin'] = pd.to_datetime(basis_sleep['ts_begin'], dayfirst=True)
    basis_sleep['ts_end'] = pd.to_datetime(basis_sleep['ts_end'], dayfirst=True)
    basis_sleep['value'] = pd.to_numeric(basis_sleep['value'])
    
    info['basis_sleep'] = [0.0] * info.shape[0]
    j = 0
    for i in range(info.shape[0]):
        curr_values = []
        while (j < basis_sleep.shape[0]) and (basis_sleep.loc[j, 'ts_begin'] < info.loc[i, 'time']):
            curr_values.append(basis_sleep.loc[j, 'value'])
            if basis_sleep.loc[j, 'ts_end'] <= info.loc[i, 'time']:
                j += 1
            else:
                break
        info.loc[i, 'basis_sleep'] = np.mean(curr_values) if curr_values else 0
    ## only use the automatically tracked sleep quality
    # info['sleep'] = (info['sleep'] + info['basis_sleep']/100)
    info['sleep'] = info['basis_sleep']


    ## fill NA/NaN values by propagating the last valid observation to next valid
    fill_cols = ['basis_heart_rate', 'basis_gsr', 'basis_skin_temperature', 'basis_air_temperature', 'basis_steps']
    info[fill_cols] = info[fill_cols].ffill(axis=0)
    info.drop(columns='basis_sleep', inplace=True)

    
    if n == 0:
        write_header = True
    else:
        write_header = False
    info.to_csv(filename, header=write_header, mode='a', index=False)


# %%
