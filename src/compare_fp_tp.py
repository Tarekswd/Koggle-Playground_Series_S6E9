import numpy as np
import pandas as pd

train = pd.read_csv('playground-series-s6e9/train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
oof = np.load('oof_predictions.npy')

fp_mask = (y == 0) & (oof > 0.80)
fp = train[fp_mask]

print("False Positives (y=0, predicted > 0.80) categorical distribution:")
print("\nSubsidy Available:")
print(fp['Subsidy_Available'].value_counts(normalize=True))

print("\nRange Anxiety Level:")
print(fp['Range_Anxiety_Level'].value_counts(normalize=True))

print("\nHome Charging Possible:")
print(fp['Home_Charging_Possible'].value_counts(normalize=True))

print("\nCity Type:")
print(fp['City_Type'].value_counts(normalize=True))

print("\nCurrent Car Type:")
print(fp['Current_Car_Type'].value_counts(normalize=True))

print("\nCharging Stations Near Home (mean):", fp['Charging_Stations_Near_Home'].mean())
print("Charging Stations Near Work (mean):", fp['Charging_Stations_Near_Work'].mean())

# Compare to True Positives with oof > 0.80
tp_mask = (y == 1) & (oof > 0.80)
tp = train[tp_mask]
print("\n--- Compare to True Positives with oof > 0.80 ---")
print("TP Subsidy Available:", tp['Subsidy_Available'].value_counts(normalize=True).to_dict())
print("TP Range Anxiety:", tp['Range_Anxiety_Level'].value_counts(normalize=True).to_dict())
print("TP Home Charging:", tp['Home_Charging_Possible'].value_counts(normalize=True).to_dict())
print("TP City Type:", tp['City_Type'].value_counts(normalize=True).to_dict())
print("TP Current Car Type:", tp['Current_Car_Type'].value_counts(normalize=True).to_dict())
print("TP Charging Stations Near Home (mean):", tp['Charging_Stations_Near_Home'].mean())
print("TP Charging Stations Near Work (mean):", tp['Charging_Stations_Near_Work'].mean())
