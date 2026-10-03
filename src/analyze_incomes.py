import pandas as pd
train = pd.read_csv('playground-series-s6e9/train.csv')
train['target'] = (train['Will_Buy_EV'] == 'Yes').astype(int)

print("--- Analysis of Exact Incomes ---")
for inc in [86095.0, 96749.0, 65800.0, 92372.0, 113631.0, 59901.0]:
    sub = train[train['Annual_Income_USD'] == inc]
    rate = sub['target'].mean()
    s_dict = sub['Subsidy_Available'].value_counts().to_dict()
    print(f"Income ${inc:,.0f}: count={len(sub)}, buy_rate={rate:.4f}, subs={s_dict}")
