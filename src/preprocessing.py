import pandas as pd


def rename_columns(df):
    column_mapping = {
        'Unnamed: 0': 'id',
        'X1': 'limit_bal',
        'X2': 'sex',
        'X3': 'education',
        'X4': 'marriage',
        'X5': 'age',
        'X6': 'pay_1',
        'X7': 'pay_2',
        'X8': 'pay_3',
        'X9': 'pay_4',
        'X10': 'pay_5',
        'X11': 'pay_6',
        'X12': 'bill_amt1',
        'X13': 'bill_amt2',
        'X14': 'bill_amt3',
        'X15': 'bill_amt4',
        'X16': 'bill_amt5',
        'X17': 'bill_amt6',
        'X18': 'pay_amt1',
        'X19': 'pay_amt2',
        'X20': 'pay_amt3',
        'X21': 'pay_amt4',
        'X22': 'pay_amt5',
        'X23': 'pay_amt6',
        'Y': 'default'
    }
    return df.rename(columns=column_mapping)



def transform_variables(df):
    df = df.copy()

    #Numéricas reais (valores contínuos)
    numeric_cols = [
        'limit_bal',
        'sex',
        'education',
        'marriage',
        'age',
        
        'bill_amt1','bill_amt2','bill_amt3',
        'bill_amt4','bill_amt5','bill_amt6',
        
        'pay_amt1','pay_amt2','pay_amt3',
        'pay_amt4','pay_amt5','pay_amt6',
        
        'pay_1','pay_2','pay_3',
        'pay_4','pay_5','pay_6'
    ]

    #Converter para numérico
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    #Target (garantir inteiro)
    df['default'] = pd.to_numeric(df['default'], errors='coerce').astype('int')

    return df


def clean_data(df):
    df = df.copy()
    df['education'] = df['education'].apply(lambda x: 4 if x in [0, 5, 6] else x)
    df['marriage'] = df['marriage'].replace(0, 3)
    return df

def replace_to_zero(df, col):
    filter = (df[col] == -2) | (df[col] == -1) | (df[col] == 0)
    df.loc[filter, col] = 0
    return df

def process_status(df):
    df = df.copy()
    
    pay_columns = ['pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    
    for i in pay_columns:
        df = replace_to_zero(df, i)
        
    return df