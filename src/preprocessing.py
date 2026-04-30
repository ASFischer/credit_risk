def rename_columns(df):
    column_mapping = {
        'X1': 'limit_bal',
        'X2': 'sex',
        'X3': 'education',
        'X4': 'marriage',
        'X5': 'age',
        'X6': 'pay_0',
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