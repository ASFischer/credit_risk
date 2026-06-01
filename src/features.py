def feature_engineering(df):
    df = df.copy()

    # Utilização do limite: quanto do crédito está sendo usado
    # Quanto mais próximo de 1.0, mais "espremido" o cliente está
    bill_cols = ['bill_amt1','bill_amt2','bill_amt3','bill_amt4','bill_amt5','bill_amt6']
    df['avg_bill'] = df[bill_cols].mean(axis=1)
    df['util_ratio'] = df['avg_bill'] / (df['limit_bal'] + 1)  # +1 evita divisão por zero

    # Razão pagamento/fatura: pagar pouco da fatura é sinal de risco
    # pay_amt1 refere-se ao pagamento feito no mês mais recente (set/2005)
    # bill_amt2 é a fatura do mês anterior (ago/2005), que é o que foi pago
    df['pay_ratio_1'] = df['pay_amt1'] / (df['bill_amt2'].abs() + 1)
    df['pay_ratio_2'] = df['pay_amt2'] / (df['bill_amt3'].abs() + 1)
    df['pay_ratio_3'] = df['pay_amt3'] / (df['bill_amt4'].abs() + 1)

    # Quantidade de meses com atraso nos últimos 6 meses
    pay_cols = ['pay_1','pay_2','pay_3','pay_4','pay_5','pay_6']
    df['n_delays'] = (df[pay_cols] > 0).sum(axis=1)

    # Atraso máximo: captura o pior momento do cliente
    df['max_delay'] = df[pay_cols].max(axis=1)

    return df