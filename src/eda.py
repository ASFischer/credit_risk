import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------
# TARGET
# -------------------------
def plot_target(df):
    default = df['default'].value_counts()
    
    fig, ax = plt.subplots()
    ax.bar(default.values)

    ax.set_title('Default Distribution')
    ax.set_xlabel('Default')
    ax.set_ylabel('Count')

    return fig


# -------------------------
# DEMOGRAPHICS
# -------------------------
def plot_demographics(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df['age'], bins=30)
    axes[0].set_title('Age')

    vc_sex = df['sex'].value_counts()
    axes[1].bar(vc_sex.index, vc_sex.values)
    axes[1].set_title('Sex')

    vc_edu = df['education'].value_counts()
    axes[2].bar(vc_edu.index, vc_edu.values)
    axes[2].set_title('Education')

    plt.tight_layout()
    return fig


# -------------------------
# CREDIT LIMIT
# -------------------------
def plot_credit_limit(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df['limit_bal'], bins=50)
    axes[0].set_title('Limit Balance')

    df.boxplot(column='limit_bal', by='default', ax=axes[1])
    axes[1].set_title('Limit vs Default')

    plt.suptitle('')
    plt.tight_layout()

    return fig


# -------------------------
# PAYMENT HISTORY
# -------------------------
def plot_payment_history(df):
    pay_cols = ['pay_0','pay_2','pay_3','pay_4','pay_5','pay_6']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, col in enumerate(pay_cols):
        vc = df[col].value_counts().sort_index()

        row, col_idx = divmod(i, 3)
        axes[row, col_idx].bar(vc.index, vc.values)
        axes[row, col_idx].set_title(col)

    plt.tight_layout()
    return fig


# -------------------------
# FINANCIALS
# -------------------------
def plot_financials(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0,0].hist(df['bill_amt1'], bins=50)
    axes[0,0].set_title('Bill Amount 1')

    axes[0,1].hist(df['pay_amt1'], bins=50)
    axes[0,1].set_title('Payment Amount 1')

    # Feature rápida
    utilization = df['bill_amt1'] / df['limit_bal']
    axes[1,0].hist(utilization.replace([np.inf, -np.inf], np.nan).dropna(), bins=50)
    axes[1,0].set_title('Utilization')

    axes[1,1].boxplot([utilization[df['default']==0].dropna(),
                      utilization[df['default']==1].dropna()],
                     labels=['No Default', 'Default'])
    axes[1,1].set_title('Utilization vs Default')

    plt.tight_layout()
    return fig


# -------------------------
# CORRELATION
# -------------------------
def plot_correlation(df):
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, aspect='auto')

    fig.colorbar(cax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    return fig