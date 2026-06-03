# 💳 Credit Card Default Prediction

Modelo preditivo de inadimplência em cartão de crédito desenvolvido com dados reais de clientes taiwaneses (UCI, 2005). O projeto cobre o pipeline completo de Machine Learning com **orientação a negócio real**: análise exploratória, pré-processamento, engenharia de features, modelagem com custo assimétrico e avaliação por cenários de política de crédito.

---

## 📁 Estrutura do Projeto

```
credit-default/
│
├── data/
│   └── credit.csv                        # Dataset original (UCI)
│
├── notebooks/
│   ├── eda.ipynb                         # Análise exploratória completa
│   └── modeling.ipynb                    # Modelagem orientada a negócio
│
├── src/
│   ├── preprocessing.py                  # Renomeação, tipagem e limpeza
│   └── features.py                       # Engenharia de features derivadas
│
├── plots/                                # Gráficos gerados pelos notebooks
│   ├── target_distribution.png
│   ├── demographic_default_rate.png
│   ├── pay_status_default_rate.png
│   ├── temporal_evolution.png
│   ├── correlation_heatmap_top.png
│   ├── roc_pr_curves.png
│   └── confusion_matrix_business.png
│
├── venv/                                 # Ambiente virtual Python
└── README.md
```

---

## 📊 Dataset

| Atributo | Detalhe |
|----------|---------|
| **Fonte** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) |
| **Período** | Abril a Setembro de 2005 |
| **Registros** | 30.000 clientes |
| **Features originais** | 23 |
| **Target** | `default` — inadimplência no mês seguinte (1 = sim, 0 = não) |
| **Desbalanceamento** | ~22% inadimplentes / ~78% adimplentes |

### Variáveis principais

| Grupo | Variáveis | Descrição |
|-------|-----------|-----------|
| Perfil | `limit_bal`, `age`, `sex`, `education`, `marriage` | Dados cadastrais do cliente |
| Status de pagamento | `pay_1` … `pay_6` | Meses de atraso em set–abr/2005 (0 = em dia) |
| Faturas | `bill_amt1` … `bill_amt6` | Saldo devedor mensal em NT$ |
| Pagamentos | `pay_amt1` … `pay_amt6` | Valor pago mensalmente em NT$ |

---

## ⚙️ Pré-processamento (`src/preprocessing.py`)

| Função | O que faz |
|--------|-----------|
| `rename_columns` | Mapeia `X1`…`X23`, `Y` para nomes legíveis |
| `transform_variables` | Converte todas as colunas para tipos numéricos corretos |
| `clean_data` | Agrupa categorias inválidas: `education` (0,5,6 → 4), `marriage` (0 → 3) |
| `process_status` | Recodifica `pay_*`: valores -2, -1 e 0 tratados como 0 (em dia) |

---

## 🔧 Engenharia de Features (`src/features.py`)

Features criadas para capturar padrões financeiros que as colunas originais não expressam diretamente:

| Feature | Fórmula | Intuição |
|---------|---------|----------|
| `avg_bill` | Média de `bill_amt1`…`bill_amt6` | Nível médio de endividamento |
| `util_ratio` | `avg_bill / (limit_bal + 1)` | % do limite em uso — quanto maior, mais risco |
| `pay_ratio_1` | `pay_amt1 / (bill_amt2 + 1)` | Pagou quanto da fatura de ago/05? Próximo de 0 = rolando dívida |
| `pay_ratio_2` | `pay_amt2 / (bill_amt3 + 1)` | Idem para jul/05 |
| `pay_ratio_3` | `pay_amt3 / (bill_amt4 + 1)` | Idem para jun/05 |
| `n_delays` | `sum(pay_* > 0)` | Quantos meses dos últimos 6 tiveram atraso |
| `max_delay` | `max(pay_1…pay_6)` | Pior mês de atraso — captura severidade |

> **Por que criar features?** Modelos baseados em árvore fazem cortes por coluna individual. Razões como `pay_amt / bill_amt` e agregações como `n_delays` precisam ser construídas explicitamente — o modelo não as infere com a mesma eficiência.

---

## 🏦 Contexto de Negócio

Em concessão de crédito real, os dois tipos de erro têm custos **assimétricos**:

| Erro | Situação | Consequência |
|------|----------|--------------|
| **Falso Negativo (FN)** | Modelo aprova → cliente dá calote | Perda do principal emprestado + custos de cobrança + provisão regulatória |
| **Falso Positivo (FP)** | Modelo recusa → cliente pagaria | Perda apenas da receita de juros (custo de oportunidade) |

> Um calote custa tipicamente **5× a 20×** mais do que uma recusa indevida.  
> Por isso o modelo é calibrado para **maximizar Recall**, aceitando um nível controlado de Falsos Positivos em troca.

---

## 🤖 Modelagem (`notebooks/modeling.ipynb`)

### Pipeline

```
Split Estratificado (70/30, stratify=y)
            ↓
      XGBClassifier
  · n_estimators     = 300
  · learning_rate    = 0.05
  · max_depth        = 5
  · subsample        = 0.8
  · colsample_bytree = 0.8
  · scale_pos_weight = 10        ← orientado a negócio
  · eval_metric      = 'aucpr'
            ↓
    Threshold Tuning por meta de Recall
            ↓
      Análise de Custo Financeiro
```

### Decisões de modelagem

**`stratify=y` no split**
Garante que a proporção de ~22% de defaults seja mantida igual no treino e no teste, evitando que o split aleatório concentre casos de default em apenas um dos conjuntos.

**`scale_pos_weight = 10`**
Valor orientado a negócio real. Cada erro em um inadimplente gera um gradiente 10× maior durante o treino, deslocando a fronteira de decisão para o lado conservador. Justificativa: um calote implica perda do principal + custos de cobrança + provisão regulatória, enquanto uma recusa indevida implica apenas perda da margem de juros — proporção de custo estimada em 10:1.

> O valor acadêmico equivalente seria `neg/pos ≈ 3.44`, que apenas equilibra as classes sem refletir o custo assimétrico real.

**`eval_metric = 'aucpr'`**
Otimiza a curva Precision-Recall internamente durante o treino — métrica mais adequada para datasets desbalanceados do que ROC-AUC.

---

## 📈 Threshold Tuning por Cenário de Negócio

Em vez de usar threshold 0.5 (padrão) ou o threshold de F1 máximo (acadêmico), o modelo define o threshold a partir de uma **meta de Recall** estabelecida pela política de risco:

```python
def threshold_para_recall(meta_recall, precision_arr, recall_arr, thresholds):
    """Retorna o threshold que atinge a meta de Recall com maior Precision possível."""
    indices = recall_arr[:-1] >= meta_recall
    melhor  = precision_arr[:-1][indices].argmax()
    return thresholds[indices][melhor]
```

### Cenários disponíveis

| Cenário | Meta de Recall | Perfil do banco |
|---------|---------------|-----------------|
| Moderado | ≥ 75% | Fintech com apetite de risco — aceita mais risco para crescer carteira |
| Conservador | ≥ 80% | Banco de varejo — equilibra captura e falsos alarmes |
| Muito conservador | ≥ 90% | Banco com histórico de alta inadimplência — prioriza capital |

---

## 💰 Análise de Custo Financeiro

O notebook traduz FN e FP em impacto financeiro estimado usando premissas ajustáveis:

| Premissa | Valor padrão | Descrição |
|----------|-------------|-----------|
| Valor médio emprestado | NT$ 50.000 | Principal em risco por cliente |
| Taxa de recuperação | 20% | % recuperado via cobrança em caso de calote |
| Receita perdida por recusa | NT$ 2.500 | Juros não recebidos por FP |

**Custo por FN** = `50.000 × (1 − 0.20)` = NT$ 40.000  
**Custo por FP** = NT$ 2.500  
**Proporção real de custo** = 16:1

O cenário com **menor custo total** (`FN × custo_FN + FP × custo_FP`) é o recomendado para implantação.

---

## 🔍 Métricas de Avaliação

| Métrica | Relevância |
|---------|------------|
| **Recall (classe 1)** | Métrica principal — % de inadimplentes reais capturados |
| **PR-AUC** | Desempenho global sem inflação pelo número de TN |
| **ROC-AUC** | Referência geral de separabilidade |
| **Precision (classe 1)** | Taxa de falsos alarmes — impacta volume de recusas indevidas |
| **Custo total estimado** | Traduz os erros em impacto financeiro direto |
| **Matriz de Confusão** | Visualização de TP, TN, FP e FN por cenário |

---

## 🔎 Explicabilidade — SHAP

O modelo utiliza SHAP (SHapley Additive exPlanations) para justificar cada decisão — requisito regulatório em concessão de crédito:

- **Summary plot:** importância global das features e direção do impacto
- **Waterfall plot:** contribuição de cada feature para um cliente específico

```python
explainer   = shap.TreeExplainer(pipe['xgb'])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns.tolist())
```

---

## 🚀 Como Executar

### 1. Clonar e configurar o ambiente

```bash
git clone https://github.com/seu-usuario/credit-default.git
cd credit-default

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Dependências principais

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
scipy
shap
jupyter
```

### 3. Executar os notebooks em ordem

```bash
jupyter notebook notebooks/eda.ipynb        # 1. Análise exploratória
jupyter notebook notebooks/modeling.ipynb   # 2. Modelagem e avaliação
```

> O dataset deve estar em `data/credit.csv`. Os módulos `src/preprocessing.py` e `src/features.py` são importados automaticamente pelos notebooks via `sys.path.append('../src')`.

---

## 🔬 Uso do Modelo em Produção

```python
import pandas as pd

THRESHOLD_PRODUCAO = 0.35  # substituir pelo threshold do cenário escolhido

def score_cliente(features: dict, modelo=pipe, threshold=THRESHOLD_PRODUCAO) -> dict:
    """
    Pontua um novo cliente e retorna decisão de crédito.

    Args:
        features: dicionário com as features do cliente (após preprocessing e feature engineering)
        modelo:   pipeline treinado
        threshold: ponto de corte da política de risco vigente

    Returns:
        dict com probabilidade de default, decisão e threshold utilizado
    """
    X_novo = pd.DataFrame([features])
    prob   = modelo.predict_proba(X_novo)[0, 1]
    return {
        'probabilidade_default': round(prob, 4),
        'decisao':               'RECUSAR' if prob >= threshold else 'APROVAR',
        'threshold_utilizado':   threshold
    }
```

---

## 🔍 Principais Achados da EDA

- **Desbalanceamento:** ~22% de inadimplentes exige tratamento explícito — `scale_pos_weight=10` no treino
- **Variáveis de atraso** (`pay_1`–`pay_6`): mais correlacionadas com default — comportamento recente é o maior sinal de risco
- **Limite de crédito** (`limit_bal`): inadimplentes têm limites significativamente menores — o banco já percebia o risco na concessão
- **Faturas** (`bill_amt*`): alta multicolinearidade entre si — agregadas em `avg_bill` e `util_ratio`
- **Pagamentos** (`pay_amt*`): a diferença entre grupos está em *quanto pagam*, não em quanto devem — capturado por `pay_ratio_*`

---

## 📚 Referências

- Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.
- [UCI ML Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16.
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
