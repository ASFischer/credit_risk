# 💳 Credit Card Default Prediction

Modelo preditivo de inadimplência em cartão de crédito com foco em **orientação a negócio**: em vez de maximizar métricas técnicas isoladas, o projeto calibra o threshold de decisão segundo premissas explícitas de custo assimétrico — porque um calote não detectado custa muito mais do que uma recusa indevida.

---

## 📋 Sumário

- [Contexto do Problema](#-contexto-do-problema)
- [Dataset](#-dataset)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Pipeline](#-pipeline)
- [Resultados](#-resultados)
- [Cenários de Negócio](#-cenários-de-negócio)
- [Explicabilidade (SHAP)](#-explicabilidade-shap)
- [Como Executar](#-como-executar)
- [Tecnologias](#-tecnologias)

---

## 🎯 Contexto do Problema

Em concessão de crédito, os dois tipos de erro têm custos **assimétricos**:

| Erro | Situação | Consequência |
|------|----------|--------------|
| **Falso Negativo (FN)** | Modelo aprova → cliente dá calote | Prejuízo real de capital |
| **Falso Positivo (FP)** | Modelo recusa → cliente pagaria | Receita perdida (custo de oportunidade) |

Um FN custa ~16× mais do que um FP nas premissas adotadas. Isso muda completamente como o modelo deve ser avaliado e calibrado.

> **Pergunta central:** a partir de qual probabilidade de inadimplência devemos recusar o crédito?

A resposta depende do **perfil de risco do banco** — e é exatamente isso que este projeto simula por meio de três cenários de conservadorismo.

---

## 📊 Dataset

**UCI Default of Credit Card Clients** — Yeh & Lien (2009)

| Atributo | Valor |
|----------|-------|
| Fonte | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| Observações | 30.000 clientes |
| Features brutas | 23 |
| Features após engenharia | 29 |
| Período | Abril – Setembro de 2005 |
| Contexto | Banco taiwanês |
| Variável alvo | `default` (1 = inadimplente no mês seguinte) |
| Desbalanceamento | ~78% adimplentes / ~22% inadimplentes |

### Dicionário de variáveis

| Grupo | Variável | Descrição |
|-------|----------|-----------|
| Perfil | `limit_bal` | Limite de crédito concedido (NT$) |
| Perfil | `sex` | Sexo (1 = M, 2 = F) |
| Perfil | `education` | Escolaridade (1 = Pós-Grad, 2 = Universidade, 3 = Ens. Médio, 4 = Outros) |
| Perfil | `marriage` | Estado civil (1 = Casado, 2 = Solteiro, 3 = Outros) |
| Perfil | `age` | Idade (anos) |
| Pagamento | `pay_1` … `pay_6` | Status de pagamento (set–abr/2005); valores > 0 = meses de atraso |
| Fatura | `bill_amt1` … `bill_amt6` | Valor da fatura mensal (NT$) |
| Pagamento | `pay_amt1` … `pay_amt6` | Valor efetivamente pago por mês (NT$) |
| **Alvo** | `default` | 1 = inadimplente em out/2005 |

---

## 📁 Estrutura do Projeto

```
credit-default/
│
├── data/
│   └── credit.csv                  # Dataset bruto
│
├── notebooks/
│   ├── eda.ipynb                   # Análise exploratória completa
│   └── modeling.ipynb              # Treinamento, avaliação e SHAP
│
├── src/
│   ├── preprocessing.py            # Renomeação, tipagem, limpeza e recodificação
│   ├── features.py                 # Engenharia de variáveis derivadas
│   └── modeling.py                 # Função de calibração de threshold por Recall
│
├── models/
│   └── xgb_credit_model.pkl        # Modelo treinado (gerado ao executar modeling.ipynb)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Pipeline

```
Raw CSV
   │
   ▼
rename_columns()          ← nomes semânticos (X1→limit_bal, Y→default, …)
   │
   ▼
transform_variables()     ← conversão para tipos numéricos
   │
   ▼
clean_data()              ← agrupa categorias espúrias em education e marriage
   │
   ▼
process_status()          ← unifica -2, -1, 0 → 0 em pay_1…pay_6
   │
   ▼
feature_engineering()     ← cria util_ratio, pay_ratio_1/2/3, n_delays, max_delay
   │
   ▼
train_test_split()        ← 70/30, estratificado por default
   │
   ▼
XGBClassifier             ← scale_pos_weight=10 (simula custo FN/FP ≈ 16×)
   │
   ▼
threshold_para_recall()   ← calibra ponto de corte por meta de negócio
   │
   ▼
Análise de custo + SHAP
```

---

## 📐 Engenharia de Variáveis

Seis features foram derivadas para capturar sinais que as variáveis brutas não expressam diretamente:

| Feature | Fórmula | Interpretação |
|---------|---------|---------------|
| `util_ratio` | `avg_bill / (limit_bal + 1)` | Percentual do limite sendo utilizado — quanto mais perto de 1, maior o estresse financeiro |
| `pay_ratio_1` | `pay_amt1 / (bill_amt2 + 1)` | Proporção da fatura de ago/2005 que foi paga em set/2005 |
| `pay_ratio_2` | `pay_amt2 / (bill_amt3 + 1)` | Proporção da fatura de jul/2005 paga em ago/2005 |
| `pay_ratio_3` | `pay_amt3 / (bill_amt4 + 1)` | Proporção da fatura de jun/2005 paga em jul/2005 |
| `n_delays` | `Σ(pay_i > 0)` | Quantidade de meses com atraso nos últimos 6 meses |
| `max_delay` | `max(pay_1…pay_6)` | Pior atraso registrado no período (severidade) |

---

## 📈 Resultados

### Métricas base (threshold = 0.50)

| Métrica | Valor |
|---------|-------|
| ROC-AUC | 0.78 |
| PR-AUC | 0.54 |
| Recall (inadimplente) | — |
| Precision (inadimplente) | — |

> O PR-AUC é a métrica principal: em datasets desbalanceados, o ROC-AUC pode ser enganosamente otimista por ignorar a alta proporção de verdadeiros negativos.

---

## 💼 Cenários de Negócio

O threshold é calibrado via `threshold_para_recall()` para atingir uma **meta mínima de Recall**, maximizando a Precision dentro dessa restrição.

| Cenário | Meta de Recall | Perfil de banco indicado |
|---------|---------------|--------------------------|
| **Moderado** | ≥ 75% | Banco com margem folgada, aceita risco moderado |
| **Conservador** | ≥ 80% | Banco de varejo — melhor equilíbrio custo/captura |
| **Muito conservador** | ≥ 90% | Banco com histórico de inadimplência alto |

### Premissas financeiras (ajustáveis)

```python
VALOR_MEDIO_EMPRESTIMO = 50_000   # NT$
TAXA_RECUPERACAO       = 0.20     # 20% recuperado em cobrança
RECEITA_POR_RECUSA     = 2_500    # NT$ de juros perdidos por FP

custo_fn = 40_000  # NT$ por calote não detectado
custo_fp =  2_500  # NT$ por bom cliente recusado
```

> O cenário **Conservador (Recall ≥ 80%)** apresentou o menor custo total nas premissas adotadas — coerente com a proporção FN/FP de ~16:1.

---

## 🔍 Explicabilidade (SHAP)

O SHAP (`TreeExplainer`) é aplicado para tornar o modelo auditável — requisito em ambientes regulatórios (LGPD, normas do Banco Central).

**Summary plot (importância global):**  
Cada ponto representa um cliente. Vermelho = valor alto da feature, azul = valor baixo. Posição no eixo X indica o quanto a feature empurra o modelo para default (direita) ou não (esquerda).

**Waterfall plot (explicação individual):**  
Decompõe a previsão de um cliente específico em contribuições positivas e negativas de cada feature — útil para justificar uma recusa ao cliente ou ao comitê de crédito.

**Top features por importância SHAP (ordem aproximada):**

1. `pay_1` — status de pagamento do mês mais recente
2. `pay_2` — status de pagamento do mês anterior
3. `max_delay` — pior atraso registrado
4. `n_delays` — frequência de atrasos
5. `util_ratio` — utilização do limite
6. `pay_ratio_1` — proporção da fatura paga
7. `limit_bal` — limite concedido

> Variáveis demográficas (`sex`, `education`, `marriage`) contribuem marginalmente — comportamento de pagamento é o sinal dominante.

---

## 🚀 Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/credit-default.git
cd credit-default
```

### 2. Crie um ambiente virtual e instale as dependências

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Execute os notebooks em ordem

```bash
jupyter notebook
```

| Notebook | O que faz |
|----------|-----------|
| `notebooks/eda.ipynb` | Análise exploratória, distribuições, correlações, insights |
| `notebooks/modeling.ipynb` | Treino, avaliação, cenários de negócio, SHAP, exportação do modelo |

> O modelo treinado é salvo automaticamente em `models/xgb_credit_model.pkl` ao executar a última célula do `modeling.ipynb`.

### 4. Usando o modelo salvo

```python
import joblib
import pandas as pd
from src.preprocessing import rename_columns, transform_variables, clean_data, process_status
from src.features import feature_engineering

pipe = joblib.load('models/xgb_credit_model.pkl')

# Prepare seu DataFrame com as mesmas colunas do dataset original
# df_novo = ...

proba = pipe.predict_proba(df_novo)[:, 1]   # probabilidade de default
pred  = (proba >= 0.35).astype(int)         # threshold conservador (~Recall 80%)
```

---

## 🛠️ Tecnologias

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| `pandas` | 3.0.2 | Manipulação de dados |
| `numpy` | 2.4.4 | Operações numéricas |
| `scikit-learn` | 1.8.0 | Split, métricas, pipeline |
| `xgboost` | 3.2.0 | Classificador principal |
| `imbalanced-learn` | 0.14.1 | Pipeline com suporte a desbalanceamento |
| `shap` | 0.52.0 | Explicabilidade do modelo |
| `matplotlib` | 3.10.9 | Visualizações |
| `seaborn` | 0.13.2 | Visualizações estatísticas |
| `joblib` | 1.5.3 | Serialização do modelo |

---

## 📚 Referências

- YEH, I.-C.; LIEN, C.-H. *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, v. 36, n. 2, p. 2473–2480, 2009.
- LUNDBERG, S. M.; LEE, S.-I. *A unified approach to interpreting model predictions.* NeurIPS, 2017.
- CHEN, T.; GUESTRIN, C. *XGBoost: a scalable tree boosting system.* KDD, 2016.
- HAND, D. J. *Classifier technology and the illusion of progress.* Statistical Science, v. 21, n. 1, 2006.
- SAITO, T.; REHMSMEIER, M. *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.* PLOS ONE, 2015.

---

## 📄 Licença

Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
