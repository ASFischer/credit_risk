def threshold_para_recall(meta_recall, precision_arr, recall_arr, thresholds):
    """
    Dado uma meta de Recall mínimo, retorna o threshold que atinge essa meta
    com a maior Precision possível (menor taxa de falsos alarmes).
    """
    indices = recall_arr[:-1] >= meta_recall #Olha para todos os valores de Recall calculados pelo modelo e pergunta: "Este valor é maior ou igual à meta?"
    if not indices.any():
        return None, None, None #Aborta função caso não encontre um valor proposto pelo 'indices'
    melhor = precision_arr[:-1][indices].argmax() 
    '''
    Olha dentro das precisões encontradas os melhores valores encontrados no 'indices'
    Na prática: Ele diz: "Ok, temos 50 limiares diferentes que conseguem capturar os 90% de Recall que queremos.
    Mas qual deles faz isso gerando o menor número de alarmes falsos (maior Precisão)? Encontre a posição dele para mim".
    '''
    thr = thresholds[indices][melhor]
    '''
    Agora que sabemos a posição (melhor) do cenário ideal dentro do nosso subgrupo filtrado ([indices]),
    nós vamos na lista original de thresholds, aplicamos o mesmo filtro,
    e pegamos o valor exato do limiar de corte. 
    ''' 
    r   = recall_arr[:-1][indices][melhor]
    p   = precision_arr[:-1][indices][melhor]
    '''
    O que fazem: Exatamente a mesma lógica da linha acima. Vão nas listas de Recall e Precision, 
    aplicam o filtro dos que bateram a meta, e resgatam o valor exato que estava na melhor posição.
    '''
    return thr, r, p


'''
1. O que significa [:-1] em Python?
Em Python puro e no NumPy, isso é uma técnica chamada slicing (fatiamento).

O : antes da vírgula significa "comece do primeiro item e vá até o fim".

O -1 significa "o último item da lista".

Portanto, [:-1] é um atalho elegante para dizer: "Pegue todos os elementos desta lista, exceto o último".

2. Por que jogar o último elemento fora no Scikit-Learn?
O "problema" nasce quando você gera as matrizes usando a função clássica do Scikit-Learn chamada precision_recall_curve(). Quando você chama essa função, ela te devolve três listas:

Um array de Precision

Um array de Recall

Um array de Thresholds (os limiares de corte)

A "pegadinha" matemática é que o Scikit-Learn faz as listas de Precision e Recall virem sempre com 1 elemento a mais do que a lista de Thresholds. Se a lista de thresholds tem 400 itens, as de métricas terão 401.

Por que o Scikit-Learn faz isso?
Para garantir que, ao desenhar o gráfico da curva no Matplotlib ou Plotly, a linha sempre termine tocando o eixo Y. Ele adiciona artificialmente um "cenário extremo" no final das listas onde o limiar seria tão alto (tendendo ao infinito) que o modelo não faria nenhuma previsão positiva. Nesse cenário hipotético:

Recall é forçado para 0.0 (o modelo não acha nada).

Precision é forçada para 1.0 (quando não prevê nada, não comete falsos positivos).

O problema é que não existe um valor numérico real de threshold para esse ponto extremo.
'''