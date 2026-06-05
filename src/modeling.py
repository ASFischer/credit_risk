
# Dado uma meta de Recall mínimo, retorna o threshold que atinge essa meta com a maior Precision possível (menor taxa de falsos alarmes).
def threshold_para_recall(meta_recall, precision_arr, recall_arr, thresholds):

    indices = recall_arr[:-1] >= meta_recall # Olha para todos os valores de Recall calculados pelo modelo e pergunta: "Este valor é maior ou igual à meta?"

    if not indices.any():
        return None, None, None # Aborta função caso não encontre um valor proposto pelo 'indices'
    melhor = precision_arr[:-1][indices].argmax() # Encontra dentro de 'indices' o maior valor (argmax) de precision que atingiu a meta _recall
    thr = thresholds[indices][melhor] # Agora encontra o valor exato do threshold encontrado para o melhor valor de precision

    r   = recall_arr[:-1][indices][melhor]
    p   = precision_arr[:-1][indices][melhor] # Vão nas listas de Recall e Precision,  aplicam o filtro dos que bateram a meta, e resgatam o valor exato que estava na melhor posição.

    return thr, r, p


# Utilizar o [:-1] dentro da função é necessário, pois dentro do scikitlearn, ao chamar a função precision_recall_curve, é adicionado
# ao fim da lista um valor a mais. Por esse motivo, pedimos que seja lido todos os valores menos o último.