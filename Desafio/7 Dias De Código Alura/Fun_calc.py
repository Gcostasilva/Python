import pandas as pd

def Valores_Variavel(df, variável):
    '''
    Retorna qual a distribuição da variável a ser analizada analizando a coluna id_emprestimo como valor.

    df = DataFrame

    Variável = Coluna alvo da análise
    '''
    resumo = df.groupby([variável])['id_emprestimo'].count().reset_index()
    pd.dataframe(resumo)
    return resumo