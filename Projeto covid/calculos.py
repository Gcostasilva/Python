import pandas as pd

def outlier_sup(dados):
    '''
        Função para calcular o Outlier Superior sendo a função
        Limite Superior = Terceiro Quartil + 1,5 * (Terceiro Quartil – Primeiro Quartil)
        Calculo
        return dados.quantile(0.75) + 1.5 * (dados.quantile(0.75) - dados.quantile(0.25))
    '''
    return dados.quantile(0.75) + 1.5 * (dados.quantile(0.75) - dados.quantile(0.25))


def outlier_inf(dados):
    '''
        Função para calcular o Outlier Inferior sendo a função
        
        Limite Inferior = Primeiro Quartil – 1,5 * (Terceiro Quartil – Primeiro Quartil)
        
        Calculo
        
        return dados.quantile(0.25) - 1.5 * (dados.quantile(0.75) - dados.quantile(0.25))
    '''
    return dados.quantile(0.25) - 1.5 * (dados.quantile(0.75) - dados.quantile(0.25))

