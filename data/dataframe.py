import unicodedata
import pandas as pd

def normalizar_texto(texto):
    if pd.isna(texto):
        return texto
    return unicodedata.normalize('NFC', str(texto).strip())


def carregar_dados():
    df_completo = pd.read_csv('data/previsoes_finais_com_sondagens.csv', sep=';', decimal=',', encoding='utf-8-sig')
    simbolos = pd.read_csv('data/siglas_partidos.csv', sep=';', decimal=',')
    # Limpar espaços e normalizar encoding
    df_completo['Partido'] = df_completo['Partido'].str.strip().str.normalize('NFC')
    simbolos['Partido'] = simbolos['Partido'].str.strip().str.normalize('NFC')
    df_completo['Ano'] = df_completo['Ano'].str.strip().str.normalize('NFC')

    return df_completo, simbolos

def compara_previsoes():
    # Desempacota os dois DataFrames
    df_completo, _ = carregar_dados()

    # Garante que a coluna 'Ano' é texto
    df_completo['Ano'] = df_completo['Ano'].astype(str).str.strip().str.replace('\u200b', '').str.replace('\xa0', '').str.normalize('NFKD')

   
    # Filtra previsões
    df_completo_previsoes = df_completo.loc[
        (df_completo['Ano'].isin(['Histórico', 'Sondagens', 'Expetativas'])) & 
        (df_completo['Distrito'] == ' Totais'),
        ['Ano', 'Partido','Cor', 'Percentual','Distrito']
    ]

    return df_completo_previsoes


def resultados_partido_ano():

    df_completo, _ = carregar_dados()

    # Garante que a coluna 'Ano' é texto
    df_completo['Ano'] = df_completo['Ano'].astype(str)
   
    # Filtra previsões
    df_resultados_partido_ano = df_completo.loc[
        (df_completo['Ano'].isin(['2019', '2022', '2024','2025'])) & 
        (df_completo['Distrito'] == ' Totais'), 
        ['Ano','Inscritos','Votantes','Brancos','Nulos','Partido','Votos','Mandatos','Percentual','Abstenção','Cor']
    ]
    
    df_resultados_partido_distrito_ano = df_completo.loc[
    (df_completo['Ano'].isin(['2019', '2022', '2024','2025'])) , 
    ['Ano','Distrito','Inscritos','Votantes','Brancos','Nulos','Partido','Votos','Mandatos','Percentual','Abstenção','Cor']
    ]

    df_resultados_partido_distrito_2025 = df_completo.loc[
    (df_completo['Ano'].isin(['2025'])) , 
    ['Ano','Distrito','Inscritos','Votantes','Brancos','Nulos','Partido','Votos','Mandatos','Percentual','Abstenção','Cor']
    ]
    return df_resultados_partido_ano,df_resultados_partido_distrito_ano,df_resultados_partido_distrito_2025

def dados_filtros():

    df_completo, _ = carregar_dados()
    df_completo['Ano'] = df_completo['Ano'].apply(normalizar_texto)
   
    df_anos = df_completo[['Ano']].drop_duplicates()
    df_distritos = df_completo[['Distrito']].drop_duplicates()
 
    return df_anos, df_distritos


def cores_partidos():
    df_completo, _ = carregar_dados()
    df_cores_partidos = df_completo[['Partido', 'Cor']]
    
    return df_cores_partidos
