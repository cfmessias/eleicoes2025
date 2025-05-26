from data.dataframe import *
import pandas as pd


def dataframe_filtrado(ano_selecionado, distrito_selecionado):
    
    df_completo, simbolos=carregar_dados()
    df_completo_previsoes=compara_previsoes()
    df_cores_partidos=cores_partidos()
    df_resultados_partido_ano,df_resultados_partido_distrito_ano,df_resultados_partido_distrito_2025=resultados_partido_ano()
    df_ano_2025=df_completo[(df_completo['Ano']=='2025')]
    df_ano_2024=df_completo[(df_completo['Ano']=='2024')]
    abstencao2025 = df_ano_2025['Abstenção'].dropna().unique()[0]
    brancos2025 = df_ano_2025['Brancos'].dropna().unique()[0]
    abstencao2024 = df_ano_2024['Abstenção'].dropna().unique()[0]
    brancos2024 = df_ano_2024['Brancos'].dropna().unique()[0]

    df_completo_previsoes['Partido'] = df_completo_previsoes['Partido'].str.strip()
    simbolos['Partido'] = simbolos['Partido'].str.strip()
    df_completo_previsoes['Ano'] = df_completo_previsoes['Ano'].str.strip()
    df_completo_previsoes['Distrito'] = df_completo_previsoes['Distrito'].str.strip()
    df_compara_previsoes = pd.merge(df_completo_previsoes, simbolos, on=['Partido'], how='inner')
    df_compara_previsoes['Percentual']=df_compara_previsoes['Percentual']*100
    df_historico_partidos = df_completo[df_completo['Ano'].isin(['2011', '2015', '2019', '2022', '2024'])]
    ordem_partidos = ["BE","CDU","L","PS","PAN","JPP","AD", "IL", "CH"]
    df_completo["Ordem"] = df_completo["Partido"].apply(lambda x: ordem_partidos.index(str(x)) if pd.notna(x) and str(x) in ordem_partidos else len(ordem_partidos)
)
    df_completo_filtrado=df_completo[(df_completo['Ano'] == ano_selecionado) & (df_completo['Distrito'] == distrito_selecionado)]
    df_completo_filtrado_cor_simbolo=pd.merge(simbolos, df_completo_filtrado, on=['Partido'], how='inner')
    df_historico_partidos_cor_simbolo=pd.merge(simbolos,df_historico_partidos, on=['Partido'], how='inner')

    df_completo_filtrado2025=df_completo[(df_completo['Ano'] == '2025') & (df_completo['Distrito'] == distrito_selecionado)]
    df_completo_filtrado_cor_simbolo2025=pd.merge(simbolos, df_completo_filtrado2025, on=['Partido'], how='inner')

    df_completo_filtrado20252024=df_completo[(df_completo['Ano'].isin(['2024','2025'])) & (df_completo['Distrito'] == distrito_selecionado)]
    df_completo_filtrado_cor_simbolo20252024=pd.merge(simbolos, df_completo_filtrado20252024, on=['Partido'], how='inner')

    df_historico_partidos_cor_simbolo_filtrado=df_historico_partidos_cor_simbolo[(df_historico_partidos_cor_simbolo['Distrito']==distrito_selecionado)]
    return (
        abstencao2025,
        brancos2025,
        abstencao2024,
        brancos2024,
        df_compara_previsoes,
        ordem_partidos,
        df_completo_filtrado,
        df_completo_filtrado_cor_simbolo,        
        df_resultados_partido_ano,
        df_resultados_partido_distrito_ano,
        df_resultados_partido_distrito_2025,
        df_completo_filtrado_cor_simbolo2025,
        df_completo_filtrado_cor_simbolo20252024,
        df_historico_partidos_cor_simbolo_filtrado,
        df_completo_previsoes,
        df_completo, 
        simbolos
    )