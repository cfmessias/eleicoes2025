import pandas as pd
from dataframe import *

df_completo, simbolos= carregar_dados()

anos = sorted(df_completo['Ano'].unique(), reverse=True)
distritos = sorted(df_completo['Distrito'].unique())

df_completo_previsoes = compara_previsoes()


df_completo_previsoes.to_csv("df_completo_previsoes.csv", index=False,sep=';',encoding='utf-8-sig')

df_resultados_partido_ano,df_resultados_partido_distrito_ano = resultados_partido_ano()
df_resultados_partido_ano['Ano'] = df_resultados_partido_ano['Ano'].astype(str)
df_resultados_partido_ano.to_csv("df_resultados_partido_ano.csv", index=False,sep=';',encoding='utf-8-sig')
df_resultados_partido_distrito_ano.to_csv("df_resultados_partido_distrito_ano.csv", index=False,sep=';',encoding='utf-8-sig')

df_dados_filtros=dados_filtros()

df_dados_filtros.to_csv("df_dados_filtros.csv")

df_ano_2025=df_completo[(df_completo['Ano']=='2025')]
df_ano_2025.to_csv("df_ano_2025.csv")

df_cores_partidos=cores_partidos()
