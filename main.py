import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.colors as mcolors
import numpy as np
import os   
from io import BytesIO
import base64
from PIL import Image
from matplotlib.patches import Patch
import unicodedata
from utils.helpers import *
from utils.sidebar import aplicar_filtros_sidebar
from data.dataframe import *
from data.dataframe_filtrado import *

# Configurações iniciais
st.set_page_config(page_title="Eleições em Portugal", layout="wide")
#st.title("Legislativas 2025")
# Ler imagem como binário
with open("assets/bandeira.jpeg", "rb") as image_file:
    img_bytes = image_file.read()
    encoded_img = base64.b64encode(img_bytes).decode()


# Construir HTML com a imagem embutida
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/jpeg;base64,{encoded_img}" alt="Bandeira" style="height: 40px; margin-right: 15px;">
        <h2 style="margin: 0;">Legislativas 2025</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Leitura dos dados

#df_completo, simbolos= carregar_dados()
#df_completo_previsoes = compara_previsoes()
#df_resultados_partido_ano,df_resultados_partido_distrito_ano = resultados_partido_ano()


# Filtros
ano_selecionado, distrito_selecionado = aplicar_filtros_sidebar()

(
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
) = dataframe_filtrado(ano_selecionado, distrito_selecionado)


tabs = st.tabs(["📊Previsões", "📈Resultados","📜Histórico"])


# -----------------------------------------
# 1. PREVISÕES
# -----------------------------------------
with tabs[0]:
    
    previsao_opcao = st.radio("", [  "⚙️Tipo","🗣Partidos","🏛️Parlamento"], horizontal=True)

    if previsao_opcao == "⚙️Tipo":
        
        plot_previsoes(df_compara_previsoes)
        
        
    elif previsao_opcao == "🏛️Parlamento":
        
        col1, col2 = st.columns([3.8, 2])
        # Ordenar para manter consistência de layout
        with col1:
            ordem_partidos = ["BE","CDU","L","PS","PAN","JPP","AD", "IL", "CH"]
            df_completo_filtrado_cor_simbolo["Ordem"] = df_completo_filtrado_cor_simbolo["Partido"].apply(lambda x: ordem_partidos.index(x) if x in ordem_partidos else len(ordem_partidos))
            df_filtrado = df_completo_filtrado_cor_simbolo.sort_values("Ordem").reset_index(drop=True)
            
            # Gráfico
           
            fig = plot_hemiciclo_parlamentar(df_filtrado, ordem_partidos)
            st.pyplot(fig, transparent=True)

            abstencao = df_filtrado['Abstenção'].values[0] * 100  # Supondo que você tenha uma coluna de abstenção no DataFrame
            brancos = df_filtrado['Brancos'].values[0] 
            
            brancos_formatado = f"{brancos / 1000:.1f}K"
            subcol1, subcol2 = st.columns([0.6, 2])
            
            with subcol2:
                st.markdown(f"""

                <div style=" padding:6px 6px; border-radius:0 0 5px 5px; width:300px; text-align:center; ">
                    <span style="color:white; font-weight:bold; font-size:14px;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Abstenção: {abstencao:.2f}%</span>
                    <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> 
                    <span style="color:white; font-weight:bold; font-size:14px;">Brancos: {brancos_formatado}</span>
                </div>
                """, unsafe_allow_html=True)
    
    elif previsao_opcao == "🗣Partidos":
        
        dataset = pd.merge(df_completo_previsoes, simbolos, on=['Partido'], how='inner')
        dataset = dataset[dataset['Ano'].isin(['Histórico','Sondagens','Expetativas'])]
        dataset['Percentual']=dataset['Percentual']*100
        plot_metricas_com_simbolos(
            dataset,
            coluna_valor='Percentual',
            titulo_eixo_y='Percentual (%)',
            ylim_max=45
        )
# -----------------------------------------
# 2. RESULTADOS
# -----------------------------------------
with tabs[1]:
    
    df_resultados_distrito_ano_filtrado = df_resultados_partido_distrito_ano[(df_resultados_partido_distrito_ano['Distrito'] == distrito_selecionado)]
    resultado_opcao = st.radio("", ["📅2025","🏛️Parlamento","🗣 Partidos","🗓️2024-2025"], horizontal=True)
   
    if resultado_opcao == "📅2025":
        
        col1, col2 , col3 = st.columns([4,1,1])

        with col1:

            mostrar_titulo_custom(f"{distrito_selecionado}")
            df_resultados_distrito_ano_filtrado=df_resultados_distrito_ano_filtrado[df_resultados_distrito_ano_filtrado['Ano'].isin(['2025'])]     
            df_resultados_distrito_ano_filtrado['Percentual']=df_resultados_distrito_ano_filtrado['Percentual']*100
            #mostrar_abstencao_brancos(df_filtrado)
            info = f"Abstenção: {abstencao2025*100:.2f}%\nBrancos: {brancos2025/1000:.1f}K"
            fig = plot_bar_chart(df_resultados_distrito_ano_filtrado, coluna_valores="Percentual", formato="percentagem", legenda="2025", info_extra=info,n="2")

            st.pyplot(fig, transparent=True)

    elif resultado_opcao == "🏛️Parlamento":
        
        col1, col2 = st.columns([3, 1])
        # Ordenar para manter consistência de layout
        with col1:
            #st.markdown('<div class="custom-subheader">Assentos parlamentares</div>', unsafe_allow_html=True)
            ordem_partidos = ["BE","CDU","L","PS","PAN","JPP","AD", "IL", "CH"]
            df_completo_filtrado_cor_simbolo2025["Ordem"] = df_completo_filtrado_cor_simbolo2025["Partido"].apply(lambda x: ordem_partidos.index(x) if x in ordem_partidos else len(ordem_partidos))
            resultados_distrito_ano_filtrado = df_completo_filtrado_cor_simbolo2025.sort_values("Ordem").reset_index(drop=True)
            resultados_distrito_filtrado2025=resultados_distrito_ano_filtrado[resultados_distrito_ano_filtrado['Ano'].isin(['2025'])] 
            mostrar_titulo_custom(f"{distrito_selecionado}")
            fig = plot_hemiciclo_parlamentar(resultados_distrito_filtrado2025, ordem_partidos)
            st.pyplot(fig, transparent=True)
                      
          
            subcol1, subcol2 = st.columns([0.85, 2])

            with subcol2:
                st.markdown(f"""

                <div style=" padding:6px 6px; border-radius:0 0 5px 5px; width:300px; text-align:center; ">
                    <span style="color:white; font-weight:bold; font-size:14px;">Abstenção: {abstencao2025*100:.2f}%</span>
                    <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> 
                    <span style="color:white; font-weight:bold; font-size:14px;">Brancos: {brancos2025 / 1000:.1f}K</span>
                </div>
                """, unsafe_allow_html=True)

    elif resultado_opcao == "🗣 Partidos":
       
        #-------------------------
        #st.subheader("Evolução de Mandatos")
        # 👉 Coloca aqui o código da comparação de resultados (por ano ou partido)
        
        df_completo_filtrado_cor_simbolo20252024['Percentual'] = df_completo_filtrado_cor_simbolo20252024['Percentual'] * 100  # Converter para percentual
        
        plot_metricas_com_simbolos4(
            df_completo_filtrado_cor_simbolo20252024,
            coluna_valor='Percentual',
            titulo_eixo_y='Percentual (%)',
            ylim_max=45
        )
        
        
    elif resultado_opcao == "🗓️2024-2025":
        
        #st.subheader("Resultados 2025-2024")
        # 👉 Coloca aqui o código da visualização 2024-2025 (pie ou barras)
        col1, col2  = st.columns([4,2])
        with col1:
            mostrar_titulo_custom(f"{distrito_selecionado}")
            df_resultados_distrito_ano_filtrado=df_resultados_distrito_ano_filtrado[df_resultados_distrito_ano_filtrado['Ano'].isin(['2025','2024'])]     
            df_resultados_distrito_ano_filtrado['Percentual']=df_resultados_distrito_ano_filtrado['Percentual']*100

            df_resultados_distrito_ano_filtrado['Mandatos'] = df_resultados_distrito_ano_filtrado['Mandatos'].astype(int)  # Converter para inteiro
            dataset_filtrados2025=df_resultados_distrito_ano_filtrado[df_resultados_distrito_ano_filtrado['Ano'].isin(['2025'])]
            dataset_filtrados2024=df_resultados_distrito_ano_filtrado[df_resultados_distrito_ano_filtrado['Ano'].isin(['2024'])]
            

            brancos_formatado2025 = f"{brancos2025 / 1000:.1f}K"
            brancos_formatado2024 = f"{brancos2024 / 1000:.1f}K"
            legenda = (
                f"2025\nAbstenção: {abstencao2025*100:.2f}%\nBrancos: {brancos_formatado2025}\n\n"
                f"2024\nAbstenção: {abstencao2024*100:.2f}%\nBrancos: {brancos_formatado2024}"
            )
                        
            fig = plot_bar_chart_comparativo(
                df=df_resultados_distrito_ano_filtrado,
                #coluna_valores='Mandatos',         # ou 'Percentual'
                coluna_valores='Percentual',   
                formato='percentagem',                      # ou 'percentagem'
                legenda=legenda,
                info_extra="",
                n="2",
                ano_ordem=2025,
                ano_esbatido=2024
            )
            st.pyplot(fig, transparent=True)
            
with tabs[2]:           
    
    df_historico_partidos_cor_simbolo_filtrado['Percentual']=df_historico_partidos_cor_simbolo_filtrado['Percentual']*100
    plot_metricas_com_simbolos4(
        df_historico_partidos_cor_simbolo_filtrado,
        coluna_valor='Percentual',
        titulo_eixo_y='Percentual (%)',
        ylim_max=50
    )


    # Exemplo de uso:
    plot_hemiciclos_2x2(df_historico_partidos_cor_simbolo_filtrado, ordem_partidos= ["BE","CDU","L","PS","PAN","JPP","AD", "IL", "CH"])
