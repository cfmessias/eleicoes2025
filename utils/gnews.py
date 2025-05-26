import requests
import pandas as pd
import re
from datetime import datetime, timedelta
from unidecode import unidecode

API_KEY = "ccb923df5c5b16848a8eb7adc4c4b289"  # Tua chave GNews API

def obter_noticias_melhorado(termo, dias=365, linguas=["pt", "es"], max_artigos=1000):
    base_url = "https://gnews.io/api/v4/search"
    data_final = datetime.today()
    data_inicio = data_final - timedelta(days=dias)

    noticias = []

    for lingua in linguas:
        params = {
            "q": termo,
            "lang": lingua,
            "from": data_inicio.strftime("%Y-%m-%d"),
            "to": data_final.strftime("%Y-%m-%d"),
            "token": API_KEY,
            "max": max_artigos,
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            dados = response.json().get("articles", [])
            for artigo in dados:
                noticias.append({
                    "Idioma": lingua,
                    "Título": artigo.get("title", ""),
                    "Descrição": artigo.get("description", ""),
                    "Fonte": artigo.get("source", {}).get("name", ""),
                    "Data": artigo.get("publishedAt", ""),
                    "URL": artigo.get("url", "")
                })
        else:
            print(f"Erro ao buscar notícias em {lingua}: {response.status_code}")

    return pd.DataFrame(noticias)

def calcular_indice_populismo_avancado(df):
    palavras_chave = [
        # Português
        "populismo", "demagogia", "desinformação", "extrema[- ]?direita", "ultradireita",
        "xenofobia", "ódio", "fake news", "manipulação", "mentira", "propaganda",
        "nacionalismo", "anti[- ]?sistema", "corrupção", "imigrantes", "moção de censura","imigração","radical",
        # Espanhol
        "ultraderecha", "desinformación", "odio", "corrupción", "manipulación", "mentiras",
        "sistema", "populista", "xenofobia", "inmigrantes", "extrema derecha"
    ]

    padrao = re.compile(r"|".join(palavras_chave), re.IGNORECASE)

    def verificar_populismo(texto):
        texto_normalizado = unidecode(str(texto).lower())
        return bool(padrao.search(texto_normalizado))

    df["Populista"] = df.apply(lambda row: verificar_populismo(f"{row['Título']} {row['Descrição']}"), axis=1)
    indice = df["Populista"].mean() * 100
    return df, round(indice, 2)

if __name__ == "__main__":
    termo = "Chega"
    noticias_df = obter_noticias_melhorado(termo, dias=90)
    noticias_pop_df, indice_populismo = calcular_indice_populismo_avancado(noticias_df)

    ficheiro_saida = f"noticias_{termo}_pt_es_populismo.csv"
    noticias_pop_df.to_csv(ficheiro_saida, index=False, encoding="utf-8-sig", sep=";")

    print(f"\n✅ Exportado: {ficheiro_saida}")
    print(f"📊 Índice de Populismo: {indice_populismo:.2f}% em {len(noticias_pop_df)} notícias\n")
    print(noticias_pop_df[noticias_pop_df["Populista"] == True][["Idioma", "Título", "Fonte", "URL"]].head())
