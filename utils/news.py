import requests
import pandas as pd
import feedparser
from datetime import datetime, timedelta
from unidecode import unidecode
import re
from datetime import timezone

# --- CONFIGURAÇÕES ---

API_KEYS = {
    "gnews": "ccb923df5c5b16848a8eb7adc4c4b289",  # Substitui pela tua chave real
}

TERMOS_PESQUISA = [    
    # Português
    "chega","populismo", "demagogia", "desinformação", "extrema[- ]?direita", "ultradireita","extrema","extremismo",
    "xenofobia", "ódio", "fake news", "manipulação", "mentira", "propaganda",
    "nacionalismo", "anti[- ]?sistema", "corrupção", "imigrantes", "moção de censura", "imigração", "radical","ciganos",
    # Espanhol
    "ultraderecha", "desinformación", "odio", "corrupción", "manipulación", "mentiras",
    "sistema", "populista", "xenofobia", "inmigrantes", "extrema derecha"
]

DIAS = 365
IDIOMAS = ["pt", "es"]

# --- FUNÇÕES ---

def normalizar_texto(texto):
    return unidecode(str(texto).lower())

def buscar_gnews(query, dias=DIAS, idioma="pt", max_artigos=100):
    base_url = "https://gnews.io/api/v4/search"
    data_final = datetime.utcnow()
    data_inicio = data_final - timedelta(days=dias)

    params = {
        "q": query,
        "lang": idioma,
        "from": data_inicio.strftime("%Y-%m-%d"),
        "to": data_final.strftime("%Y-%m-%d"),
        "token": API_KEYS["gnews"],
        "max": max_artigos,
        "sortby": "relevance"
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"GNews falhou ({response.status_code}): {response.text}")
        return pd.DataFrame()
    artigos = response.json().get("articles", [])
    dados = []
    for a in artigos:
        dados.append({
            "Fonte": "GNews",
            "Idioma": idioma,
            "Título": a.get("title", ""),
            "Descrição": a.get("description", ""),
            "Data": a.get("publishedAt", ""),
            "URL": a.get("url", "")
        })
    return pd.DataFrame(dados)

RSS_FEEDS = [
    "https://observador.pt/feed/",
    "https://elpais.com/rss/elpais/portada.xml",
    "https://www.publico.pt/rss",
]

def detectar_idioma(texto):
    pt_palavras = [" e ", " de ", " que ", " o ", " a ", " para ", " por ", " com "]
    es_palavras = [" y ", " de ", " que ", " el ", " la ", " para ", " por ", " con "]
    texto_norm = f" {normalizar_texto(texto)} "
    pt_count = sum(texto_norm.count(p) for p in pt_palavras)
    es_count = sum(texto_norm.count(p) for p in es_palavras)
    return "pt" if pt_count >= es_count else "es"

def buscar_rss_feeds(feeds=RSS_FEEDS):
    dados = []
    for feed_url in feeds:
        d = feedparser.parse(feed_url)
        for entry in d.entries:
            dados.append({
                "Fonte": "RSS",
                "Idioma": detectar_idioma(entry.title),
                "Título": entry.title,
                "Descrição": entry.get("summary", ""),
                "Data": entry.get("published", ""),
                "URL": entry.link
            })
    return pd.DataFrame(dados)

def filtrar_por_termos(df, termos=TERMOS_PESQUISA):
    pattern = re.compile("|".join(termos), re.IGNORECASE)
    filtro = df.apply(lambda row: bool(pattern.search(row["Título"])) or bool(pattern.search(row["Descrição"])), axis=1)
    return df[filtro]

def classificar_populista(texto, termos):
    texto_norm = unidecode(texto.lower())
    for termo in termos:
        if re.search(termo, texto_norm, re.IGNORECASE):
            return True
    return False

def calcular_indice_populismo(df):
    if len(df) == 0:
        return 0
    populistas = df["Populista"].sum()
    return round((populistas / len(df)) * 100, 2)

# --- EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    query = " OR ".join(TERMOS_PESQUISA)
    dfs = []
    for idioma in IDIOMAS:
        print(f"Buscando termos em {idioma} - GNews")
        df_gnews = buscar_gnews(query, idioma=idioma)
        dfs.append(df_gnews)

    print("Buscando RSS feeds")
    df_rss = buscar_rss_feeds()
    dfs.append(df_rss)

    noticias = pd.concat(dfs, ignore_index=True)

    # Normalizar datas e timezone
    data_limite = datetime.utcnow() - timedelta(days=DIAS)
    noticias["Data"] = pd.to_datetime(noticias["Data"], errors="coerce").dt.tz_localize(None)
    noticias = noticias[noticias["Data"] >= data_limite]

    # Remover duplicados por título
    noticias = noticias.drop_duplicates(subset=["Título"])

    # Filtrar por termos (opcional)
    noticias = filtrar_por_termos(noticias)

    # Adicionar coluna Populista
    noticias["Populista"] = noticias.apply(
        lambda row: classificar_populista(row["Título"] + " " + row["Descrição"], TERMOS_PESQUISA),
        axis=1
    )

    # Calcular índice populismo
    indice = calcular_indice_populismo(noticias)
    print(f"\nÍndice de populismo (últimos {DIAS} dias): {indice}%")

    # Exportar para CSV
    nome_csv = "noticias_aggregadas_pt_es.csv"
    noticias.to_csv(nome_csv, sep=";", encoding="utf-8-sig", index=False)

    print(f"✅ Exportado {len(noticias)} notícias para {nome_csv}")
