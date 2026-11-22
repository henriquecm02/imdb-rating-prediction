"""Ferramentas do agente de filmes. Cada uma devolve um dict com 'resumo'."""
from __future__ import annotations

import re
import unicodedata

import numpy as np

from . import core as C
from .agentcore import Agent, Tool, make_llm

REPORTS = C.ROOT / "reports"


def _films(df) -> list[dict]:
    return [{"titulo": r.Series_Title, "ano": int(r.Released_Year), "nota": float(r.IMDB_Rating), "genero": r.Genre,
             "diretor": r.Director, "votos": int(r.No_of_Votes),
             "faturamento": None if np.isnan(r.Gross) else float(r.Gross), "duracao": int(r.Runtime)} for r in df.itertuples()]


def t_consultar(genero: str | None = None, diretor: str | None = None, ator: str | None = None, ano_min: int | None = None,
                ano_max: int | None = None, nota_min: float | None = None, duracao_max: int | None = None,
                ordenar_por: str = "nota", limite: int = 10) -> dict:
    df = C.query(genero, diretor, ator, ano_min, ano_max, nota_min, duracao_max, ordenar_por, limite)
    total = len(C.query(genero, diretor, ator, ano_min, ano_max, nota_min, duracao_max, ordenar_por, None))
    filmes = _films(df)
    lista = "; ".join(f"{f['titulo']} ({f['ano']}, nota {f['nota']:.1f})" for f in filmes[:5])
    return {"total_encontrado": total, "filmes": filmes,
            "resumo": f"{total} filmes atendem ao filtro. Os primeiros por {ordenar_por}: {lista}." if total else "Nenhum filme atende ao filtro."}


def t_estatisticas(genero: str | None = None, diretor: str | None = None) -> dict:
    df = C.query(genero=genero, diretor=diretor, limite=None)
    if df.empty:
        return {"n": 0, "resumo": "Nenhum filme para esse recorte."}
    s = {"n": len(df), "nota_media": round(float(df["IMDB_Rating"].mean()), 2),
         "duracao_mediana": float(df["Runtime"].median()), "votos_medianos": float(df["No_of_Votes"].median()),
         "faturamento_mediano": round(float(df["Gross"].median()), 0) if df["Gross"].notna().any() else None,
         "metascore_medio": round(float(df["Meta_score"].mean()), 1)}
    rec = " e ".join(x for x in (genero, diretor) if x) or "toda a base"
    fat = f"; faturamento mediano de US$ {s['faturamento_mediano']:,.0f}".replace(",", ".") if s["faturamento_mediano"] else ""
    return {**s, "resumo": (f"Em {rec}: {s['n']} filmes, nota média {s['nota_media']:.2f}, duração mediana de "
                            f"{s['duracao_mediana']:.0f} min, Meta Score médio {s['metascore_medio']:.1f}{fat}.")}


def t_prever(titulo: str = "filme", ano: int = 2020, duracao: int = 120, genero: str = "Drama", metascore: float | None = None,
             votos: int = 100000, faturamento: float | None = None, certificado: str = "Sem") -> dict:
    f = {"Series_Title": titulo, "Released_Year": int(ano), "Runtime": int(duracao), "Genre": genero, "Certificate": certificado,
         "Meta_score": np.nan if metascore is None else float(metascore), "No_of_Votes": int(votos),
         "Gross": np.nan if faturamento is None else float(faturamento)}
    p = C.predict(f)
    rmse = next(r["rmse"] for r in C.compare() if r["modelo"] == C.best_name())
    return {"nota_prevista": round(p, 2), "modelo": C.best_name(), "rmse_cv": rmse,
            "resumo": f"Nota prevista para {titulo}: {p:.2f} (modelo {C.best_name()}, erro típico de {rmse:.2f} ponto em validação cruzada)."}


def t_shawshank() -> dict:
    p = C.predict(C.SHAWSHANK)
    return {"nota_prevista": round(p, 2), "nota_real": C.SHAWSHANK_REAL, "erro": round(C.SHAWSHANK_REAL - p, 2),
            "resumo": (f"Para The Shawshank Redemption, o modelo prevê {p:.2f}; a nota real é {C.SHAWSHANK_REAL:.1f}. "
                       f"O erro de {C.SHAWSHANK_REAL - p:.2f} é esperado: nenhum filme da base passa de 9,2, e o modelo não extrapola.")}


def t_comparar() -> dict:
    rows = C.compare()
    txt = "; ".join(f"{r['modelo']}: RMSE {r['rmse']:.3f}" for r in rows)
    return {"tabela": rows, "melhor": C.best_name(),
            "resumo": f"Validação cruzada de 5 dobras. {txt}. Melhor: {C.best_name()}."}


def t_explicar() -> dict:
    imp = C.importance()
    top = ", ".join(f"{i['variavel']} (+{i['aumento_rmse']:.3f})" for i in imp[:4])
    return {"importancia": imp, "resumo": f"Variáveis que mais pesam na nota, pelo aumento do RMSE ao embaralhar cada uma: {top}."}


def t_genero(sinopse: str) -> dict:
    g = C.genre_from_overview(sinopse)
    acc = C.genre_classifier()[2]["acerto_top1_cv"]
    lista = ", ".join(f"{x['genero']} ({x['prob']:.3f})" for x in g)
    return {"generos": g, "acerto_top1_cv": acc,
            "resumo": f"Gêneros mais prováveis pela sinopse: {lista}. O classificador acerta o gênero principal em {acc * 100:.1f}% dos filmes."}


def t_recomendar(genero: str | None = None, duracao_max: int | None = None, ano_min: int | None = None) -> dict:
    df = C.query(genero=genero, duracao_max=duracao_max, ano_min=ano_min, limite=None)
    if df.empty:
        return {"filmes": [], "resumo": "Nenhum filme atende ao perfil."}
    # nota bayesiana: puxa para a média filmes com poucos votos
    m, c = 25_000, C.load()["IMDB_Rating"].mean()
    v = df["No_of_Votes"]
    df = df.assign(score=(v / (v + m)) * df["IMDB_Rating"] + (m / (v + m)) * c).sort_values("score", ascending=False).head(5)
    filmes = _films(df)
    lista = "; ".join(f"{f['titulo']} ({f['ano']}, nota {f['nota']:.1f})" for f in filmes)
    return {"filmes": filmes, "resumo": f"Recomendações pela nota ponderada por votos: {lista}."}


def t_faturamento() -> dict:
    df = C.load().dropna(subset=["Gross"])
    corr = {c: round(float(np.corrcoef(np.log10(df["Gross"] + 1), df[c].astype(float).fillna(df[c].median()))[0, 1]), 3)
            for c in ("No_of_Votes", "Meta_score", "IMDB_Rating", "Runtime")}
    gen = df.explode("generos").groupby("generos")["Gross"].median().sort_values(ascending=False)
    gen = gen[df.explode("generos")["generos"].value_counts().reindex(gen.index) >= 15].head(5)
    top = "; ".join(f"{g}: US$ {v / 1e6:.1f} mi" for g, v in gen.items())
    return {"correlacao_log_faturamento": corr, "faturamento_mediano_por_genero": {g: float(v) for g, v in gen.items()},
            "resumo": (f"Correlação do log do faturamento com votos: {corr['No_of_Votes']:.3f}; com Meta Score: {corr['Meta_score']:.3f}; "
                       f"com a nota: {corr['IMDB_Rating']:.3f}. Gêneros com maior faturamento mediano: {top}.")}


def t_publicar(titulo: str = "Relatório de filmes", conteudo: str = "") -> dict:
    """Ação sensível: grava um relatório para ser compartilhado com o time."""
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "relatorio.md"
    path.write_text(f"# {titulo}\n\n{conteudo}\n", encoding="utf-8")
    return {"arquivo": "reports/relatorio.md", "resumo": f"Relatório '{titulo}' publicado em reports/relatorio.md."}


TOOLS = [
    Tool("consultar_filmes", "Lista filmes por filtros e ordenação.",
         {"genero": "texto", "diretor": "texto", "ator": "texto", "ano_min": "int", "ano_max": "int", "nota_min": "número",
          "duracao_max": "minutos", "ordenar_por": "nota|votos|faturamento|ano|metascore|duracao", "limite": "int"}, t_consultar),
    Tool("estatisticas", "Médias e medianas de um gênero, de um diretor ou da base toda.", {"genero": "texto", "diretor": "texto"}, t_estatisticas),
    Tool("prever_nota", "Prevê a nota IMDb de um filme novo.",
         {"titulo": "texto", "ano": "int", "duracao": "minutos", "genero": "texto, ex. 'Crime, Drama'", "metascore": "número",
          "votos": "int", "faturamento": "US$", "certificado": "texto"}, t_prever),
    Tool("prever_shawshank", "Prevê a nota do filme de teste do desafio (The Shawshank Redemption).", {}, t_shawshank),
    Tool("comparar_modelos", "RMSE em validação cruzada de média, linear, random forest e GBM.", {}, t_comparar),
    Tool("explicar_nota", "Quais variáveis mais pesam na nota prevista.", {}, t_explicar),
    Tool("genero_pela_sinopse", "Infere o gênero a partir da sinopse em inglês.", {"sinopse": "texto"}, t_genero),
    Tool("recomendar", "Recomenda filmes para um perfil.", {"genero": "texto", "duracao_max": "minutos", "ano_min": "int"}, t_recomendar),
    Tool("fatores_de_faturamento", "O que se relaciona com faturamento alto.", {}, t_faturamento),
    Tool("publicar_relatorio", "Publica um relatório para o time.", {"titulo": "texto", "conteudo": "texto"}, t_publicar, sensitive=True),
]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()


GEN_PT = {"drama": "Drama", "comedia": "Comedy", "crime": "Crime", "aventura": "Adventure", "acao": "Action", "suspense": "Thriller",
          "romance": "Romance", "biografia": "Biography", "misterio": "Mystery", "animacao": "Animation", "ficcao": "Sci-Fi",
          "fantasia": "Fantasy", "historia": "History", "familia": "Family", "guerra": "War", "terror": "Horror", "faroeste": "Western"}


def offline_router(question: str) -> list[dict]:
    q = _norm(question)
    genero = next((v for k, v in GEN_PT.items() if k in q), None)
    diretor = None
    for nome in ("nolan", "spielberg", "scorsese", "kubrick", "tarantino", "hitchcock", "fincher", "miyazaki", "kurosawa"):
        if nome in q:
            diretor = nome
    if "publi" in q:
        return [{"tool": "publicar_relatorio", "args": {"titulo": "Relatório", "conteudo": question}}]
    if "shawshank" in q or "um sonho de liberdade" in q:
        return [{"tool": "prever_shawshank", "args": {}}]
    if "sinopse" in q:
        texto = question.split(":", 1)[-1].strip() if ":" in question else question
        return [{"tool": "genero_pela_sinopse", "args": {"sinopse": texto}}]
    if "fatur" in q or "bilheteria" in q:
        return [{"tool": "fatores_de_faturamento", "args": {}}]
    if any(w in q for w in ("compar", "rmse", "melhor modelo")):
        return [{"tool": "comparar_modelos", "args": {}}]
    if any(w in q for w in ("pesa", "importan", "explic", "influencia")):
        return [{"tool": "explicar_nota", "args": {}}]
    if "recomend" in q or "indica" in q:
        dur = re.search(r"(\d+)\s*min", q)
        return [{"tool": "recomendar", "args": {"genero": genero, "duracao_max": int(dur.group(1)) if dur else None}}]
    if any(w in q for w in ("media", "mediana", "estatistic")):
        return [{"tool": "estatisticas", "args": {"genero": genero, "diretor": diretor}}]
    if "prev" in q and "nota" in q:
        return [{"tool": "prever_nota", "args": {"genero": genero or "Drama"}}]
    if genero or diretor or any(w in q for w in ("filmes", "quais", "lista", "melhores")):
        ordem = "faturamento" if "fatur" in q else "votos" if "votad" in q else "nota"
        return [{"tool": "consultar_filmes", "args": {"genero": genero, "diretor": diretor, "ordenar_por": ordem, "limite": 10}}]
    return []


def build_agent(llm=None, use_env: bool = True) -> Agent:
    if llm is None and use_env:
        llm = make_llm()
    return Agent(TOOLS, llm=llm, offline_router=offline_router)
