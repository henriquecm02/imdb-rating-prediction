"""Dados, modelos e consultas sobre os 999 filmes do IMDb (base do desafio Lighthouse/Indicium).

Tudo aqui é determinístico. O agente só chama estas funções.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "dfbruto.csv"
MODEL_PATH = ROOT / "models" / "nota_imdb.joblib"

# Filme de teste do desafio (retirado da base): The Shawshank Redemption, nota real 9,3
SHAWSHANK = {
    "Series_Title": "The Shawshank Redemption", "Released_Year": 1994, "Certificate": "A", "Runtime": 142,
    "Genre": "Drama", "Meta_score": 80.0, "Director": "Frank Darabont", "No_of_Votes": 2343110, "Gross": 28341469.0,
    "Overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
}
SHAWSHANK_REAL = 9.3

NUM = ["Released_Year", "Runtime", "Meta_score", "log_votes", "log_gross"]
TOP_GENRES = ["Drama", "Comedy", "Crime", "Adventure", "Action", "Thriller", "Romance", "Biography", "Mystery",
              "Animation", "Sci-Fi", "Fantasy", "History", "Family", "War", "Music", "Horror", "Western"]


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    df = pd.read_csv(CSV).drop(columns=["Unnamed: 0"], errors="ignore")
    df.loc[df["Series_Title"] == "Apollo 13", "Released_Year"] = "1995"  # a base traz "PG" no ano
    df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce").astype("Int64")
    df["Runtime"] = df["Runtime"].str.extract(r"(\d+)").astype(int)
    df["Gross"] = pd.to_numeric(df["Gross"].astype(str).str.replace(",", ""), errors="coerce")
    df["Certificate"] = df["Certificate"].fillna("Sem")
    df["generos"] = df["Genre"].str.split(", ")
    return df


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Released_Year"] = df["Released_Year"].astype(float)
    X["Runtime"] = df["Runtime"].astype(float)
    X["Meta_score"] = df["Meta_score"].astype(float)
    X["log_votes"] = np.log10(df["No_of_Votes"].astype(float))
    X["log_gross"] = np.log10(df["Gross"].astype(float) + 1)
    X["Certificate"] = df["Certificate"].fillna("Sem")
    gen = df["Genre"].str.split(", ")
    for g in TOP_GENRES:
        X[f"g_{g}"] = gen.apply(lambda lst, g=g: g in lst).astype(float)
    return X


def _pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cert", OneHotEncoder(handle_unknown="ignore", min_frequency=10), ["Certificate"]),
    ], remainder="passthrough")
    return Pipeline([("prep", FunctionTransformer(_prep)), ("cols", pre), ("model", estimator)])


MODELOS = {
    "linear": lambda: Ridge(alpha=1.0),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1),
    "gbm": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0),
}


@lru_cache(maxsize=1)
def compare() -> list[dict]:
    """RMSE e MAE em validação cruzada de 5 dobras, mais a linha de base (prever a média)."""
    df = load()
    y = df["IMDB_Rating"].to_numpy()
    cv = KFold(5, shuffle=True, random_state=42)
    rows = [{"modelo": "media", "rmse": round(float(np.sqrt(np.mean((y - y.mean()) ** 2))), 4),
             "mae": round(float(np.mean(np.abs(y - y.mean()))), 4)}]
    for name, make in MODELOS.items():
        p = cross_val_predict(_pipeline(make()), df, y, cv=cv)
        rows.append({"modelo": name, "rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 4),
                     "mae": round(float(np.mean(np.abs(y - p))), 4)})
    return rows


def best_name() -> str:
    return min((r for r in compare() if r["modelo"] != "media"), key=lambda r: r["rmse"])["modelo"]


@lru_cache(maxsize=1)
def model() -> Pipeline:
    """Pipeline completo (pré-processamento + modelo) salvo em models/, ao contrário do .pkl antigo."""
    if MODEL_PATH.exists():
        m = joblib.load(MODEL_PATH)
        if getattr(m, "_nome", None) == best_name():
            return m
    df = load()
    m = _pipeline(MODELOS[best_name()]()).fit(df, df["IMDB_Rating"])
    m._nome = best_name()
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(m, MODEL_PATH)
    return m


def predict(filme: dict) -> float:
    row = {**{"Certificate": "Sem", "Meta_score": np.nan, "Gross": np.nan}, **filme}
    for k in ("Released_Year", "Runtime", "No_of_Votes", "Genre"):
        if k not in row:
            raise ValueError(f"campo obrigatório ausente: {k}")
    return float(model().predict(pd.DataFrame([row]))[0])


@lru_cache(maxsize=1)
def importance() -> list[dict]:
    """Importância por permutação: quanto o RMSE piora ao embaralhar cada coluna original da base."""
    df = load()
    cols = ["Released_Year", "Runtime", "Certificate", "Genre", "Meta_score", "No_of_Votes", "Gross"]
    out = []
    base = np.sqrt(np.mean((df["IMDB_Rating"] - model().predict(df)) ** 2))
    rng = np.random.default_rng(0)
    for c in cols:
        deltas = []
        for _ in range(5):
            d2 = df.copy()
            d2[c] = rng.permutation(d2[c].to_numpy())
            deltas.append(np.sqrt(np.mean((df["IMDB_Rating"] - model().predict(d2)) ** 2)) - base)
        out.append({"variavel": c, "aumento_rmse": round(float(np.mean(deltas)), 4)})
    return sorted(out, key=lambda x: -x["aumento_rmse"])


# ----------------------------------------------------------------- Gênero pela sinopse
@lru_cache(maxsize=1)
def genre_classifier() -> tuple[Pipeline, MultiLabelBinarizer, dict]:
    df = load()
    mlb = MultiLabelBinarizer(classes=TOP_GENRES)
    Y = mlb.fit_transform(df["generos"].apply(lambda g: [x for x in g if x in TOP_GENRES]))
    clf = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")),
                    ("ovr", OneVsRestClassifier(LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")))])
    P = cross_val_predict(clf, df["Overview"], Y, cv=KFold(5, shuffle=True, random_state=42), method="predict_proba")
    top1 = P.argmax(axis=1)
    acerto_top1 = float(np.mean(Y[np.arange(len(Y)), top1] == 1))
    clf.fit(df["Overview"], Y)
    return clf, mlb, {"acerto_top1_cv": round(acerto_top1, 3), "n": len(df)}


def genre_from_overview(texto: str, k: int = 3) -> list[dict]:
    clf, mlb, _ = genre_classifier()
    p = clf.predict_proba([texto])[0]
    idx = np.argsort(-p)[:k]
    return [{"genero": mlb.classes_[i], "prob": round(float(p[i]), 3)} for i in idx]


# --------------------------------------------------------------------- Consultas
ORDENAVEIS = {"nota": "IMDB_Rating", "votos": "No_of_Votes", "faturamento": "Gross", "ano": "Released_Year",
              "metascore": "Meta_score", "duracao": "Runtime"}


def query(genero: str | None = None, diretor: str | None = None, ator: str | None = None, ano_min: int | None = None,
          ano_max: int | None = None, nota_min: float | None = None, duracao_max: int | None = None,
          ordenar_por: str = "nota", limite: int | None = 10) -> pd.DataFrame:
    """Filtro seguro: só os campos listados, sem eval de texto livre."""
    if ordenar_por not in ORDENAVEIS:
        raise ValueError(f"ordenar_por deve ser um de {list(ORDENAVEIS)}")
    df = load()
    m = pd.Series(True, index=df.index)
    if genero:
        m &= df["generos"].apply(lambda g: genero.lower() in [x.lower() for x in g])
    if diretor:
        m &= df["Director"].str.contains(diretor, case=False, regex=False)
    if ator:
        m &= df[["Star1", "Star2", "Star3", "Star4"]].apply(lambda r: r.str.contains(ator, case=False, regex=False)).any(axis=1)
    if ano_min:
        m &= df["Released_Year"] >= int(ano_min)
    if ano_max:
        m &= df["Released_Year"] <= int(ano_max)
    if nota_min:
        m &= df["IMDB_Rating"] >= float(nota_min)
    if duracao_max:
        m &= df["Runtime"] <= int(duracao_max)
    col = ORDENAVEIS[ordenar_por]
    out = df[m].sort_values([col, "No_of_Votes"], ascending=False)
    return out if limite is None else out.head(max(1, min(int(limite), 50)))
