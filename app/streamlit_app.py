"""Dashboard e agente de filmes do IMDb. Rodar: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import tema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filmeagent import core as C  # noqa: E402
from filmeagent.tools import build_agent, t_prever  # noqa: E402

st.set_page_config(page_title="Filmes IMDb · agente", page_icon="🎬", layout="wide")


@st.cache_resource
def get_agent():
    C.compare()
    C.model()
    return build_agent()


df = C.load()
agent = get_agent()
st.session_state.setdefault("pending", [])
st.session_state.setdefault("log", [])

tema.aplicar_tema(
    "Agente de filmes: notas do IMDb",
    "Consulta os 999 filmes da base, prevê a nota de um filme novo, explica o que pesa na nota e infere o gênero pela "
    "sinopse. Os números vêm do pandas e do scikit-learn; o modelo de linguagem só interpreta.",
    kicker="Agente de IA · cinema",
    chips=["Gemini", "Random forest", "TF-IDF", "999 filmes do IMDb"])
if agent.llm is None:
    st.info("Modo offline: sem GEMINI_API_KEY, o agente escolhe ferramentas por palavras-chave.", icon="🔌")

tab_dash, tab_pred, tab_agent, tab_eval, tab_aprov = st.tabs(["Dashboard", "Prever nota", "Agente", "Avaliação", "Aprovações"])

with tab_dash:
    k1, k2, k3, k4 = st.columns(4)
    best = C.best_name()
    tab = {r["modelo"]: r for r in C.compare()}
    k1.metric("Filmes na base", f"{len(df)}")
    k2.metric("Nota média", f"{df['IMDB_Rating'].mean():.2f}")
    k3.metric("RMSE do melhor modelo (CV)", f"{tab[best]['rmse']:.3f}", delta=f"{tab[best]['rmse'] - tab['media']['rmse']:+.3f} vs. média",
              delta_color="inverse")
    k4.metric("Shawshank previsto", f"{C.predict(C.SHAWSHANK):.2f}", help="Nota real: 9,3")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Nota × número de votos")
        fig = px.scatter(df, x="No_of_Votes", y="IMDB_Rating", log_x=True, hover_name="Series_Title", opacity=0.55,
                         color_discrete_sequence=[tema.cor(0)], labels={"No_of_Votes": "votos (escala log)", "IMDB_Rating": "nota"})
        fig.update_layout(height=360, margin={"l": 10, "r": 10, "t": 10, "b": 10})
        st.plotly_chart(tema.estilo_plotly(fig), use_container_width=True)
    with c2:
        st.subheader("Gêneros: quantidade e nota média")
        g = df.explode("generos").groupby("generos").agg(filmes=("Series_Title", "count"), nota=("IMDB_Rating", "mean"))
        g = g[g["filmes"] >= 15].sort_values("filmes")
        fig2 = px.bar(g.reset_index(), x="filmes", y="generos", orientation="h", color="nota", color_continuous_scale=tema.sequencial(),
                      labels={"generos": "", "filmes": "filmes", "nota": "nota média"})
        fig2.update_layout(height=360, margin={"l": 10, "r": 10, "t": 10, "b": 10})
        st.plotly_chart(tema.estilo_plotly(fig2), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Modelos (validação cruzada de 5 dobras)")
        st.dataframe(pd.DataFrame(C.compare()), hide_index=True, use_container_width=True)
    with c4:
        st.subheader("O que pesa na nota")
        imp = pd.DataFrame(C.importance())
        fig3 = px.bar(imp.sort_values("aumento_rmse"), x="aumento_rmse", y="variavel", orientation="h", color_discrete_sequence=[tema.cor(0)],
                      labels={"aumento_rmse": "aumento do RMSE ao embaralhar", "variavel": ""})
        fig3.update_layout(height=280, margin={"l": 10, "r": 10, "t": 10, "b": 10})
        st.plotly_chart(tema.estilo_plotly(fig3), use_container_width=True)

with tab_pred:
    st.subheader("Prever a nota de um filme novo")
    a, b, c = st.columns(3)
    titulo = a.text_input("Título", "Meu filme")
    ano = a.number_input("Ano", 1920, 2030, 2024)
    dur = b.number_input("Duração (min)", 60, 300, 130)
    gen = b.multiselect("Gêneros", C.TOP_GENRES, ["Drama"])
    meta = c.number_input("Meta Score (0 = sem nota)", 0, 100, 75)
    votos = c.number_input("Votos no IMDb", 25_000, 3_000_000, 300_000, step=25_000)
    if st.button("Prever", type="primary"):
        r = t_prever(titulo, ano, dur, ", ".join(gen) or "Drama", meta or None, votos)
        st.success(r["resumo"])

with tab_agent:
    st.subheader("Pergunte ao agente")
    ex = st.selectbox("Exemplos", ["", "Quais os filmes do Nolan?", "Qual a nota prevista para The Shawshank Redemption?",
                                   "O que mais pesa na nota de um filme?", "Me recomende um filme de comédia de até 100 min",
                                   "Quais fatores se relacionam com faturamento alto?"], placeholder="Escolha um exemplo")
    q = st.text_input("Pergunta", value=ex)
    if st.button("Perguntar", type="primary") and q.strip():
        r = agent.run(q)
        st.session_state["log"].append(r)
        st.session_state["pending"].extend(r.pending)
    for r in reversed(st.session_state["log"][-5:]):
        with st.container(border=True):
            st.markdown(f"**{r.question}**")
            st.write(r.answer)
            badge = {"ok": "✅ números conferidos", "revisar": "⚠️ revisar: número sem fonte", "erro": "❌ erro"}[r.status]
            st.caption(f"{badge} · modo {r.mode} · ferramentas: {', '.join(c['tool'] for c in r.calls) or 'nenhuma'}")
            with st.expander("Rastro"):
                st.json({"calls": r.calls, "results": r.results, "errors": r.errors}, expanded=False)

with tab_eval:
    p = ROOT / "eval" / "resultados.json"
    if p.exists():
        res = json.loads(p.read_text(encoding="utf-8"))
        st.metric("Casos corretos", f"{res['acertos']} de {res['total']}")
        st.dataframe(pd.DataFrame(res["casos"]).astype(str), hide_index=True, use_container_width=True)

with tab_aprov:
    if not st.session_state["pending"]:
        st.write("Nenhuma ação pendente. Peça ao agente para publicar um relatório.")
    for i, pa in enumerate(list(st.session_state["pending"])):
        with st.container(border=True):
            st.markdown(f"**{pa.tool}** · `{pa.args}`")
            x, y = st.columns(2)
            if x.button("Aprovar e executar", key=f"ok{i}", type="primary"):
                st.success(agent.approve(pa)["resumo"])
                st.session_state["pending"].remove(pa)
            if y.button("Recusar", key=f"no{i}"):
                st.session_state["pending"].remove(pa)
                st.rerun()

