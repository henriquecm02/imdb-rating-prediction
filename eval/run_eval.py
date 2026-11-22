"""Avaliação contra gabarito. Roda sem GEMINI_API_KEY.

As respostas esperadas são calculadas aqui direto da base bruta (pandas), de forma independente
das ferramentas, e comparadas com o que o agente devolve.
Uso: python eval/run_eval.py [--gemini]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filmeagent import core as C  # noqa: E402
from filmeagent.agentcore import ScriptedLLM  # noqa: E402
from filmeagent.tools import build_agent  # noqa: E402

ROTEAMENTO = [
    ("Quais os filmes do Nolan?", "consultar_filmes"),
    ("Liste os melhores filmes de animação", "consultar_filmes"),
    ("Qual a nota média dos filmes de crime?", "estatisticas"),
    ("Qual a nota prevista para The Shawshank Redemption?", "prever_shawshank"),
    ("Compare os modelos pelo RMSE", "comparar_modelos"),
    ("O que mais pesa na nota de um filme?", "explicar_nota"),
    ("Qual o gênero desta sinopse: A young wizard attends a school of magic and fights a dark lord.", "genero_pela_sinopse"),
    ("Me recomende um filme de comédia de até 100 min", "recomendar"),
    ("Quais fatores se relacionam com faturamento alto?", "fatores_de_faturamento"),
    ("Publique um relatório com os achados", "publicar_relatorio"),
]


def check(nome, ok, obtido, esperado):
    return {"caso": nome, "ok": bool(ok), "obtido": obtido, "esperado": esperado}


def main() -> int:
    raw = pd.read_csv(ROOT / "data" / "dfbruto.csv")
    raw_gross = pd.to_numeric(raw["Gross"].astype(str).str.replace(",", ""), errors="coerce")
    raw_runtime = raw["Runtime"].str.extract(r"(\d+)")[0].astype(int)
    agent = build_agent(use_env=False)
    out = []

    def tool(name, **kw):
        return agent.tools[name].run(**kw)

    nolan = int(raw["Director"].str.contains("Nolan").sum())
    out.append(check("quantidade de filmes do Nolan", tool("consultar_filmes", diretor="Nolan")["total_encontrado"] == nolan,
                     tool("consultar_filmes", diretor="Nolan")["total_encontrado"], nolan))
    top_gross = raw.loc[raw_gross.idxmax(), "Series_Title"]
    got = tool("consultar_filmes", ordenar_por="faturamento", limite=1)["filmes"][0]["titulo"]
    out.append(check("filme de maior faturamento", got == top_gross, got, top_gross))
    med = float(raw_runtime.median())
    out.append(check("duração mediana da base", tool("estatisticas")["duracao_mediana"] == med, tool("estatisticas")["duracao_mediana"], med))
    drama = round(float(raw.loc[raw["Genre"].str.contains("Drama"), "IMDB_Rating"].mean()), 2)
    out.append(check("nota média dos dramas", tool("estatisticas", genero="Drama")["nota_media"] == drama,
                     tool("estatisticas", genero="Drama")["nota_media"], drama))
    out.append(check("ano de Apollo 13 corrigido (base traz 'PG')",
                     int(C.load().loc[C.load()["Series_Title"] == "Apollo 13", "Released_Year"].iloc[0]) == 1995, 1995, 1995))
    tab = {r["modelo"]: r["rmse"] for r in C.compare()}
    best = C.best_name()
    out.append(check("melhor modelo com RMSE de CV <= 0,20", tab[best] <= 0.20, {best: tab[best]}, "<= 0.20"))
    out.append(check("melhor modelo supera linear e média", tab[best] < tab["linear"] < tab["media"], tab, "melhor < linear < media"))
    sh = C.predict(C.SHAWSHANK)
    out.append(check("Shawshank previsto entre 8,5 e 9,5", 8.5 <= sh <= 9.5, round(sh, 3), "[8.5, 9.5]"))
    acc = C.genre_classifier()[2]["acerto_top1_cv"]
    out.append(check("gênero principal pela sinopse (acerto top-1 em CV >= 0,60)", acc >= 0.60, acc, ">= 0.60"))

    usar_gemini = "--gemini" in sys.argv
    ag = build_agent(use_env=usar_gemini)
    if usar_gemini and ag.llm is None:
        print("--gemini pedido, mas GEMINI_API_KEY não está definida")
        return 2
    for pergunta, esperado in ROTEAMENTO:
        r = ag.run(pergunta)
        tools = [x["tool"] for x in r.calls]
        ok = tools[:1] == [esperado]
        ok = ok and (bool(r.pending) and not r.results if esperado == "publicar_relatorio" else r.status == "ok")
        out.append(check(f"roteamento: {pergunta}", ok, tools, esperado))

    llm = ScriptedLLM([{"calls": [{"tool": "consultar_filmes", "args_json": "{\"diretor\": \"Nolan\"}"}]},
                       {"answer": "Nolan tem 14 filmes na base.", "numbers_cited": [14]}])
    r = build_agent(llm=llm, use_env=False).run("Quantos filmes do Nolan?")
    out.append(check("número inventado é barrado", r.status == "revisar", r.status, "revisar"))
    llm = ScriptedLLM([{"calls": [{"tool": "consultar_filmes", "args_json": "{\"diretor\": \"Nolan\"}"}]},
                       {"answer": f"Nolan tem {nolan} filmes na base.", "numbers_cited": [nolan]}])
    r = build_agent(llm=llm, use_env=False).run("Quantos filmes do Nolan?")
    out.append(check("número correto passa", r.status == "ok", r.status, "ok"))

    acertos = sum(x["ok"] for x in out)
    suf = "_gemini" if usar_gemini else ""
    res = {"acertos": acertos, "total": len(out), "modelos": C.compare(), "shawshank": round(sh, 3), "genero_top1": acc,
           "importancia": C.importance(), "casos": out}
    (ROOT / "eval" / f"resultados{suf}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = [f"# Resultados da avaliação\n\n**{acertos} de {len(out)} casos corretos.**\n", "| Caso | Resultado | Obtido | Esperado |",
          "|---|---|---|---|"] + [f"| {x['caso']} | {'ok' if x['ok'] else 'FALHOU'} | {x['obtido']} | {x['esperado']} |" for x in out]
    md += ["\n## Modelos (validação cruzada de 5 dobras)\n", "| Modelo | RMSE | MAE |", "|---|---|---|"]
    md += [f"| {r['modelo']} | {r['rmse']:.3f} | {r['mae']:.3f} |" for r in C.compare()]
    (ROOT / "eval" / f"resultados{suf}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{acertos}/{len(out)} casos corretos")
    for x in out:
        if not x["ok"]:
            print("FALHOU:", x)
    return 0 if acertos == len(out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
