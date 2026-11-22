"""Testes sem rede."""
import pytest

from filmeagent import core as C
from filmeagent.agentcore import ScriptedLLM
from filmeagent.tools import build_agent, t_estatisticas


def test_cleaning():
    df = C.load()
    assert df["Released_Year"].notna().all()
    assert df["Runtime"].dtype.kind == "i"
    assert df["Gross"].dtype.kind == "f"


def test_stats_use_whole_base():
    assert t_estatisticas()["n"] == len(C.load())


def test_query_rejects_unknown_order():
    with pytest.raises(ValueError):
        C.query(ordenar_por="DROP TABLE")


def test_prediction_requires_fields():
    with pytest.raises(ValueError):
        C.predict({"Genre": "Drama"})


def test_shawshank_in_range():
    assert 8.5 <= C.predict(C.SHAWSHANK) <= 9.5


def test_best_model_beats_baseline():
    tab = {r["modelo"]: r["rmse"] for r in C.compare()}
    assert tab[C.best_name()] < tab["media"]


def test_publish_needs_approval():
    r = build_agent(use_env=False).run("Publique um relatório com os achados")
    assert r.pending and not r.results


def test_grounding():
    llm = ScriptedLLM([{"calls": [{"tool": "estatisticas", "args_json": "{}"}]},
                       {"answer": "A nota média é 9.9.", "numbers_cited": [9.9]}])
    assert build_agent(llm=llm, use_env=False).run("x").status == "revisar"
