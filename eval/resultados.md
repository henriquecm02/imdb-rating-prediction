# Resultados da avaliação

**21 de 21 casos corretos.**

| Caso | Resultado | Obtido | Esperado |
|---|---|---|---|
| quantidade de filmes do Nolan | ok | 8 | 8 |
| filme de maior faturamento | ok | Star Wars: Episode VII - The Force Awakens | Star Wars: Episode VII - The Force Awakens |
| duração mediana da base | ok | 119.0 | 119.0 |
| nota média dos dramas | ok | 7.96 | 7.96 |
| ano de Apollo 13 corrigido (base traz 'PG') | ok | 1995 | 1995 |
| melhor modelo com RMSE de CV <= 0,20 | ok | {'random_forest': 0.1864} | <= 0.20 |
| melhor modelo supera linear e média | ok | {'media': 0.2722, 'linear': 0.2289, 'random_forest': 0.1864, 'gbm': 0.1908} | melhor < linear < media |
| Shawshank previsto entre 8,5 e 9,5 | ok | 8.779 | [8.5, 9.5] |
| gênero principal pela sinopse (acerto top-1 em CV >= 0,60) | ok | 0.673 | >= 0.60 |
| roteamento: Quais os filmes do Nolan? | ok | ['consultar_filmes'] | consultar_filmes |
| roteamento: Liste os melhores filmes de animação | ok | ['consultar_filmes'] | consultar_filmes |
| roteamento: Qual a nota média dos filmes de crime? | ok | ['estatisticas'] | estatisticas |
| roteamento: Qual a nota prevista para The Shawshank Redemption? | ok | ['prever_shawshank'] | prever_shawshank |
| roteamento: Compare os modelos pelo RMSE | ok | ['comparar_modelos'] | comparar_modelos |
| roteamento: O que mais pesa na nota de um filme? | ok | ['explicar_nota'] | explicar_nota |
| roteamento: Qual o gênero desta sinopse: A young wizard attends a school of magic and fights a dark lord. | ok | ['genero_pela_sinopse'] | genero_pela_sinopse |
| roteamento: Me recomende um filme de comédia de até 100 min | ok | ['recomendar'] | recomendar |
| roteamento: Quais fatores se relacionam com faturamento alto? | ok | ['fatores_de_faturamento'] | fatores_de_faturamento |
| roteamento: Publique um relatório com os achados | ok | ['publicar_relatorio'] | publicar_relatorio |
| número inventado é barrado | ok | revisar | revisar |
| número correto passa | ok | ok | ok |

## Modelos (validação cruzada de 5 dobras)

| Modelo | RMSE | MAE |
|---|---|---|
| media | 0.272 | 0.218 |
| linear | 0.229 | 0.181 |
| random_forest | 0.186 | 0.147 |
| gbm | 0.191 | 0.148 |
