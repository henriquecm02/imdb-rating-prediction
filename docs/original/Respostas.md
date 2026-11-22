###Respondendo Questões Primordiais:
### a. Qual filme você recomendaria para uma pessoa que você não conhece?
### Para recomendar um filme a um desconhecido, do qual não se tem as informações necessárias para uma escolha personalizada, o ideal é indicar o filme de melhor popularidade entre os assistiram. Neste caso o filme seria "The Shawshank Redemption" (1994), que segundo seu overview se trata de um drama carcerário com toques de esperança e amizade.

###b. Quais são os principais fatores que estão relacionados com alta expectativa de faturamento de um filme?
### Segundo a EDA realizada, os principais fatores que afetam o faturamento são, o gênero, a composição do elenco, o marketing realizado no filme e a percepção das críticas em geral, segue a explicação detalhada:
###Gênero: Filmes de Ação, Aventura, Ficção-científica e Fantasia costumam ter maior apelo comercial.
###Elenco: A presença de atores famosos e populares bem como diretores consolidados pode atrair um público maior.
###Orçamento: Filmes com maior orçamento podem investir em efeitos especiais, marketing e distribuição, aumentando a visibilidade e o interesse do público.
###Marketing: Campanhas publicitárias eficientes e bem direcionadas podem gerar grande expectativa e aumentar o número de espectadores.
###Recepção da crítica: Críticas positivas e premiações podem influenciar a decisão do público em assistir ao filme.

###Quais insights podem ser tirados com a coluna Overview? É possível inferir o gênero do filme a partir dessa coluna?
###A coluna Overview fornece uma breve sinopse do enredo do filme, permitindo identificar temas, personagens e conflitos principais. É possível inferir o gênero do filme a partir da Overview, mas nem sempre com precisão.
### Por exemplo, a sinopse de "The Godfather" (1972) menciona "crime" e "dinastia", indicando um filme de Gângster. Já a sinopse de "Spirited Away" (2001) descreve uma "menina" que "entra em um mundo governado por deuses", sugerindo um filme de Fantasia.
### Entretanto, algumas sinopses podem ser ambíguas. A sinopse de "Inception" (2010) menciona "roubo de segredos corporativos" e "tecnologia de compartilhamento de sonhos", dificultando a definição precisa do gênero, que mescla Ação, Sci-Fi e Thriller.
### Em suma, a coluna Overview pode fornecer pistas sobre o gênero do filme, mas a análise de outras variáveis, como as palavras-chave na sinopse, o elenco, o diretor e o ano de lançamento, contribuem para uma classificação mais precisa. Para detectar palavras chaves pode se usar o código abaixo:
```python
import pandas as pd

imdb_data = pd.read_csv('imdb_data.csv')

# Exemplo de como extrair palavras-chave da coluna Overview
imdb_data['Keywords'] = imdb_data['Overview'].str.lower().str.split()

# Exibir as palavras-chave para alguns filmes
print(imdb_data[['Overview', 'Keywords']].head())
```


## Prevendo Resultados com Machine Learning
### 3. Explique como você faria a previsão da nota do imdb a partir dos dados. Quais variáveis e/ou suas transformações você utilizou e por quê? Qual tipo de problema estamos resolvendo (regressão, classificação)? Qual modelo melhor se aproxima dos dados e quais seus prós e contras? Qual medida de performance do modelo foi escolhida e por quê?

A previsão da nota do IMDb pode ser realizada com técnicas de aprendizado de máquina, especificamente por meio de um modelo de regressão.

Para essa tarefa, utilizaremos as seguintes variáveis e transformações:

Variáveis Categóricas:

Certificate, Genre, Director, Star1, Star2, Star3, Star4: Serão transformadas em codificação One-Hot para que o modelo consiga processar as informações categóricas.

Variáveis Numéricas:

Released_Year: Pode ser utilizada como está ou transformada em variáveis categóricas (ex: "Anos 70", "Anos 80", etc.) para capturar padrões específicos de épocas.

Runtime: Pode ser utilizada como está.

No_of_Votes: Será transformada em logaritmo para reduzir a influência de valores extremos e obter uma distribuição mais equilibrada.

Meta_score: Será utilizada como está, com o tratamento de valores faltantes definido.

Gross: Será utilizada como está, com o tratamento de valores faltantes definido.

A transformação das variáveis categóricas e a normalização das variáveis numéricas visam melhorar o desempenho do modelo.

Dentre os modelos de regressão, a Random Forest se destaca como uma opção ideal. Ela é um modelo de ensemble que combina várias árvores de decisão para realizar a previsão, diminuindo o risco de overfitting.

Prós:

Alta precisão em dados com relações não-lineares complexas.

Robusto a outliers.

Boa capacidade de generalização.

Contras:

Maior tempo de treinamento em relação a modelos mais simples.

Dificuldade de interpretação em relação a modelos lineares.

Para avaliar o desempenho do modelo, a métrica RMSE (Erro Quadrático Médio) será utilizada. Essa medida é sensível a erros grandes e fornece uma indicação precisa do erro médio de previsão.

4. Supondo um filme com as seguintes características:

{'Series_Title': 'The Shawshank Redemption', 'Released_Year': '1994', 'Certificate': 'A', 'Runtime': '142 min', 'Genre': 'Drama', 'Overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.', 'Meta_score': 80.0, 'Director': 'Frank Darabont', 'Star1': 'Tim Robbins', 'Star2': 'Morgan Freeman', 'Star3': 'Bob Gunton', 'Star4': 'William Sadler', 'No_of_Votes': 2343110, 'Gross': '28,341,469'}
Segundo o modelo abaixo a nota IMDB para o filme 'The Shawshank Redemption' é: 9.30
