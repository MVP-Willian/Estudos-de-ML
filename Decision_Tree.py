import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


iris = sns.load_dataset("iris")
iris.to_csv("iris.csv", index=False)


#Carregar certo dataframe que está em csv
dataFrame = pd.read_csv("iris.csv")
dataFrame2 = pd.read_csv("iris.csv")

#imprimir as primeiras linhas de um data set
print(dataFrame.head())

#Renomear colunas, parâmetro inplace serve para modificar no dataFrame original, porém no altera no arquivo principal do diretório ainda.
dataFrame.rename(columns={"sepal_length":"Comprimento da sépala"}, inplace=True)
print(dataFrame.head())

#mapeando classes em valores discretos(classificação)
#etapa para separação das categorias:
categories  = []
for instance in range(len(dataFrame)):
    if(dataFrame['species'][instance] not in categories):
        categories.append(dataFrame['species'][instance])
#ou 
categories = dataFrame['species'].unique()
print(categories)

#parte de mapear o dataFrame de acordo com as categorias indentifacadas usando o map
dataFrame["species"] = dataFrame["species"].map({"setosa":1, "versicolor":2, "virginica":3})
#caso não saiba o número de categorias: 
#zip associa elemento de indice x a outro elemento de outra estrutura de indice x também.
dataFrame['species'] = dataFrame['species'].map(dict(zip(categories, range(len(categories))))) 

#separar os targets das features, o parâmetro index do drop pode ser 1:colunas ou 0:linhas 
y = dataFrame['species']
x = dataFrame.drop(columns='species')

y2 = dataFrame2['sepal_length']
x2 = dataFrame2.drop(columns='sepal_length')

#transformar Palavras em uma matriz númerica para treinar o modelo
'''
fit(): A etapa de "aprendizado" ou "captação". É aqui que a ferramenta analisa os seus dados para coletar as informações necessárias para a transformação. No nosso caso, é o momento em que o CountVectorizer constrói o vocabulário a partir de todos os textos.

transform(): A etapa de "implementação". Aqui, a ferramenta usa o que aprendeu no fit() para aplicar a transformação nos dados. No nosso caso, o CountVectorizer usa o vocabulário que ele acabou de construir para converter o texto em uma matriz numérica.

No caso do CountVectorizer a saída depois de aplicar o transform são matrizes esparças 
'''
x_only_test = x2['species']
vectorizer = CountVectorizer()
vectorizer.fit(x_only_test)
x2_transformed = vectorizer.transform(x_only_test)
#ou
x2_transformed = vectorizer.fit_transform(x_only_test)

print(type(x2_transformed))
#juntar as matrizes depois de tratar separada, np.hstack: faz uma junção de colunas de arrays
#to.numpy() funções geralmente utilizada para transformar valores númericos de tabelas csv para arrays
#toarray() geralmente usada em matrizes esparsas(matrizes com valores faltantes por questão de memória) com finalidade de preencher esse valores faltantes
x2_final = np.hstack((x2.drop(columns='species').to_numpy(), x2_transformed.toarray()))

# dividir os dados em dados de treino e dados de teste
#train_test_split(dadosDeEntrada, dadosDeSaida, test_size=porcentagem de dados para teste)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(x2_final, y2, test_size=0.2)



