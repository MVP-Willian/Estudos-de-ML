import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

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
    categorie = 1
    if(dataFrame['species'][instance] not in categories):
        categories.append(dataFrame['species'][instance])
#ou 
categories = dataFrame['species'].unique()
print(categories)

#parte de mapear o dataFrame de acordo com as categorias indentifacadas usando o map
dataFrame["species"] = dataFrame["species"].map({"setosa":1, "versicolor":2, "virginica":3})
#caso não saiba o número de categorias:
dataFrame['species'] = dataFrame['species'].map(dict(zip(categories, range(len(categories))))) 

#separar os targets das features, o parâmetro index do drop pode ser 1:colunas ou 0:linhas 
y = dataFrame['species'].values
x = dataFrame.drop('species', index=1)

vectorizer = CountVectorizer()
