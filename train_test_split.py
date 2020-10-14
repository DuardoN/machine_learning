from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

seed = 20 # definir números aleatórios
# stratify ajuda na proporção de treino e teste. Importante para o mesmo.
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = seed, test_size = 0.25, stratify = y)
print("Treinos com %d elementos e teste com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

taxa_acerto = accuracy_score(teste_y, previsoes) * 100
print("A acuracia foi %.2f%%" % taxa_acerto)