
######################
#     CSV READERS    # 
#                    #
######################


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from model_suport import load_best_score, plot_confusion_matrix_with_labels
import optuna

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split


data = pd.read_csv('../Data/1. Dataset Competicao.csv') 
data_test = pd.read_csv('../Data/1. Dataset Test Competicao.csv')


X = data.drop('Transition', axis=1)
y = data['Transition']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=34, stratify=y)


best_score, _ = load_best_score()





# Random Forest Model
#random_forest_model = RandomForestClassifier(n_estimators=800, max_features='sqrt', random_state=2022)
#random_forest_model.fit(X_train, y_train)
#random_forest_cross_val_score = cross_val_score(random_forest_model, X, y, cv=10)
#random_forest_score = random_forest_model.score(X_test,y_test)



#print("Random Forest Scores: %.2f%%" % (random_forest_score * 100))
#print("Average Random Forest Accuracy: %.2f%%" % (random_forest_cross_val_score.mean() * 100))
#print()
#

#if random_forest_cross_val_score.mean() > best_score : 
#    best_score = random_forest_cross_val_score.mean()
#    print("Best Score ever: %.2f%%" % (random_forest_cross_val_score.mean()))

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)  # Número de árvores
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])  # Número de features
    max_depth = trial.suggest_int('max_depth', 5, 20)  # Profundidade máxima
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)  # Min amostras para split
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)  # Min amostras em folha
    bootstrap = trial.suggest_categorical('bootstrap', [False, True])  # Bootstrap



    model1 = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=34,
        bootstrap=bootstrap
    )

    scores_5 = cross_val_score(model1, X_train, y_train, cv=5, scoring='accuracy')
    scores_10 = cross_val_score(model1, X_train, y_train, cv=10, scoring='accuracy')

    accuracy_5 = scores_5.mean()
    accuracy_10 = scores_10.mean()

    return max(accuracy_5, accuracy_10) 

# Criação do estudo utilizando TPE como método de amostragem
#study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
#study.optimize(objective, n_trials=100)  # número de iterações (trials)
#Resultado
#print(f"Melhor acurácia: {study.best_value}")
#print(f"Melhores hiperparâmetros: {study.best_params}")


# *? ###################################################################################################################################################################
# *?     ACCURACY    ##          DataSet        ##      Parametros
# *? ###################################################################################################################################################################
# *?     0.49822     ##  1. Dataset Competicao  ##  {'n_estimators': 151, 'max_features': 'log2', 'max_depth': 13, 'min_samples_split': 10, 'min_samples_leaf': 3, 'bootstrap': True} 
# *? 
# *? 







random_forest_model = RandomForestClassifier(n_estimators= 151, max_features= 'log2', max_depth= 13, min_samples_split=10, min_samples_leaf=3, bootstrap= True, random_state=34 )
random_forest_model.fit(X_train, y_train)
random_forest_cross_val_score = cross_val_score(random_forest_model, X_train, y_train, cv=10)
random_forest_cross_val_score_scoring = cross_val_score(random_forest_model, X_train, y_train, cv=5, scoring='f1_macro')

print(f"Accuracy score: {random_forest_cross_val_score.mean()}")
print(f"Standard deviation: {random_forest_cross_val_score.std()}")

print(f"Accuracy score scoring: {random_forest_cross_val_score_scoring.mean()}")
print(f"Standard deviation scoring: {random_forest_cross_val_score_scoring.std()}")



label_mapping = {
    'CN-CN': 0,
    'AD-AD': 1,
    'MCI-AD': 2,
    'MCI-MCI': 3, 
#    'CN-MCI' : 4

}



y_pred = random_forest_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix_with_labels(cm)