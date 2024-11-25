import os
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score # type: ignore
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.base import ClassifierMixin
from xgboost import XGBClassifier
import funcoes_suport
from funcoes_suport import export_columns_by_group_with_newline_csv, correlation_by_group_to_csv
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


######################
#     CSV READERS    # 
#                    #
######################
df = pd.read_csv('RawData/train_radiomics_hipocamp.csv') 
df_test = pd.read_csv('RawData/test_radiomics_hipocamp.csv')
#
# 305 linhas por 2181 colunas 
#


###########################################################################################################################################

# *? #####################
# *?   NULL VALUES       # 
# *?                     #
# *? #####################

#heat = sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#nullsum = df.isnull().sum()
#print(list(nullsum))
#for nullsum1 in  nullsum:
#    total = 0 
#    total += nullsum1
#    print(nullsum1)
#print("Total:", total)

# *? ########################################
# *? R:                                     #
# *? O DATASET NÂO APRESENTA NULL VALUES    #
# *? ########################################

###########################################################################################################################################

# *? ########################################
# *?    Single Values Coluns                #
# *?                                        #
# *? RETIRAR COLUNAS COM APENAS 1 VALOR     #
# *? de 2181 para 2022 Colunas              #
# *?                                        #    
# *? ########################################
df = df.loc[:, df.nunique() > 1]
df_test = df_test.loc[:, df_test.nunique() > 1]



#                               Analise das colunas que tem menos de 50 valores unicos 
#n = df.nunique()
#for col, e in n.items():
#    if e < 50:  
#        print(f"Coluna: {col}, Valores Unicos : {e}")



###########################################################################################################################################

# *? #####################
# *?  Valores Duplicados # 
# *?                     #
# *?      NÃO HÁ         #
# *? #####################

#a = df.dtypes
#
#for col, b in a.items():
#    print(f"Coluna: {col}, Tipo: {b}")
#
#  *? Não há duplicados 


###########################################################################################################################################
# *? #####################
# *?  MISSING VALUES     # 
# *?                     #
# *?      NÃO HÁ         #
# *? #####################
# Não há missing values 
#missing_data = df.isna().sum()
#print(missing_data[missing_data > 0])


###########################################################################################################################################

# *? ########################################
# *?    COLUNAS DE VALORES CATEGORICOS      #
# *?                                        #
# *? RETIRAR COLUNAS COM APENAS 1 VALOR     #
# *? de 2181 para 2022 Colunas              #
# *?                                        #    
# *? ########################################

# Analisar a contagem de valores únicos para cada coluna categórica
#
#for col in categorical_columns:
#    print(f"\nColuna: {col}")
#    print(df[col].value_counts())

#colunas_catagoricas_a_remover = ['ID', 'Image', 'Mask', 'diagnostics_Image-original_Hash', 'diagnostics_Mask-original_Hash'] 

colunas_catagoricas_a_remover = ['ID', 'Image', 'Mask', 'diagnostics_Image-original_Hash', 'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass'] 

# ** Bounding Box
#
# ** as colunas do 'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass'
# ** Deveriam ser retiradas, mas o bounding box para ser importante para a zona de maior ativação do Alzimeir 
# **  ja a de centro de maxima devem ser muito correlacionados, por isso devem ser retirados mais para a frente 
# *TODO acabei por retirar para correr melhor os modelos, mas analisar se se deve retirar ou nao 

df.drop(columns=colunas_catagoricas_a_remover,axis= 1 , inplace= True)
df_test.drop(columns=colunas_catagoricas_a_remover,axis= 1 , inplace= True)

# ! 2018 colunas

###########################################################################################################################################

# *? ########################################
# *?   AGE HANDLER                          #
# *?                                        #
# *? ########################################


age_bins = [0, 65, 75, 85, 100]
# BINS_SIZER = ['<65', '65-74', '75-84', '85+']
age_labels = [60, 70, 80, 90] # VALOR MEDIO DO BIN 
df['Age'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels).astype(int)
df_test['Age'] = pd.cut(df_test['Age'], bins=age_bins, labels=age_labels).astype(int)

###########################################################################################################################################
# *? ########################################
# *?  ELIMINAR LINHAS CN-MCI                #
# *?                                        #
# *? ########################################

print(df.shape)
df = df[df['Transition'] != 'CN-MCI']
print(df.shape)



label_mapping = {
    'CN-CN': 0,
    'AD-AD': 1,
    'MCI-AD': 2,
    'MCI-MCI': 3, 
#    'CN-MCI' : 4

}


# Apply the mapping to the target column
df['Transition'] = df['Transition'].map(label_mapping)

###########################################################################################################################################
# *? ########################################
# *? Normalização 
# *?
# *?
# *? ########################################


df_normalizado = df.copy()
df_normalizado_test = df_test.copy()

## Armazenar a coluna 'Transition'
transition_column = df_normalizado['Transition']

# Selecionar as colunas para normalização (todas menos 'Transition')
features = df_normalizado.drop(columns='Transition')

# Normalizar as colunas selecionadas
scaler = MinMaxScaler()
df_normalizado[features.columns] = scaler.fit_transform(features)
df_normalizado_test_scaled = scaler.transform(df_normalizado_test)
df_normalizado_test = pd.DataFrame(df_normalizado_test_scaled, columns=df_normalizado_test.columns)
# Adicionar a coluna 'Transition' de volta
df_normalizado['Transition'] = transition_column



###########################################################################################################################################

# *? ########################################
# *?   COLUNAS CORRELACIONADAS   ENTRE SIM  #
# *?                                        #   
# *? ########################################




df_1 = df.copy()
df_1_test = df_test.copy()


# *? REMOÇÃO DE COLUNAS COM POUCO CORRELACAO COM O TARGET
# Calcular as correlações com a coluna 'Transition'
correlations = df_1.corr()['Transition']

# Identificar colunas que têm correlação menor que 0.13
low_correlation_columns = correlations[correlations.abs() < 0.1].index

# Remover a coluna 'Transition' da lista, se presente
low_correlation_columns = low_correlation_columns[low_correlation_columns != 'Transition']

# Armazenar o número de colunas a serem removidas
num_removed_columns = len(low_correlation_columns)

# Remover as colunas do DataFrame
df_1_filtered = df_1.drop(columns=low_correlation_columns)
df_1_filtered_test = df_1_test.drop(columns=low_correlation_columns)

# Exibir o número de colunas removidas
#print(f"Número de colunas removidas: {num_removed_columns}")



# *? REMOÇÃO DE GRUPOS COM POUCO CORRELACAO COM O TARGET

groups_to_remove = [
    'lbp-3D-k',
    'wavelet-LHL',
    'wavelet-LHH',
    'wavelet-HLH',
    'wavelet-HHL',
    'log-sigma-5-0',
    'square_',
    'exponential'

]

# Encontrar colunas que começam com os grupos especificados
columns_to_remove = [col for col in df_1_filtered.columns if any(col.startswith(group) for group in groups_to_remove)]

# Armazenar o número de colunas a serem removidas
num_removed_columns = len(columns_to_remove)

# Remover as colunas do DataFrame

df_1_final = df_1_filtered.drop(columns=columns_to_remove)
df_1_final_test = df_1_filtered_test.drop(columns=columns_to_remove)






###########################################################################################################################################

# *? ##########################
# *? IMPRESSAO DATASET 
# *?
# *? ##########################



#df_1_final.to_csv('Data/1. Dataset Competicao.csv', index=False)
#df_1_final_test.to_csv('Data/1. Dataset Test Competicao.csv', index=False)




############################################################################################################################################
#
## *! ######################
## *!        MODELS                  
## *! ######################
#
# ** GERAR A MELHOR SUBMISSAO 
best_score = float(0.0); 
modelo_escolhido = None


# Path to the file that stores the best score
best_score_file = 'best_score.txt'

# Check if the file exists and load the stored best score and model name, else initialize
if os.path.exists(best_score_file):
    with open(best_score_file, 'r') as file:
        lines = file.readlines()
        best_score_ever = float(lines[0].strip())
        best_model_name = lines[1].strip() if len(lines) > 1 else "N/A"
else:
    best_score_ever = 0.0
    best_model_name = "N/A"

# Print current best score and model name
print(f"Current Best Score Stored: {best_score_ever * 100:.20f}%")
print(f"Model with Best Score: {best_model_name}\n")









# *?    #############
# *TODO #############
# *! ESCOLHER AQUI O DATA PARA USAR NO MODELO E O DATASET TEST
data_model = df_1_final; 
Dataset_test = df_1_final_test; 



print(data_model.shape)
print(Dataset_test.shape)



# Split data
X = data_model.drop('Transition', axis=1)
y = data_model['Transition']
print("Original:")
print(y.value_counts())
#undersample = RandomUnderSampler(sampling_strategy={0: 60, 3: 60, 2: 60, 1: 60}, random_state=2022)
#smote = SMOTE(sampling_strategy={0: 60, 1: 60, 2: 60, 3: 60}, random_state=2022)
#
## Crie o pipeline com undersampling seguido por SMOTE
#pipeline = Pipeline([
#    ('undersample', undersample),  # Reduz classes para 60 onde houver excesso
#    ('smote', smote)               # Aumenta classes para 60 onde houver necessidade
#])
#
## Aplicar o pipeline aos dados
#X_res, y_res = pipeline.fit_resample(X, y)
#smote = SMOTE(random_state=2022)
#X_res, y_res = smote.fit_resample(X, y)


#print('Resampled dataset shape %s %s' % X_res.shape,y_res.shape)
#print(y_res.value_counts())
#X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=2022, stratify=y_res)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2022, stratify=y)



# *! ####################################################
# *! SVM

#  ** best_params = {'svm__C': 1, 'svm__gamma': 0.001}
#  ** final_svm_model = SVC(kernel='rbf', C=best_params['svm__C'], gamma=best_params['svm__gamma'])
#  ** 
#  ** final_svm_model.fit(X_train, y_train)
#  ** 
#  ** final_accuracy = final_svm_model.score(X_test, y_test)
#  ** print("Final SVM Test Accuracy with Best Parameters: {:.2f}%".format(final_accuracy * 100))
#  ** 
#  ** 
#  ** if final_accuracy > best_score : 
#  **     best_score = final_accuracy
#  **     modelo_escolhido = final_svm_model
#  ** 


# *! ####################################################
# *! DECISION TREE 

# Decision Tree Model
decision_tree_model = DecisionTreeClassifier(random_state=2022)
decision_tree_model.fit(X_train, y_train)
decision_tree_cross_val_score = cross_val_score(decision_tree_model, X, y, cv=10)
decision_tree_score = decision_tree_model.score(X_test,y_test)

if decision_tree_cross_val_score.mean() > best_score : 
    best_score = decision_tree_cross_val_score.mean()
    modelo_escolhido = decision_tree_model


print("Decision Tree Scores: %.2f%%" % (decision_tree_score * 100))
print("Average Decision Tree Accuracy: %.2f%%" % (decision_tree_cross_val_score.mean() * 100))
print()

# *! ####################################################
# *! RANDOM FOREST 

# Random Forest Model
random_forest_model = RandomForestClassifier(n_estimators=800, max_features='sqrt', random_state=2022)
random_forest_model.fit(X_train, y_train)
random_forest_cross_val_score = cross_val_score(random_forest_model, X, y, cv=10)
random_forest_score = random_forest_model.score(X_test,y_test)

if random_forest_cross_val_score.mean() > best_score : 
    best_score = random_forest_cross_val_score.mean()
    modelo_escolhido = random_forest_model

print("Random Forest Scores: %.2f%%" % (random_forest_score * 100))
print("Average Random Forest Accuracy: %.2f%%" % (random_forest_cross_val_score.mean() * 100))
print()



# *! ####################################################
# *! XGBOOST 



# Initialize and fit the model
xgBoost_model = XGBClassifier(n_estimators=800, max_depth=4, learning_rate=0.05,
                            colsample_bytree=0.4, subsample=0.8, random_state=2022)
xgBoost_model.fit(X_train, y_train)

xgBoost_cross_val_score = cross_val_score(xgBoost_model, X, y, cv=10)

xgBoost_score = xgBoost_model.score(X_test,y_test)


if xgBoost_cross_val_score.mean() > best_score : 
    best_score = xgBoost_cross_val_score.mean()
    modelo_escolhido = xgBoost_model


print("XGBOOST Acurracy: %.2f%%" % (xgBoost_score * 100))
print("Average XGBOOST Accuracy: %.2f%%" % (xgBoost_cross_val_score.mean() * 100))
print()


# *! ######################################################

extra_tress_model = ExtraTreesClassifier(criterion='gini', max_depth=20, random_state=2022)
extra_tress_model.fit(X_train, y_train)

extra_tress_cross_val_score = cross_val_score(extra_tress_model, X,y, cv=10 )
extra_tress_score = extra_tress_model.score(X_test,y_test)

if extra_tress_cross_val_score.mean() > best_score : 
    best_score = extra_tress_cross_val_score.mean()
    modelo_escolhido = extra_tress_model


print("Extra Trees Acurracy: %.2f%%" % (extra_tress_score * 100))
print("Average Extra Trees Accuracy: %.2f%%" % (extra_tress_cross_val_score.mean() * 100))
print()



# *! ####################################################
# *! STACKING 

# ** estimators = [("random_forest", random_forest_model),("XGboost", xgBoost_model)]
# ** staking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
# ** 
# ** staking_model.fit(X_train, y_train)
# ** 
# ** staking_score = staking_model.score(X_test, y_test)
# ** 
# ** if staking_score > best_score : 
# **     best_score = staking_score
# **     modelo_escolhido = staking_model
# ** 
# ** print("Stacking Accurary : %2ff%%" % (staking_score *100))
# ** print()















###########################################################################################################################################


# *? ########################################
# *?   Geração Da SUBMISSAO                 #
# *?                                        #
# *? ########################################




if best_score > best_score_ever:
# *!                                          MODELO          Dataset de testes  
    funcoes_suport.generate_predictions_csv(modelo_escolhido,      Dataset_test         )# If the new best score is higher, update the file





if best_score > best_score_ever:
    with open(best_score_file, 'w') as file:
        file.write(f"{best_score}\n")  # Write the best score on the first line
        file.write(f"{modelo_escolhido.__class__.__name__}\n")  # Write the name of the model on the second line
    print(f"New Best Score {best_score * 100:.2f}% saved in {best_score_file} with model: {modelo_escolhido.__class__.__name__}")
else:
    print("No improvement over the best score.")












