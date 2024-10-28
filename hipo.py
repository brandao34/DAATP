from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
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
from sklearn.preprocessing import KBinsDiscretizer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from collections import Counter




from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



df = pd.read_csv('train_radiomics_hipocamp.csv') 
#print(list(df.columns))
#for column in df.columns:
#    print(column)

orinal_shape = df.shape
print(orinal_shape)
print("orginal info")
#df.info()
#
#
# 305 linhas por 2181 colunas 
#


                                                    # NULL VALUES 
###########################################################################################################################################
#heat = sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#nullsum = df.isnull().sum()
#print(list(nullsum))
#for nullsum1 in  nullsum:
#    total = 0 
#    total += nullsum1
#    print(nullsum1)
#print("Total:", total)

# R: 

# O dataset nao apresenta nenhum nenhuma coluna com o valor null 


###########################################################################################################################################

# Selecionar colunas categóricas

#                                 RETIRAR COLUNAS COM APENAS 1 VALOR 
#                                  de 2181 para 2022 Colunas 
df = df.loc[:, df.nunique() > 1]
#shape_v2 = df.shape
#print(shape_v2)


#                               Analise das colunas que tem menos de 50 valores unicos 
#n = df.nunique()
#for col, e in n.items():
#    if e < 50:  
#        print(f"Coluna: {col}, Valores Unicos : {e}")
###########################################################################################################################################

#a = df.dtypes
#
#for col, b in a.items():
#    print(f"Coluna: {col}, Tipo: {b}")
#

# Não há duplicados 

print("valores duplicados", df.duplicated().sum())

###########################################################################################################################################

# Não há missing values 
missing_data = df.isna().sum()
print(missing_data[missing_data > 0])


###########################################################################################################################################

#categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# Exibir os nomes das colunas categóricas
#print("Colunas categóricas:", list(categorical_columns))

# Analisar a contagem de valores únicos para cada coluna categórica
#for col in categorical_columns:
#    print(f"\nColuna: {col}")
#    print(df[col].value_counts())

colunas_catagoricas_a_remover = ['Image', 'Mask', 'diagnostics_Image-original_Hash', 'diagnostics_Mask-original_Hash'] 

#Bounding Box
#
# as colunas do 'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass'
# Deveriam ser retiradas, mas o bounding box para ser importante para a zona de maior ativação do Alzimeir 
# ja a de centro de maxima devem ser muito correlacionados, por isso devem ser retirados mais para a frente 

df.drop(columns=colunas_catagoricas_a_remover,axis= 1 , inplace= True)


# 2018 colunas

###########################################################################################################################################


df1 = df.copy()



# Calcular a matriz de correlação
correlation_matrix = df1.corr(numeric_only=True).abs()

# Selecionar os pares de colunas com correlação maior que 0.95 (95%) e abaixo de 1 (evita self-correlation)
high_corr_pairs = [
    (col1, col2) 
    for col1 in correlation_matrix.columns 
    for col2 in correlation_matrix.columns 
    if col1 != col2 and correlation_matrix.loc[col1, col2] > 0.95
]

# Remover duplicatas de pares (já que A-B é o mesmo que B-A)
high_corr_pairs = list(set(tuple(sorted(pair)) for pair in high_corr_pairs))

# Exibir os pares de colunas com alta correlação
#print("Pares de colunas com correlação > 90%:")
#for pair in high_corr_pairs:
   # print(pair)


colunas_a_remover = set()


# Contar a frequência de cada coluna nos pares correlacionados
column_counts = Counter()
for col1, col2 in high_corr_pairs:
    column_counts[col1] += 1
    column_counts[col2] += 1

# Identificar as colunas a serem removidas
for col1, col2 in high_corr_pairs:
    # Comparar as contagens e remover a coluna que aparece menos vezes
    if column_counts[col1] < column_counts[col2]:
        colunas_a_remover.add(col1)
    else:
        colunas_a_remover.add(col2)


#print(column_counts.total())
# Remover as colunas do DataFrame
df1.drop(columns=colunas_a_remover, axis=1 ,inplace= True) 

num_colunas_removidas = len(colunas_a_remover)
print(num_colunas_removidas)





#df1.info()
# Apos remover as colunas com 95 porcento de correlacao, temos 874 colunas 

###########################################################################################################################################
