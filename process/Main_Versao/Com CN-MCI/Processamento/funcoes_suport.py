
from sklearn.base import ClassifierMixin
import pandas as pd

def generate_predictions_csv(model: ClassifierMixin, test_data: pd.DataFrame):
    """
    Gera um arquivo CSV com as previsões do modelo fornecido e o conjunto de teste.

    Parameters:
    - model: Classificador treinado (ex: RandomForestClassifier, DecisionTreeClassifier).
    - test_data: DataFrame de teste para o qual as previsões serão feitas.
    - output_filename: Nome do arquivo de saída para salvar as previsões (padrão é 'predictions.csv').
    """
    # Realizar as previsões
    predictions = model.predict(test_data)
    
    # Mapeamento inverso para converter de números para labels
    label_mapping = {
        0: 'CN-CN',
        1: 'AD-AD',
        2: 'MCI-AD',
        3: 'MCI-MCI',
        4:  'CN-MCI'

    }
    
    # Converter as previsões numéricas para as labels correspondentes
    predictions_labels = [label_mapping[pred] for pred in predictions]
    
    # Criar DataFrame com as previsões
    predictions_df = pd.DataFrame(predictions_labels, columns=['Result'])
    predictions_df.index = range(1, len(predictions_df) + 1)
    predictions_df.index.name = 'RowId'
    
    # Exibir a contagem de cada label prevista
    counts = predictions_df['Result'].value_counts()
    print("Contagem de previsões para cada label:")
    print(counts)
    
    # Extrair o nome do modelo para o arquivo de saída
    model_name = str(model)  
    if '(' in model_name:
        output_filename = model_name.split('(')[0] + ".csv"
    else:
        output_filename = model_name + ".csv"
    
    # Salvar o DataFrame em um arquivo CSV
    predictions_df.to_csv(output_filename, index=True)
    print(f"Resultados salvos no arquivo {output_filename}")


import os

def load_best_score():
    # Path to the file that stores the best score
    best_score_file = 'best_score.txt'

    # Initialize default values
    best_score_ever = 0.0
    best_model_name = "N/A"

    # Check if the file exists and load the stored best score and model name
    if os.path.exists(best_score_file):
        with open(best_score_file, 'r') as file:
            lines = file.readlines()
            best_score_ever = float(lines[0].strip())
            best_model_name = lines[1].strip() if len(lines) > 1 else "N/A"

    # Print current best score and model name
    print(f"Current Best Score Stored: {best_score_ever * 100:.20f}%")
    print(f"Model with Best Score: {best_model_name}\n")

    return best_score_ever, best_model_name










def export_columns_by_group_with_newline_csv(dataset: pd.DataFrame, output_filename: str = 'columns_by_group.csv'):
    """
    Exporta as colunas do DataFrame em grupos definidos pelos prefixos e inclui uma linha de cabeçalho para cada grupo 
    com o nome do grupo e a quantidade de colunas que possui.
    
    Parameters:
    - dataset: DataFrame contendo o conjunto de dados.
    - output_filename: Nome do arquivo CSV de saída (padrão é 'columns_by_group.csv').
    """
    # Define os grupos com base nos prefixos
    groups = ['original',
        'wavelet-LLH', 'wavelet-LHL', 'wavelet-LHH', 'wavelet-HLL', 'wavelet-HLH', 
        'wavelet-HHL', 'wavelet-HHH', 'wavelet-LLL', 'log-sigma-1-0', 'log-sigma-2-0', 
        'log-sigma-3-0', 'log-sigma-4-0', 'log-sigma-5-0', 'square_', 'squareroot', 
        'logarithm', 'exponential', 'gradient', 'lbp-2D', 'lbp-3D'
    ]
    
    with open(output_filename, 'w') as file:
        for group in groups:
            # Filtra as colunas que começam com o prefixo do grupo
            group_columns = [col for col in dataset.columns if col.startswith(group)]
            
            if group_columns:
                # Escreve o cabeçalho do grupo com o nome e a quantidade de colunas
                file.write(f"{group}, {len(group_columns)}\n")
                
                # Escreve cada coluna do grupo em uma linha
                for column in group_columns:
                    file.write(f"{column}\n")
                
                # Adiciona uma linha em branco para separar os grupos
                file.write("\n")
    
    print(f"Dataset salvo com as colunas organizadas em grupos no arquivo {output_filename}")



def correlation_by_group_to_csv(dataset: pd.DataFrame, target: str = 'Transition', output_filename: str = 'correlations_by_group.csv'):
    """
    Exporta as correlações de cada grupo de colunas com a coluna alvo em um arquivo CSV, 
    incluindo um resumo no início do arquivo e correlações ao lado de cada coluna.
    
    Parameters:
    - dataset: DataFrame contendo o conjunto de dados.
    - target: Nome da coluna alvo (padrão é 'Transition').
    - output_filename: Nome do arquivo CSV de saída (padrão é 'correlations_by_group.csv').
    """
    # Atualizado: Define os grupos com base nos prefixos fornecidos
    groups = [
        'original',
        'wavelet-LLH', 'wavelet-LHL', 'wavelet-LHH', 'wavelet-HLL', 'wavelet-HLH', 
        'wavelet-HHL', 'wavelet-HHH', 'wavelet-LLL', 'log-sigma-1-0', 'log-sigma-2-0', 
        'log-sigma-3-0', 'log-sigma-4-0', 'log-sigma-5-0', 'square_', 'squareroot', 
        'logarithm', 'exponential', 'gradient', 'lbp-2D', 'lbp-3D-k', 'lbp-3D-m1', 'lbp-3D-m2'
    ]
    
    # Lista para armazenar o resumo das correlações
    summary = []

    # Calcular as correlações e armazenar no resumo
    for group in groups:
        # Seleciona as colunas do grupo atual
        group_columns = [col for col in dataset.columns if col.startswith(group)]
        
        if group_columns:
            # Calcula a correlação entre cada coluna do grupo e o target
            group_data = dataset[group_columns + [target]]
            corr = group_data.corr()[target][:-1]  # Exclui a correlação de 'target' consigo mesmo
            
            # Calcula a média do módulo das correlações
            mean_correlation = corr.abs().mean()
            
            # Adiciona as informações do grupo ao resumo
            summary.append((group, len(group_columns), mean_correlation))
    
    # Escreve o resumo e os detalhes no arquivo CSV
    with open(output_filename, 'w') as file:
        # Escreve o cabeçalho do resumo
        file.write(f"Group, Number of Columns, Mean Absolute Correlation\n")
        for group, count, mean_corr in summary:
            file.write(f"{group}, {count}, {mean_corr:.4f}\n")
        
        # Adiciona uma linha em branco após o resumo
        file.write("\n")
        
        # Escreve os detalhes de cada grupo
        for group, _, _ in summary:
            # Seleciona as colunas do grupo atual novamente
            group_columns = [col for col in dataset.columns if col.startswith(group)]
            
            # Calcula a correlação entre as colunas do grupo e o target
            group_data = dataset[group_columns + [target]]
            corr = group_data.corr()[target][:-1]  # Exclui a correlação de 'target' consigo mesmo
            
            # Escreve o cabeçalho do grupo
            file.write(f"\nGroup: {group}\n")
            
            # Escreve cada coluna do grupo com seu valor de correlação absoluta
            for column in group_columns:
                correlation_value = corr[column]
                file.write(f"{column}, {correlation_value:.4f}\n")
            
            top = int(len(group_columns)/2)
            # Top 10 colunas mais correlacionadas (baseado no módulo)
            top_10 = corr.abs().nlargest(top)
            file.write("\nTop 10 Most Correlated:\n")
            for column, value in top_10.items():
                file.write(f"{column}, {value:.4f}\n")
            
            # Top 10 colunas menos correlacionadas (baseado no módulo)
            bottom_10 = corr.abs().nsmallest(top)
            file.write("\nTop 10 Least Correlated:\n")
            for column, value in bottom_10.items():
                file.write(f"{column}, {value:.4f}\n")
            
            # Adiciona uma linha em branco para separar os grupos
            file.write("\n")
    
    print(f"Correlações salvas no arquivo {output_filename}")

# Exemplo de uso
# correlation_by_group_to_csv(df_0_age)

