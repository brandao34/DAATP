import matplotlib.pyplot as plt
import numpy as np

# Dados (removendo o 0 inicial para evitar log(0))
x = np.array([42, 84, 168, 336])
y = np.array([1.131, 3.588, 23.951, 194.329])

# Criar o gráfico em escala log-log
plt.figure(figsize=(8, 6))
plt.loglog(x, y, marker='o', linestyle='-', color='b', label='Tempo vs x')

# Adicionar títulos e rótulos
plt.title("Gráfico em Escala Log-Log", fontsize=14)
plt.xlabel("x (log scale)", fontsize=12)
plt.ylabel("Tempo (log scale)", fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()

# Exibir o gráfico
plt.show()
