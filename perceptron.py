import numpy as np

ETA = 0.01  # Taxa de aprendizagem 
X0 = -1     # Bias 

def funcao_ativacao(u):
    return 1 if u >= 0 else 0

def treinar_perceptron(entradas, saídas_desejadas):
    pesos = np.random.rand(3) 
    pesos_iniciais = pesos.copy()
    
    epocas = 0
    while True:
        erros_na_epoca = 0
        epocas += 1
        for i in range(len(entradas)):
            x = np.array([X0, entradas[i][0], entradas[i][1]])
            d = saídas_desejadas[i]
            u = np.dot(x, pesos)
            y = funcao_ativacao(u)

            if y != d:
                erro = d - y
                pesos = pesos + ETA * erro * x  
                erros_na_epoca += 1
        
        if erros_na_epoca == 0 or epocas > 1000:
            break
    
    return pesos_iniciais, pesos, epocas
    
# Preparação 
dados_treino = np.loadtxt('dados-tra.txt')
X_treino = dados_treino[:, :2]
d_treino = dados_treino[:, 2]
lista_de_pesos_finais = []

print(f"{'Treino':<8} | {'Iniciais (w0, w1, w2)':<28} | {'Finais (w0, w1, w2)':<28} | {'Épocas'}")
print("-" * 80)

for t in range(1, 6):
    w_init, w_final, total_epocas = treinar_perceptron(X_treino, d_treino)
    
    lista_de_pesos_finais.append(w_final)
    
    init_str = f"{w_init[0]:.4f}, {w_init[1]:.4f}, {w_init[2]:.4f}"
    final_str = f"{w_final[0]:.4f}, {w_final[1]:.4f}, {w_final[2]:.4f}"
    print(f"{t}º (T{t})  | {init_str:<28} | {final_str:<28} | {total_epocas}")

    if t == 5:
        np.savetxt('pesos_finais.txt', w_final, fmt='%.4f')

# Fase de Teste 

try:
    X_teste = np.loadtxt('dados-tst.txt')[:, :2] 

    print("\n" + "="*85)
    print("TABELA DE CLASSIFICAÇÃO (ITEM 3)")
    print(f"{'Amostra':<8} | {'x1':<8} | {'x2':<8} | {'y(T1)':<6} | {'y(T2)':<6} | {'y(T3)':<6} | {'y(T4)':<6} | {'y(T5)':<6}")
    print("-" * 85)

    for i, amostra in enumerate(X_teste):
        saidas_dos_modelos = []
        x_input = np.array([X0, amostra[0], amostra[1]])
        
        for pesos in lista_de_pesos_finais:
            u = np.dot(x_input, pesos)
            y = funcao_ativacao(u)
            saidas_dos_modelos.append(y)
        
        res = "  |  ".join([f"{val}" for val in saidas_dos_modelos])
        print(f"{i+1:<8} | {amostra[0]:<8.4f} | {amostra[1]:<8.4f} |   {res}")

except Exception as e:
    print(f"Erro no teste: {e}")