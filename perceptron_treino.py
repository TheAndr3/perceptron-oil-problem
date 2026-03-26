import numpy as np

ETA = 0.01  # Taxa de aprendizagem 
X0 = -1     # Bias 

def funcao_ativacao(u):
    #Convenção: 1 para C2 e 0 para C1 
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
            
            # (u) 
            u = np.dot(x, pesos)
            
            y = funcao_ativacao(u)

            #Aprendizado 
            if y != d:
                erro = d - y
                pesos = pesos + ETA * erro * x  #Regra de Hebb
                erros_na_epoca += 1
        
        #  treino para quando não houver mais erros na época
        if erros_na_epoca == 0 or epocas > 1000:
            break
    
    return pesos_iniciais, pesos, epocas
    
dados_treino = np.loadtxt('dados-tra.txt')
X_treino = dados_treino[:, :2]
d_treino = dados_treino[:, 2]

for t in range(1, 6):
    w_init, w_final, total_epocas = treinar_perceptron(X_treino, d_treino)
    
    # Formatação para 4 casas decimais como você salvou no txt
    init_str = f"{w_init[0]:.4f}, {w_init[1]:.4f}, {w_init[2]:.4f}"
    final_str = f"{w_final[0]:.4f}, {w_final[1]:.4f}, {w_final[2]:.4f}"
    
    print(f"{t}º (T{t})  | {init_str:<28} | {final_str:<28} | {total_epocas}")

    # Opcional: Salvar o último treinamento para usar no teste depois
    if t == 5:
        np.savetxt('pesos_finais.txt', w_final, fmt='%.4f')