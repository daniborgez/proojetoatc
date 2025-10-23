import pandas as pd
import numpy as np
from scipy.optimize import linprog
import os

def carregar_dados_csv(nome_nutrientes="nutrientes.csv", nome_alimentos="ingrediente.csv"):
    """
    Carrega os dados dos arquivos CSV de forma robusta e os prepara em DataFrames.
    """
    if not os.path.exists(nome_nutrientes) or not os.path.exists(nome_alimentos):
        raise FileNotFoundError(f"Um ou ambos os arquivos CSV ('{nome_nutrientes}' e '{nome_alimentos}') não foram encontrados. Certifique-se de que estão na mesma pasta do script.")

    # 1. Carregar dados de Nutrientes
    dados_nutrientes = pd.read_csv(nome_nutrientes, sep=',', encoding='utf-8')
    
    # 2. Carregar dados de Alimentos
    try:
        # Tenta a codificação padrão e separador de vírgula
        dados_alimentos = pd.read_csv(nome_alimentos, sep=',', encoding='utf-8')
    except Exception:
        # Em caso de erro, tenta a codificação Latin-1 (comum em sistemas Windows)
        dados_alimentos = pd.read_csv(nome_alimentos, sep=',', encoding='latin1')
    
    # 3. Tratamento de Dados (Limpeza de nomes de colunas e conversão)
    dados_alimentos.columns = dados_alimentos.columns.str.strip()
    
    colunas_chave = ['ingrediente', 'quantidade']
    
    # Verifica se as colunas estão presentes
    for col in colunas_chave:
        if col not in dados_alimentos.columns:
            raise KeyError(f"A coluna '{col}' não foi encontrada nos dados. Por favor, verifique se seu arquivo CSV está correto. Colunas lidas: {dados_alimentos.columns.tolist()}")

    # Colunas a serem convertidas (todos os nutrientes e o preço)
    colunas_numericas = dados_alimentos.columns.drop(colunas_chave)
    dados_alimentos[colunas_numericas] = dados_alimentos[colunas_numericas].apply(pd.to_numeric, errors='raise')

    return dados_nutrientes, dados_alimentos


def resolver_problema_dieta_scipy(dados_nutrientes, dados_alimentos):
    """
    Resolve o Problema da Dieta usando scipy.optimize.linprog.
    O problema é transformado para a forma padrão: Min(c^T * x) sujeito a A_ub * x <= b_ub e x >= 0.
    """
    
    # 1. Parâmetros do Problema
    c = dados_alimentos['preco'].to_numpy() # c (Objetivo: Preços)
    nomes_nutrientes = dados_nutrientes['nutrientes'].tolist()
    nomes_alimentos = dados_alimentos['ingrediente'].tolist()

    # 2. Restrições (A*x >= b) -> (-A)*x <= (-b)
    
    # b_ub (Lado Direito da Restrição) -> Mínimos Nutricionais (NEGATIVADOS)
    b_ub = -dados_nutrientes['minimo'].to_numpy()
    
    # A_ub (Matriz de Coeficientes) -> Conteúdo Nutricional (NEGATIVADO)
    A_ub = -dados_alimentos[nomes_nutrientes].to_numpy().T

    # 3. Solução
    resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')

    # 4. Exibir a Solução
    if resultado.success:
        output = "\n--- SOLUÇÃO ÓTIMA ENCONTRADA (usando SciPy) ---\n"
        output += f"Custo Mínimo Total: R$ {resultado.fun:.4f}\n"
        output += "\nQuantidades de Alimentos a Comprar (Unidades de Compra):\n"
        
        melhor_solucao = []
        for i, valor in enumerate(resultado.x):
            if valor > 1e-6: # Filtra quantidades insignificantes
                nome_alimento = nomes_alimentos[i]
                unidade = dados_alimentos.loc[dados_alimentos['ingrediente'] == nome_alimento, 'quantidade'].iloc[0]
                
                melhor_solucao.append({
                    'Ingrediente': nome_alimento, 
                    'Quantidade Necessária': f"{valor:.6f}",
                    'Unidade de Compra (Referência)': unidade
                })
        
        df_solucao = pd.DataFrame(melhor_solucao)
        output += df_solucao.to_string(index=False)
        
        # Verificação de Nutrientes (A * x)
        consumido = np.dot(A_ub * -1, resultado.x)
        minimo = b_ub * -1
        
        output += "\n\nVerificação de Nutrientes (Consumido vs Mínimo):\n"
        for i, nome_nutriente in enumerate(nomes_nutrientes):
            atendido = "OK" if consumido[i] >= minimo[i] - 1e-6 else "FALHA"
            output += f"- {nome_nutriente:<15}: Consumido={consumido[i]:.4f}, Mínimo={minimo[i]:.1f} ({atendido})\n"
        
        print(output)
            
    else:
        if resultado.status == 2:
            print("\nO PROBLEMA É INVIÁVEL. Não há combinação de alimentos que satisfaça todas as restrições mínimas de nutrientes.")
        else:
            print(f"\nO problema não possui solução ótima. Status do solver: {resultado.status}\n{resultado.message}")

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == '__main__':
    try:
        dados_nutrientes, dados_alimentos = carregar_dados_csv()
        resolver_problema_dieta_scipy(dados_nutrientes, dados_alimentos)
    except Exception as e:
        print(f"\nOcorreu um erro: {e}")
        print("\nERRO CRÍTICO: Verifique se os arquivos CSV estão salvos na mesma pasta e se a primeira linha está correta.")