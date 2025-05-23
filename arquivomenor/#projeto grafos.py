# Projeto de Análise de Redes Sociais para Detecção de Notícias Falsas
import networkx as nx
import matplotlib.pyplot as plt
import io, sys
import os
from networkx.algorithms.community import greedy_modularity_communities

# Configuração para exibição correta de caracteres
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def carregar_grafo(caminho_arquivo):
    """
    Carrega os dados do arquivo e constrói o grafo direcionado.
    
    Parâmetros:
    caminho_arquivo (str): Caminho para o arquivo de dados
    
    Retorna:
    nx.DiGraph: Grafo direcionado construído a partir dos dados
    """
    print(f"Carregando dados do arquivo: {caminho_arquivo}")
    grafo = nx.DiGraph()
    
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
            interacoes_processadas = 0
            
            for linha in arquivo:
                try:
                    tweet, retweet, peso = linha.strip().split()
                    peso = int(peso)
                    grafo.add_edge(tweet, retweet, weight=peso)
                    interacoes_processadas += 1
                except ValueError:
                    print(f"AVISO: Formato incorreto na linha: {linha.strip()}")
                    continue
            
            print(f"Dados carregados com sucesso: {interacoes_processadas} interações processadas")
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado: {caminho_arquivo}")
        print("Verifique o caminho do arquivo e tente novamente.")
        sys.exit(1)
        
    return grafo

def mostrar_estatisticas(grafo):
    """
    Exibe estatísticas básicas do grafo.
    """
    print("\n=== ESTATÍSTICAS BÁSICAS DO GRAFO ===")
    print(f"Número de nós (usuários): {grafo.number_of_nodes()}")
    print(f"Número de arestas (interações): {grafo.number_of_edges()}")
    
    # Estatísticas adicionais
    densidade = nx.density(grafo)
    print(f"Densidade do grafo: {densidade:.6f}")
    
    grau_medio = sum(dict(grafo.degree()).values()) / grafo.number_of_nodes()
    print(f"Grau médio: {grau_medio:.2f}")
    
    # Analisar componentes conectados
    grafo_und = grafo.to_undirected()
    componentes = list(nx.connected_components(grafo_und))
    print(f"Número de componentes conectados: {len(componentes)}")
    maior_componente = max(componentes, key=len)
    print(f"Tamanho do maior componente: {len(maior_componente)} nós ({len(maior_componente)/grafo.number_of_nodes()*100:.1f}% do grafo)")

def exibir_top(metricas, titulo, n=5):
    """
    Exibe os n principais nós segundo uma métrica específica.
    """
    top = sorted(metricas.items(), key=lambda x: x[1], reverse=True)[:n]
    print(f"\n{titulo}")
    for no, valor in top:
        print(f"{no}: {valor:.5f}")

def analisar_centralidades(grafo):
    """
    Analisa diferentes medidas de centralidade para identificar nós importantes.
    """
    print("\n=== ANÁLISE DE CENTRALIDADES ===")
    print("Calculando métricas de centralidade...")
    
    exibir_top(nx.pagerank(grafo, alpha=0.85), "Top 5 PageRank (Usuários mais influentes)")
    exibir_top(nx.degree_centrality(grafo), "Top 5 Grau de Centralidade (Usuários com mais conexões)")
    exibir_top(nx.betweenness_centrality(grafo), "Top 5 Intermediação (Usuários que conectam comunidades)")
    exibir_top(nx.closeness_centrality(grafo), "Top 5 Proximidade (Usuários que espalham informação rapidamente)")

def detectar_comunidades(grafo, top_n=5):
    """
    Detecta comunidades no grafo usando algoritmo de modularidade.
    """
    print("\n=== DETECÇÃO DE COMUNIDADES ===")
    print("Detectando comunidades na rede...")
    
    # Converter para grafo não direcionado para detecção de comunidades
    grafo_und = grafo.to_undirected()
    
    # Usar algoritmo baseado em modularidade (similar ao Louvain)
    comunidades = list(greedy_modularity_communities(grafo_und))
    print(f"\nTotal de comunidades detectadas: {len(comunidades)}")
    
    # Exibir as maiores comunidades
    top_comunidades = sorted(comunidades, key=len, reverse=True)[:top_n]
    for i, comunidade in enumerate(top_comunidades):
        print(f"Comunidade {i+1}: {len(comunidade)} nós ({len(comunidade)/grafo_und.number_of_nodes()*100:.1f}% do grafo)")
    
    return comunidades

def visualizar_subgrafo(grafo, tamanho=100):
    """
    Visualiza uma amostra do grafo para análise visual.
    """
    print("\n=== VISUALIZAÇÃO DE AMOSTRA DO GRAFO ===")
    print(f"Gerando visualização com {tamanho} nós...")
    
    sub = grafo.subgraph(list(grafo.nodes())[:tamanho])
    pos = nx.spring_layout(sub, seed=42)
    
    plt.figure(figsize=(12, 8))
    nx.draw(sub, pos, 
            node_size=30, 
            arrows=True, 
            with_labels=False,
            node_color='skyblue',
            edge_color='gray',
            alpha=0.8)
    
    plt.title("Visualização de Subgrafo (amostra)")
    plt.savefig('subgrafo_amostra.png', dpi=300, bbox_inches='tight')
    plt.show()

def plotar_distribuicao_graus(grafo):
    """
    Plota a distribuição de graus da rede para análise de estrutura.
    """
    print("\n=== ANÁLISE DA DISTRIBUIÇÃO DE GRAUS ===")
    print("Gerando gráfico de distribuição de graus...")
    
    graus = [grafo.degree(n) for n in grafo.nodes()]
    plt.figure(figsize=(10, 6))
    plt.hist(graus, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribuição de Graus na Rede")
    plt.xlabel("Grau")
    plt.ylabel("Número de Nós")
    plt.grid(True)
    plt.savefig('distribuicao_graus.png', dpi=300, bbox_inches='tight')
    plt.show()

def exportar_gephi(grafo, nome_arquivo="twitter_grafo.gexf"):
    """
    Exporta o grafo para formato compatível com Gephi,
    incluindo as principais métricas como atributos.
    """
    print("\n=== EXPORTAÇÃO PARA GEPHI ===")
    print("Preparando grafo para exportação ao Gephi...")
    
    # Adicionar PageRank como atributo
    print("Calculando PageRank...")
    pagerank = nx.pagerank(grafo, alpha=0.85)
    nx.set_node_attributes(grafo, pagerank, 'pagerank')
    
    # Adicionar grau de centralidade
    print("Calculando grau de centralidade...")
    degree_cent = nx.degree_centrality(grafo)
    nx.set_node_attributes(grafo, degree_cent, 'degree_centrality')
    
    # Adicionar intermediação
    print("Calculando intermediação...")
    betweenness = nx.betweenness_centrality(grafo)
    nx.set_node_attributes(grafo, betweenness, 'betweenness')
    
    # Adicionar proximidade
    print("Calculando proximidade...")
    closeness = nx.closeness_centrality(grafo)
    nx.set_node_attributes(grafo, closeness, 'closeness')
    
    # Adicionar o grau de entrada e saída como atributos
    nx.set_node_attributes(grafo, dict(grafo.in_degree()), 'in_degree')
    nx.set_node_attributes(grafo, dict(grafo.out_degree()), 'out_degree')
    
    # Exportar para GEXF (formato Gephi)
    print(f"Exportando grafo para {nome_arquivo}...")
    nx.write_gexf(grafo, nome_arquivo)
    print(f"Grafo exportado com sucesso para '{nome_arquivo}'!")
    print("Você pode abrir este arquivo no Gephi para visualização avançada e análise.")

def visualizar_comunidades(grafo, comunidades=None, max_nodes=300):
    """
    Visualiza as comunidades detectadas no grafo com cores diferentes.
    Limita a visualização a max_nodes para melhor desempenho.
    """
    print("\n=== VISUALIZAÇÃO DE COMUNIDADES ===")
    print("Visualizando comunidades detectadas...")
    
    # Converter para grafo não direcionado para detecção de comunidade
    grafo_und = grafo.to_undirected()
    
    # Limitar o tamanho do grafo para visualização
    if grafo_und.number_of_nodes() > max_nodes:
        print(f"Limitando visualização a {max_nodes} nós para melhor desempenho")
        nos = list(grafo_und.nodes())[:max_nodes]
        grafo_und = grafo_und.subgraph(nos)
    
    # Detectar comunidades se não fornecidas
    if comunidades is None:
        comunidades = list(greedy_modularity_communities(grafo_und))
    else:
        # Filtrar comunidades para os nós presentes no subgrafo
        comunidades = [comunidade.intersection(set(grafo_und.nodes())) for comunidade in comunidades]
        comunidades = [com for com in comunidades if len(com) > 0]
    
    print(f"Detectadas {len(comunidades)} comunidades no subgrafo")
    
    # Criar mapeamento de nós para comunidades
    node_community = {}
    for i, comm in enumerate(comunidades):
        for node in comm:
            if node in grafo_und.nodes():
                node_community[node] = i
    
    # Obter cores para nós baseadas na comunidade
    color_map = [node_community.get(node, 0) for node in grafo_und.nodes()]
    
    # Calcular posições
    pos = nx.spring_layout(grafo_und, seed=42)
    
    # Criar visualização
    plt.figure(figsize=(14, 10))
    nx.draw_networkx(
        grafo_und,
        pos=pos,
        node_color=color_map,
        node_size=50,
        with_labels=False,
        edge_color='gray',
        width=0.3,
        alpha=0.8,
        cmap=plt.cm.rainbow
    )
    
    plt.title(f"Comunidades Detectadas (Amostra de {grafo_und.number_of_nodes()} nós)")
    plt.axis('off')
    plt.savefig('comunidades_detectadas.png', dpi=300, bbox_inches='tight')
    plt.show()

def identificar_potenciais_espalhadores(grafo, comunidades=None, top_n=10):
    """
    Identifica os potenciais espalhadores de notícias falsas
    usando uma combinação de métricas.
    """
    print("\n=== ANÁLISE DE POTENCIAIS ESPALHADORES DE NOTÍCIAS FALSAS ===")
    
    # Calcular métricas relevantes
    print("Calculando métricas para identificação de espalhadores...")
    pagerank = nx.pagerank(grafo, alpha=0.85)
    degree_cent = nx.degree_centrality(grafo)
    betweenness = nx.betweenness_centrality(grafo)
    closeness = nx.closeness_centrality(grafo)
    
    # Calcular um score composto (média normalizada das métricas)
    scores = {}
    for node in grafo.nodes():
        scores[node] = (
            pagerank.get(node, 0) +
            degree_cent.get(node, 0) +
            betweenness.get(node, 0) +
            closeness.get(node, 0)
        ) / 4.0
    
    # Identificar os top espalhadores
    top_spreaders = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"\nTop {top_n} potenciais espalhadores de notícias falsas:")
    print("-" * 80)
    print(f"{'Usuário':<15} | {'Score':<10} | {'PageRank':<10} | {'Grau':<10} | {'Intermediação':<10} | {'Proximidade':<10}")
    print("-" * 80)
    
    for user, score in top_spreaders:
        print(f"{user:<15} | {score:.6f} | {pagerank[user]:.6f} | {degree_cent[user]:.6f} | {betweenness[user]:.6f} | {closeness[user]:.6f}")
    
    # Analisar os espalhadores em relação às comunidades
    if comunidades is None:
        grafo_und = grafo.to_undirected()
        comunidades = list(greedy_modularity_communities(grafo_und))
    
    print("\nDistribuição dos espalhadores por comunidades:")
    comunidade_count = {}
    
    for i, comunidade in enumerate(comunidades):
        count = sum(1 for user, _ in top_spreaders if user in comunidade)
        if count > 0:
            comunidade_count[i] = count
    
    for com_id, count in sorted(comunidade_count.items(), key=lambda x: x[1], reverse=True):
        print(f"Comunidade {com_id+1}: {count} espalhadores ({count/top_n*100:.1f}%)")
    
    return top_spreaders, scores

def analisar_metricas_adicionais(grafo):
    """
    Realiza análises adicionais sobre o grafo.
    """
    print("\n=== ANÁLISES ADICIONAIS DA REDE ===")
    
    # Calcular reciprocidade (apenas para grafos direcionados)
    reciprocidade = nx.reciprocity(grafo)
    print(f"Reciprocidade (interações mútuas): {reciprocidade:.4f}")
    
    # Coeficiente de clustering (transitividade)
    grafo_und = grafo.to_undirected()
    clustering = nx.average_clustering(grafo_und)
    print(f"Coeficiente de clustering médio: {clustering:.4f}")
    
    # Tentar calcular o diâmetro do maior componente
    try:
        componentes = list(nx.connected_components(grafo_und))
        maior_componente = max(componentes, key=len)
        subgrafo = grafo_und.subgraph(maior_componente)
        
        # Para grafos grandes, isso pode demorar muito
        if len(maior_componente) < 1000:
            diametro = nx.diameter(subgrafo)
            print(f"Diâmetro do maior componente conectado: {diametro}")
        else:
            print("Cálculo do diâmetro ignorado (grafo muito grande)")
    except:
        print("Não foi possível calcular o diâmetro (possivelmente grafo desconectado)")

def main():
    """
    Função principal para executar a análise completa.
    """
    print("\n================================================")
    print("  ANÁLISE DE REDES SOCIAIS PARA DETECÇÃO DE FAKE NEWS")
    print("================================================\n")
    
    # Definir caminho do arquivo
    # Altere este caminho para o local correto do seu arquivo de dados
    caminho_arquivo = r"C:\Users\gonca\OneDrive\Documentos\GitHub\projeto-grafos\arquivomenor\amostra_rapida"
    
    # Verificar se o arquivo existe
    if not os.path.exists(caminho_arquivo):
        caminho_absoluto = os.path.abspath(caminho_arquivo)
        print(f"AVISO: Arquivo não encontrado no caminho relativo.")
        print(f"Caminho tentado: {caminho_absoluto}")
        caminho_arquivo = input("Por favor, insira o caminho correto para o arquivo de dados: ")
    
    # Carregar grafo
    grafo = carregar_grafo(caminho_arquivo)
    
    # Mostrar estatísticas básicas
    mostrar_estatisticas(grafo)
    
    # Análise de centralidades (PageRank e outras métricas)
    analisar_centralidades(grafo)
    
    # Detecção de comunidades
    comunidades = detectar_comunidades(grafo)
    
    # Identificar potenciais espalhadores
    top_spreaders, scores = identificar_potenciais_espalhadores(grafo, comunidades)
    
    # Análises adicionais
    analisar_metricas_adicionais(grafo)
    
    # Visualizações
    print("\n=== GERANDO VISUALIZAÇÕES ===")
    plotar_distribuicao_graus(grafo)
    visualizar_subgrafo(grafo)
    visualizar_comunidades(grafo, comunidades)
    
    # Exportar para Gephi
    exportar_gephi(grafo)
    
    print("\n================================================")
    print("  ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("================================================")
    print("\nArquivos gerados:")
    print("- distribuicao_graus.png: Gráfico da distribuição de graus")
    print("- subgrafo_amostra.png: Visualização de uma amostra do grafo")
    print("- comunidades_detectadas.png: Visualização das comunidades")
    print("- twitter_grafo.gexf: Arquivo para análise no Gephi")

# Executar o programa
if __name__ == "__main__":
    main()
    