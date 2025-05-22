#projeto grafos
import networkx as nx
import matplotlib.pyplot as plt
import io, sys
from networkx.algorithms.community import greedy_modularity_communities


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
grafo = nx.DiGraph()

# NÃO ESQUECER DE MUDAR O DIRETÓRIO 
with open(r"caminho do arquivo", 'r', encoding='utf-8') as arquivo:
    
    for atividade in arquivo:
        tweet, retweet, peso  = atividade.strip().split()
        peso = int (peso)
        grafo.add_edge(tweet, retweet, weight=peso) #ADICIONANDO NÓS, ARESTAS E PESOS AO GRAFO


def mostrar_estatisticas(grafo):
    print(f"Número de nós: {grafo.number_of_nodes()}")
    print(f"Número de arestas: {grafo.number_of_edges()}")

#centralidade e PageRank
def exibir_top(metricas, titulo, n=5):
    top = sorted(metricas.items(), key=lambda x: x[1], reverse=True)[:n]
    print(f"\n{titulo}")
    for no, valor in top:
        print(f"{no}: {valor:.5f}")

def analisar_centralidades(grafo):
    exibir_top(nx.pagerank(grafo, alpha=0.85), "Top 5 PageRank")
    exibir_top(nx.degree_centrality(grafo), "Top 5 Grau de Centralidade")
    exibir_top(nx.betweenness_centrality(grafo), "Top 5 Intermediação")
    exibir_top(nx.closeness_centrality(grafo), "Top 5 Proximidade")


# Detecta de comunidades 
def detectar_comunidades(grafo, top_n=5):
    comunidades = list(greedy_modularity_communities(grafo.to_undirected()))
    print(f"\nTotal de comunidades detectadas: {len(comunidades)}")
    top_comunidades = sorted(comunidades, key=len, reverse=True)[:top_n]
    for i, comunidade in enumerate(top_comunidades):
        print(f"Comunidade {i+1}: {len(comunidade)} nós")

def visualizar_subgrafo(grafo, tamanho=100):
    sub = grafo.subgraph(list(grafo.nodes())[:tamanho])
    pos = nx.spring_layout(sub, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(sub, pos, node_size=30, arrows=True, with_labels=False)
    plt.title("Subgrafo (amostra)")
    plt.show()

# Plota distribuição dos graus
def plotar_distribuicao_graus(grafo):
    graus = [grafo.degree(n) for n in grafo.nodes()]
    plt.figure(figsize=(10, 6))
    plt.hist(graus, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribuição de Graus na Rede")
    plt.xlabel("Grau")
    plt.ylabel("Número de Nós")
    plt.grid(True)
    plt.show()

# Exportar para Gephi
# tem que fazer, e tem que verificar se os gráficos estão sendo gerados corretamente, fiz um arquivo copia com menos dados 



mostrar_estatisticas(grafo)
analisar_centralidades(grafo)
detectar_comunidades(grafo)
plotar_distribuicao_graus(grafo)
visualizar_subgrafo(grafo)
#exportar_gephi(grafo)














