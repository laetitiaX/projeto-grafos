#projeto grafos

import networkx as nx
import matplotlib.pyplot as plt


grafo_retweet = nx.DiGraph()

# NÃO ESQUECER DE MUDAR O DIRETÓRIO 
with open(r"C:\Users\Letícia Oliveira\Downloads\projetoGrafos\projetoGrafos\higgs-retweet_network.edgelist", 'r') as arquivo:
    for atividade in arquivo:
        tweet, retweet, peso  = atividade.strip().split()
        peso = int (peso)
        grafo_retweet.add_edge(tweet, retweet, weight=peso) #ADICIONANDO NÓS, ARESTAS E PESOS AO GRAFO
        

# TAMANHO DO GRAFO
print("nós:", grafo_retweet.number_of_nodes())
print("arestas:", grafo_retweet.number_of_edges())





#TRECHO DE CÓDIGO PARA DESENHAR SUBGRAFOS (NÃO TEM RELEVÂNCIA PARA O PROJETO, PODE APAGAR!!!)
'''sub_nodes = list(grafo_retweet.nodes)[:100]
subgraph = grafo_retweet.subgraph(sub_nodes)
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(subgraph, seed=42)  # layout organizado
nx.draw(subgraph, pos, with_labels=False, node_size=50, arrows=True)
plt.title("Subgrafo de Retweets (100 nós)")
plt.show()'''






