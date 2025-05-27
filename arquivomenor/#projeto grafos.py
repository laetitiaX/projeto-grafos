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
    """
    print(f"Carregando dados do arquivo: {caminho_arquivo}")
    grafo = nx.DiGraph()
    
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
            interacoes_processadas = 0
            for linha in arquivo:
                try:
                    partes = linha.strip().split()
                    if len(partes) >= 3:
                        tweet = partes[0]
                        retweet = partes[1]
                        peso = int(partes[2])
                        grafo.add_edge(tweet, retweet, weight=peso)
                        interacoes_processadas += 1
                    elif len(partes) == 2:
                        tweet = partes[0]
                        retweet = partes[1]
                        grafo.add_edge(tweet, retweet, weight=1)
                        interacoes_processadas += 1
                    else:
                        print(f"AVISO: Formato incorreto na linha (poucas colunas): {linha.strip()}")
                        continue
                except ValueError:
                    print(f"AVISO: Formato incorreto na linha (erro de valor): {linha.strip()}")
                    continue
            print(f"Dados carregados com sucesso: {interacoes_processadas} interações processadas")
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado: {caminho_arquivo}")
        sys.exit(1)
    return grafo

def mostrar_estatisticas(grafo):
    """
    Exibe estatísticas básicas do grafo.
    """
    print("\n=== ESTATÍSTICAS BÁSICAS DO GRAFO ===")
    print(f"Número de nós (usuários): {grafo.number_of_nodes()}")
    print(f"Número de arestas (interações): {grafo.number_of_edges()}")
    if grafo.number_of_nodes() > 0:
        densidade = nx.density(grafo)
        print(f"Densidade do grafo: {densidade:.6f}")
        grau_medio = sum(dict(grafo.degree()).values()) / grafo.number_of_nodes()
        print(f"Grau médio: {grau_medio:.2f}")
        if grafo.is_directed():
            print(f"Número de componentes fracamente conectados: {nx.number_weakly_connected_components(grafo)}")
            componentes_fortes = list(nx.strongly_connected_components(grafo))
            print(f"Número de componentes fortemente conectados: {len(componentes_fortes)}")
            if componentes_fortes:
                maior_componente_forte = max(componentes_fortes, key=len)
                print(f"Tamanho do maior componente fortemente conectado: {len(maior_componente_forte)} nós ({len(maior_componente_forte)/grafo.number_of_nodes()*100:.1f}% do grafo)")
        else:
            componentes = list(nx.connected_components(grafo))
            print(f"Número de componentes conectados: {len(componentes)}")
            if componentes:
                maior_componente = max(componentes, key=len)
                print(f"Tamanho do maior componente: {len(maior_componente)} nós ({len(maior_componente)/grafo.number_of_nodes()*100:.1f}% do grafo)")
    else:
        print("Grafo vazio, não é possível calcular estatísticas detalhadas.")

def exibir_top(metricas, titulo, n=5):
    """
    Exibe os n principais nós segundo uma métrica específica.
    """
    if isinstance(metricas, dict):
        itens_ordenados = sorted(metricas.items(), key=lambda x: x[1], reverse=True)
    else:
        itens_ordenados = sorted(list(metricas), key=lambda x: x[1], reverse=True)
    top = itens_ordenados[:n]
    print(f"\n{titulo}")
    for no, valor in top:
        print(f"{no}: {valor:.5f}")

def analisar_centralidades(grafo):
    """
    Analisa diferentes medidas de centralidade para identificar nós importantes.
    """
    print("\n=== ANÁLISE DE CENTRALIDADES ===")
    if grafo.number_of_nodes() == 0:
        print("Grafo vazio. Não é possível calcular centralidades.")
        return
    print("Calculando métricas de centralidade (para exibição top 5)...")
    try:
        exibir_top(nx.pagerank(grafo, alpha=0.85), "Top 5 PageRank (Usuários mais influentes)")
    except Exception as e: print(f"Erro ao calcular PageRank: {e}")
    try:    
        exibir_top(nx.degree_centrality(grafo), "Top 5 Grau de Centralidade (Usuários com mais conexões)")
    except Exception as e: print(f"Erro ao calcular Grau de Centralidade: {e}")

    num_nos_para_centralidades_caras = 2000
    if grafo.number_of_nodes() < num_nos_para_centralidades_caras:
        try:
            exibir_top(nx.betweenness_centrality(grafo), "Top 5 Intermediação (Usuários que conectam comunidades)")
        except Exception as e: print(f"Erro ao calcular Intermediação: {e}")
        try:
            is_connected_check = False
            if grafo.is_directed():
                if nx.is_weakly_connected(grafo): is_connected_check = True
            elif nx.is_connected(grafo): is_connected_check = True
            
            if is_connected_check:
                 exibir_top(nx.closeness_centrality(grafo), "Top 5 Proximidade (Usuários que espalham informação rapidamente)")
            else:
                print("Grafo não é conectado (ou fracamente conectado), pulando Closeness Centrality no grafo inteiro para top 5.")
        except Exception as e: print(f"Erro ao calcular Proximidade: {e}")
    else:
        print(f"Grafo com {grafo.number_of_nodes()} nós. Pulando Betweenness e Closeness para exibição top 5 (demorado).")

def detectar_comunidades(grafo, top_n=5):
    """
    Detecta comunidades no grafo usando algoritmo de modularidade.
    """
    print("\n=== DETECÇÃO DE COMUNIDADES ===")
    if grafo.number_of_nodes() == 0: return []
    print("Detectando comunidades na rede...")
    grafo_und = grafo.to_undirected() if grafo.is_directed() else grafo
    if grafo_und.number_of_nodes() == 0: return []
    try:
        comunidades_gerador = greedy_modularity_communities(grafo_und)
        comunidades = [set(c) for c in comunidades_gerador]
        print(f"\nTotal de comunidades detectadas: {len(comunidades)}")
        comunidades_ordenadas = sorted(comunidades, key=len, reverse=True)
        for i, comunidade_set in enumerate(comunidades_ordenadas[:top_n]):
            print(f"Comunidade {i+1}: {len(comunidade_set)} nós ({len(comunidade_set)/grafo_und.number_of_nodes()*100:.1f}% do grafo)")
        return comunidades_ordenadas
    except Exception as e:
        print(f"Erro ao detectar comunidades: {e}")
        return []

def visualizar_subgrafo(grafo, tamanho=100):
    """
    Visualiza uma amostra do grafo para análise visual.
    """
    print("\n=== VISUALIZAÇÃO DE AMOSTRA DO GRAFO ===")
    if grafo.number_of_nodes() == 0: return
    num_nos_plotar = min(tamanho, grafo.number_of_nodes())
    print(f"Gerando visualização com {num_nos_plotar} nós...")
    nos_subgrafo = list(grafo.nodes())[:num_nos_plotar]
    sub = grafo.subgraph(nos_subgrafo)
    if sub.number_of_nodes() == 0: return
    try:
        pos = nx.spring_layout(sub, seed=42)
        plt.figure(figsize=(12, 8))
        nx.draw(sub, pos, node_size=30, arrows=True, with_labels=False, node_color='skyblue', edge_color='gray', alpha=0.8)
        plt.title(f"Visualização de Subgrafo ({num_nos_plotar} nós da amostra)")
        plt.savefig('subgrafo_amostra.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e: print(f"Erro ao visualizar subgrafo: {e}")

def plotar_distribuicao_graus(grafo):
    """
    Plota a distribuição de graus da rede para análise de estrutura.
    """
    print("\n=== ANÁLISE DA DISTRIBUIÇÃO DE GRAUS ===")
    if grafo.number_of_nodes() == 0: return
    print("Gerando gráfico de distribuição de graus...")
    graus = [grafo.degree(n) for n in grafo.nodes()]
    if not graus: return
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(graus, bins=max(30, min(len(set(graus)), 100)), color='skyblue', edgecolor='black')
        plt.title("Distribuição de Graus na Rede")
        plt.xlabel("Grau")
        plt.ylabel("Número de Nós")
        plt.yscale('log', nonpositive='clip')
        plt.xscale('log', nonpositive='clip')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('distribuicao_graus.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e: print(f"Erro ao plotar distribuição de graus: {e}")

def visualizar_metricas_nos(grafo, valores_metrica, titulo_metrica, nome_arquivo, max_nodes_vis=200, k_layout=0.5, seed=42):
    """
    Visualiza o grafo com tamanho/cor dos nós proporcional aos valores de uma métrica.
    Foca nos nós com maiores valores da métrica se o grafo for grande.
    """
    print(f"\n=== VISUALIZAÇÃO: {titulo_metrica} ===")
    if not valores_metrica:
        print(f"Valores da métrica '{titulo_metrica}' estão vazios ou não foram fornecidos. Pulando visualização.")
        return
    if grafo.number_of_nodes() == 0:
        print("Grafo vazio. Pulando visualização de métrica.")
        return

    subgrafo_para_vis = grafo
    nos_para_plotar = list(grafo.nodes())

    if grafo.number_of_nodes() > max_nodes_vis:
        print(f"Grafo original com {grafo.number_of_nodes()} nós. Selecionando os {max_nodes_vis} nós com maiores valores da métrica para visualização.")
        nos_ordenados_pela_metrica = sorted(
            filter(lambda item: item[0] in grafo, valores_metrica.items()),
            key=lambda item: item[1],
            reverse=True
        )
        nos_para_plotar = [no for no, val in nos_ordenados_pela_metrica[:max_nodes_vis]]
        if not nos_para_plotar and grafo.number_of_nodes() > 0:
            print(f"Nenhum nó selecionado por métrica '{titulo_metrica}'. Fallback: Pegando os primeiros {max_nodes_vis} nós.")
            nos_para_plotar = list(grafo.nodes())[:max_nodes_vis]
        
        subgrafo_para_vis = grafo.subgraph(nos_para_plotar)
        if not subgrafo_para_vis.nodes():
             print("Subgrafo para visualização de métrica está vazio após filtragem. Pulando.")
             return
    
    print(f"Visualizando '{titulo_metrica}' em {subgrafo_para_vis.number_of_nodes()} nós e {subgrafo_para_vis.number_of_edges()} arestas.")

    try:
        pos = nx.spring_layout(subgrafo_para_vis, k=k_layout, iterations=30, seed=seed)
        
        metric_values_subgraph_dict = {node: valores_metrica.get(node, 0) for node in subgrafo_para_vis.nodes()}
        metric_values_list = list(metric_values_subgraph_dict.values())

        node_sizes = 200 
        node_colors = 'skyblue' 
        min_val, max_val = (0,0)

        if metric_values_list:
            min_val = min(metric_values_list)
            max_val = max(metric_values_list)
            if max_val > min_val:
                node_sizes = [50 + 1950 * ((metric_values_subgraph_dict.get(node, min_val) - min_val) / (max_val - min_val)) for node in subgrafo_para_vis.nodes()]
            else:
                node_sizes = [500 for _ in subgrafo_para_vis.nodes()]
            node_colors = [metric_values_subgraph_dict.get(node, 0) for node in subgrafo_para_vis.nodes()]
        else:
            print(f"Nenhum valor da métrica '{titulo_metrica}' encontrado para os nós do subgrafo. Usando tamanho/cor padrão.")

        plt.figure(figsize=(16, 12))
        nx.draw_networkx(subgrafo_para_vis, pos,
                         node_size=node_sizes,
                         node_color=node_colors,
                         cmap=plt.cm.viridis,
                         with_labels=False,
                         width=0.15,
                         alpha=0.7,
                         edge_color='grey')
        
        plt.title(titulo_metrica, fontsize=16)
        if metric_values_list and max_val > min_val: 
            norm = plt.Normalize(vmin=min_val, vmax=max_val)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
            sm.set_array([])
            # LINHA CORRIGIDA ABAIXO
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.6, aspect=15, pad=0.02) 
            cbar.set_label(titulo_metrica.split('(')[0].strip(), rotation=270, labelpad=15)

        plt.axis('off')
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Visualização '{titulo_metrica}' salva como {nome_arquivo}")
    except Exception as e:
        print(f"Erro ao visualizar métrica '{titulo_metrica}': {e}")

def exportar_gephi(grafo, comunidades_detectadas=None, nome_arquivo="twitter_grafo.gexf"):
    """
    Exporta o grafo para formato compatível com Gephi.
    """
    print("\n=== EXPORTAÇÃO PARA GEPHI ===")
    if grafo.number_of_nodes() == 0: return
    print("Preparando grafo para exportação ao Gephi...")
    grafo_para_exportar = grafo.copy()

    try:
        pagerank_scores = nx.pagerank(grafo_para_exportar, alpha=0.85)
        nx.set_node_attributes(grafo_para_exportar, pagerank_scores, 'pagerank')
    except Exception as e: print(f"Erro ao definir PageRank para Gephi: {e}")
    try:
        degree_centrality_scores = nx.degree_centrality(grafo_para_exportar)
        nx.set_node_attributes(grafo_para_exportar, degree_centrality_scores, 'degree_centrality')
    except Exception as e: print(f"Erro ao definir Grau de Centralidade para Gephi: {e}")

    num_nos_para_centralidades_caras = 2000
    if grafo_para_exportar.number_of_nodes() < num_nos_para_centralidades_caras:
        try:
            betweenness_centrality_scores = nx.betweenness_centrality(grafo_para_exportar)
            nx.set_node_attributes(grafo_para_exportar, betweenness_centrality_scores, 'betweenness_centrality')
        except Exception as e: print(f"Erro ao definir Intermediação para Gephi: {e}")
        try:
            is_connected_check = False
            if grafo_para_exportar.is_directed():
                if nx.is_weakly_connected(grafo_para_exportar): is_connected_check = True
            elif nx.is_connected(grafo_para_exportar): is_connected_check = True
            if is_connected_check:
                closeness_centrality_scores = nx.closeness_centrality(grafo_para_exportar)
                nx.set_node_attributes(grafo_para_exportar, closeness_centrality_scores, 'closeness_centrality')
            else: print("Grafo não conectado, não definindo Closeness para Gephi.")
        except Exception as e: print(f"Erro ao definir Proximidade para Gephi: {e}")
    else:
        print("Grafo grande, pulando Intermediação e Proximidade na exportação para Gephi.")

    try:
        nx.set_node_attributes(grafo_para_exportar, dict(grafo_para_exportar.in_degree()), 'in_degree')
        nx.set_node_attributes(grafo_para_exportar, dict(grafo_para_exportar.out_degree()), 'out_degree')
        nx.set_node_attributes(grafo_para_exportar, dict(grafo_para_exportar.degree()), 'degree')
    except Exception as e: print(f"Erro ao definir atributos de grau para Gephi: {e}")

    if comunidades_detectadas:
        print("Adicionando IDs de comunidade como atributo...")
        node_to_community_id = {}
        for i, comunidade_set in enumerate(comunidades_detectadas):
            for node in comunidade_set:
                if node in grafo_para_exportar:
                    node_to_community_id[node] = i
        nx.set_node_attributes(grafo_para_exportar, node_to_community_id, 'community_id')
        print(f"{len(node_to_community_id)} nós mapeados para comunidades.")
    try:
        nx.write_gexf(grafo_para_exportar, nome_arquivo)
        print(f"Grafo exportado com sucesso para '{nome_arquivo}'!")
    except Exception as e: print(f"ERRO CRÍTICO ao exportar para Gephi: {e}")


def visualizar_comunidades(grafo, comunidades=None, max_nodes=500): # Usando a versão mais robusta
    """
    Visualiza as comunidades detectadas no grafo com cores diferentes.
    Limita a visualização a max_nodes para melhor desempenho, priorizando nós com maior grau.
    """
    print("\n=== VISUALIZAÇÃO DE COMUNIDADES ===")
    if grafo.number_of_nodes() == 0:
        print("Grafo vazio. Não é possível visualizar comunidades.")
        return

    print("Visualizando comunidades detectadas...")
    grafo_vis = grafo.to_undirected() if grafo.is_directed() else grafo.copy()
    num_nos_original = grafo_vis.number_of_nodes()

    if num_nos_original > max_nodes:
        print(f"Limitando visualização aos {max_nodes} nós com maior grau.")
        # Usar grau do grafo_vis (não direcionado) para seleção
        graus_subgrafo = dict(grafo_vis.degree())
        nos_ordenados_por_grau = sorted(graus_subgrafo.items(), key=lambda item: item[1], reverse=True)
        nos_subgrafo_vis = [no for no, _ in nos_ordenados_por_grau[:max_nodes]]
        grafo_vis = grafo_vis.subgraph(nos_subgrafo_vis)
    else:
        print(f"Visualizando comunidades em {num_nos_original} nós.")

    num_nos_vis = grafo_vis.number_of_nodes()
    num_arestas_vis = grafo_vis.number_of_edges()
    print(f"Subgrafo para visualização tem {num_nos_vis} nós e {num_arestas_vis} arestas.")

    if num_nos_vis == 0:
        print("Subgrafo para visualização de comunidades está vazio.")
        return

    comunidades_para_plotar = []
    node_community_map = {} 

    if comunidades: # comunidades são do grafo original
        temp_map = {}
        for i, com_original in enumerate(comunidades):
            for node in com_original:
                 if node in grafo_vis: # Se o nó da comunidade original está no nosso subgrafo de visualização
                    temp_map[node] = i 
        # Reconstruir comunidades_para_plotar baseado nos nós do subgrafo
        # e nos IDs de comunidade mapeados
        if temp_map:
            unique_com_ids_in_subgraph = sorted(list(set(temp_map.values())))
            id_map_for_subgraph = {orig_id: new_id for new_id, orig_id in enumerate(unique_com_ids_in_subgraph)}
            
            comunidades_para_plotar = [set() for _ in range(len(unique_com_ids_in_subgraph))]
            for node, orig_com_id in temp_map.items():
                new_com_id = id_map_for_subgraph[orig_com_id]
                comunidades_para_plotar[new_com_id].add(node)
                node_community_map[node] = new_com_id # Atualiza o mapa para plotagem
    
    if not comunidades_para_plotar: 
        print("Detectando comunidades diretamente no subgrafo de visualização...")
        try:
            com_gen = greedy_modularity_communities(grafo_vis) # Detecta no subgrafo_vis
            comunidades_para_plotar = [set(c) for c in com_gen if c] # Remove comunidades vazias se houver
            for i, comm_set in enumerate(comunidades_para_plotar):
                for node in comm_set:
                    node_community_map[node] = i
        except Exception as e:
            print(f"Erro ao detectar comunidades para visualização: {e}")
            return 

    num_comunidades_plotar = len(comunidades_para_plotar)
    print(f"Número de comunidades para plotar no subgrafo: {num_comunidades_plotar}")

    if not node_community_map and num_nos_vis > 0 : # Se não há mapa, mas há nós, algo está errado
         print("Nenhum nó mapeado para uma comunidade no subgrafo de visualização. Plotando com cor única.")
         color_map = 'skyblue' # Fallback para cor única
    elif not num_nos_vis: # Grafo de visualização vazio
        return
    else:
        color_map = [node_community_map.get(node, -1) for node in grafo_vis.nodes()]
    
    try:
        pos = nx.spring_layout(grafo_vis, seed=42, k=0.3)
        plt.figure(figsize=(14, 10))
        
        # Determina o número de cores distintas necessárias
        num_distinct_colors = 1
        if isinstance(color_map, list):
            num_distinct_colors = max(1, len(set(color_map)))

        nx.draw_networkx(
            grafo_vis, pos=pos, node_color=color_map, node_size=50,
            with_labels=False, edge_color='gray', width=0.3, alpha=0.8,
            cmap=plt.cm.get_cmap('rainbow', num_distinct_colors)
        )
        plt.title(f"Comunidades Detectadas (Amostra de {num_nos_vis} nós)")
        plt.axis('off')
        plt.savefig('comunidades_detectadas.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Erro ao visualizar comunidades: {e}")


def identificar_potenciais_espalhadores(grafo, comunidades=None, top_n=10):
    """
    Identifica os potenciais espalhadores de notícias falsas.
    """
    print("\n=== ANÁLISE DE POTENCIAIS ESPALHADORES DE NOTÍCIAS FALSAS ===")
    if grafo.number_of_nodes() == 0: return [], {}
        
    print("Calculando métricas para identificação de espalhadores...")
    pagerank, degree_cent, betweenness, closeness = {}, {}, {}, {}
    try: pagerank = nx.pagerank(grafo, alpha=0.85)
    except Exception: pass
    try: degree_cent = nx.degree_centrality(grafo)
    except Exception: pass
    
    num_nos_para_centralidades_caras = 2000
    if grafo.number_of_nodes() < num_nos_para_centralidades_caras:
        try: betweenness = nx.betweenness_centrality(grafo)
        except Exception: pass
        try:
            is_connected_check = False
            if grafo.is_directed():
                if nx.is_weakly_connected(grafo): is_connected_check = True
            elif nx.is_connected(grafo): is_connected_check = True
            if is_connected_check: closeness = nx.closeness_centrality(grafo)
        except Exception: pass
    
    scores = {}
    for node in grafo.nodes():
        s, count = 0, 0
        if node in pagerank: s += pagerank[node]; count +=1
        if node in degree_cent: s += degree_cent[node]; count +=1
        if node in betweenness: s += betweenness[node]; count +=1
        if node in closeness: s += closeness[node]; count +=1
        scores[node] = s / count if count > 0 else 0
    
    top_spreaders = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"\nTop {top_n} potenciais espalhadores (score combinado):")
    print("-" * 100)
    print(f"{'Usuário':<15} | {'Score':<10} | {'PageRank':<10} | {'Grau Cent.':<12} | {'Intermediação':<15} | {'Proximidade':<12}")
    print("-" * 100)
    for user, score_val in top_spreaders:
        print(f"{user:<15} | {score_val:.6f} | {pagerank.get(user, 0):.6f} | {degree_cent.get(user, 0):.6f} | {betweenness.get(user, 0):.6f} | {closeness.get(user, 0):.6f}")
    
    if comunidades:
        print("\nDistribuição dos espalhadores por comunidades:")
        comunidade_count = {}
        node_to_community_id_map = {node: i for i, comm_set in enumerate(comunidades) for node in comm_set}
        for user, _ in top_spreaders:
            if user in node_to_community_id_map:
                comm_id = node_to_community_id_map[user]
                comunidade_count[comm_id] = comunidade_count.get(comm_id, 0) + 1
        for com_id, count in sorted(comunidade_count.items(), key=lambda x: x[1], reverse=True):
            if count > 0: print(f"Comunidade ID {com_id}: {count} espalhadores ({count/len(top_spreaders)*100:.1f}%)")
    return top_spreaders, scores

def analisar_metricas_adicionais(grafo):
    """
    Realiza análises adicionais sobre o grafo.
    """
    print("\n=== ANÁLISES ADICIONAIS DA REDE ===")
    if grafo.number_of_nodes() == 0: return
    try:
        if grafo.is_directed(): print(f"Reciprocidade: {nx.reciprocity(grafo):.4f}")
    except Exception as e: print(f"Erro reciprocidade: {e}")
    try:
        g_und = grafo.to_undirected() if grafo.is_directed() else grafo
        if g_und.number_of_nodes() > 0: print(f"Coeficiente de clustering médio: {nx.average_clustering(g_und):.4f}")
    except Exception as e: print(f"Erro clustering: {e}")
    try:
        comp_gen = nx.weakly_connected_components(grafo) if grafo.is_directed() else nx.connected_components(grafo)
        maior_comp_nos = []
        try: maior_comp_nos = max(comp_gen, key=len)
        except ValueError: pass
        if maior_comp_nos and len(maior_comp_nos) > 1:
            sub_maior = grafo.subgraph(maior_comp_nos)
            if len(maior_comp_nos) < 1000:
                print(f"Calculando diâmetro do maior componente ({len(maior_comp_nos)} nós)...")
                diam = nx.diameter(sub_maior.to_undirected() if sub_maior.is_directed() else sub_maior)
                print(f"Diâmetro do maior componente: {diam}")
            else: print(f"Diâmetro ignorado (maior componente com {len(maior_comp_nos)} nós é grande)")
        else: print("Não há componente grande para calcular diâmetro.")
    except Exception as e: print(f"Erro diâmetro: {e}")

def main():
    print("\n================================================")
    print("   ANÁLISE DE REDES SOCIAIS PARA DETECÇÃO DE FAKE NEWS")
    print("================================================\n")
    
    caminho_arquivo = r"C:\Users\gonca\OneDrive\Documentos\GitHub\projeto-grafos\arquivomenor\amostra_rapida"
    
    if not os.path.exists(caminho_arquivo):
        caminho_absoluto = os.path.abspath(caminho_arquivo)
        print(f"AVISO: Arquivo não encontrado: {caminho_absoluto}")
        caminho_arquivo_input = input("Por favor, insira o caminho correto: ")
        if not os.path.exists(caminho_arquivo_input):
            print(f"ERRO: Arquivo também não encontrado em: {caminho_arquivo_input}. Encerrando.")
            sys.exit(1)
        caminho_arquivo = caminho_arquivo_input
    
    grafo = carregar_grafo(caminho_arquivo)
    if grafo.number_of_nodes() == 0:
        print("Grafo vazio. Encerrando.")
        sys.exit(1)

    mostrar_estatisticas(grafo)
    analisar_centralidades(grafo) 
    comunidades = detectar_comunidades(grafo)
    
    print("\nRecalculando/obtendo métricas para visualização e exportação...")
    pagerank_scores, degree_centrality_scores = {}, {}
    betweenness_centrality_scores, closeness_centrality_scores = {}, {}
    try: pagerank_scores = nx.pagerank(grafo, alpha=0.85)
    except Exception as e: print(f"Falha ao calcular PageRank para visualização: {e}")
    try: degree_centrality_scores = nx.degree_centrality(grafo)
    except Exception as e: print(f"Falha ao calcular Grau de Centralidade para visualização: {e}")

    num_nos_para_centralidades_caras = 2000
    if grafo.number_of_nodes() < num_nos_para_centralidades_caras:
        try: betweenness_centrality_scores = nx.betweenness_centrality(grafo)
        except Exception as e: print(f"Falha ao calcular Intermediação para visualização: {e}")
        try:
            is_connected_check = False
            if grafo.is_directed():
                if nx.is_weakly_connected(grafo): is_connected_check = True
            elif nx.is_connected(grafo): is_connected_check = True
            if is_connected_check: closeness_centrality_scores = nx.closeness_centrality(grafo)
            else: print("Grafo não conectado, Proximidade não será visualizada globalmente.")
        except Exception as e: print(f"Falha ao calcular Proximidade para visualização: {e}")
    else:
        print("Grafo muito grande, Intermediação e Proximidade não serão calculadas/visualizadas para economizar tempo.")

    top_spreaders, combined_scores = identificar_potenciais_espalhadores(grafo, comunidades)
    analisar_metricas_adicionais(grafo)
    
    print("\n=== GERANDO VISUALIZAÇÕES PYTHON ===")
    plotar_distribuicao_graus(grafo)
    visualizar_subgrafo(grafo)
    visualizar_comunidades(grafo, comunidades, max_nodes=500) 

    if pagerank_scores:
        visualizar_metricas_nos(grafo, pagerank_scores, "Visualização de PageRank", "visualizacao_pagerank.png")
    if degree_centrality_scores:
        visualizar_metricas_nos(grafo, degree_centrality_scores, "Visualização de Grau de Centralidade", "visualizacao_grau_centralidade.png")
    if betweenness_centrality_scores:
        visualizar_metricas_nos(grafo, betweenness_centrality_scores, "Visualização de Intermediação", "visualizacao_intermediacao.png")
    if closeness_centrality_scores:
        visualizar_metricas_nos(grafo, closeness_centrality_scores, "Visualização de Proximidade", "visualizacao_proximidade.png")
    
    exportar_gephi(grafo, comunidades_detectadas=comunidades, nome_arquivo="twitter_grafo_com_comunidades.gexf")
    
    print("\n================================================")
    print("   ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("================================================")
    print("\nArquivos gerados (verifique o diretório do script):")
    print("- distribuicao_graus.png")
    print("- subgrafo_amostra.png")
    print("- comunidades_detectadas.png")
    if pagerank_scores: print("- visualizacao_pagerank.png")
    if degree_centrality_scores: print("- visualizacao_grau_centralidade.png")
    if betweenness_centrality_scores: print("- visualizacao_intermediacao.png")
    if closeness_centrality_scores: print("- visualizacao_proximidade.png")
    print("- twitter_grafo_com_comunidades.gexf")

if __name__ == "__main__":
    main()