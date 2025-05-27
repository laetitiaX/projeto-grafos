# Projeto de Análise de Redes Sociais para Detecção de Notícias Falsas
import networkx as nx
import matplotlib.pyplot as plt
import random # Certifique-se que esta linha está presente
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
    # A chamada para gerar_subgrafo_reduzido foi corretamente removida daqui.
    
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
                        # print(f"AVISO: Formato incorreto na linha (poucas colunas): {linha.strip()}") # Descomente se quiser ver avisos
                        continue
                except ValueError:
                    # print(f"AVISO: Formato incorreto na linha (erro de valor): {linha.strip()}") # Descomente se quiser ver avisos
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
            if componentes_fortes: # Adicionado para evitar erro em grafo vazio
                maior_componente_forte = max(componentes_fortes, key=len, default=[]) # Default para evitar erro
                if maior_componente_forte: # Checa se não é lista vazia
                    print(f"Tamanho do maior componente fortemente conectado: {len(maior_componente_forte)} nós ({len(maior_componente_forte)/grafo.number_of_nodes()*100:.1f}% do grafo)")
        else:
            componentes = list(nx.connected_components(grafo))
            print(f"Número de componentes conectados: {len(componentes)}")
            if componentes: # Adicionado para evitar erro em grafo vazio
                maior_componente = max(componentes, key=len, default=[]) # Default para evitar erro
                if maior_componente: # Checa se não é lista vazia
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
    if grafo.number_of_nodes() == 0:
        print("Grafo vazio. Não é possível detectar comunidades.")
        return []
        
    print("Detectando comunidades na rede...")
    print("Método utilizado: greedy_modularity_communities (baseado em modularidade, similar ao Louvain)")
    
    if grafo.is_directed():
        grafo_und = grafo.to_undirected()
    else:
        grafo_und = grafo 
    
    if grafo_und.number_of_nodes() == 0:
        print("Grafo não direcionado resultante está vazio. Não é possível detectar comunidades.")
        return []

    try:
        comunidades_gerador = greedy_modularity_communities(grafo_und)
        comunidades = [set(c) for c in comunidades_gerador if c] 
        
        print(f"\nTotal de comunidades detectadas: {len(comunidades)}")
        comunidades_ordenadas = sorted(comunidades, key=len, reverse=True)
        
        for i, comunidade_set in enumerate(comunidades_ordenadas[:top_n]):
            tamanho_comunidade = len(comunidade_set)
            percentual = (tamanho_comunidade / grafo_und.number_of_nodes() * 100) if grafo_und.number_of_nodes() > 0 else 0
            print(f"Comunidade {i+1}: {tamanho_comunidade} nós ({percentual:.1f}% do grafo)")
        return comunidades_ordenadas
    except Exception as e:
        print(f"Erro ao detectar comunidades: {e}")
        return []

def visualizar_subgrafo(grafo, tamanho=100, seed=42):
    """
    Visualiza uma amostra do grafo, SELECIONANDO OS 'tamanho' NÓS COM MAIOR GRAU
    do grafo fornecido, e mostra espessura das arestas proporcional ao peso.
    """
    print(f"\n=== VISUALIZAÇÃO DE AMOSTRA DO GRAFO (TOP {tamanho} POR GRAU) ===")
    if grafo.number_of_nodes() == 0:
        print("Grafo vazio. Não é possível visualizar subgrafo.")
        return
    
    num_nos_original_do_grafo_passado = grafo.number_of_nodes()
    num_nos_plotar = min(tamanho, num_nos_original_do_grafo_passado)

    if num_nos_plotar == 0:
        print("Nenhum nó para plotar.")
        return

    print(f"Selecionando os {num_nos_plotar} nós com maior grau do grafo fornecido (que tem {num_nos_original_do_grafo_passado} nós) para visualização...")
    
    nos_subgrafo = []
    if num_nos_original_do_grafo_passado > 0 : 
        nos_ordenados_por_grau = sorted(grafo.degree(), key=lambda item: item[1], reverse=True)
        nos_subgrafo = [no for no, grau in nos_ordenados_por_grau[:num_nos_plotar]]
    
    if not nos_subgrafo: 
        if num_nos_original_do_grafo_passado > 0 and num_nos_plotar > 0:
            print(f"Fallback: pegando os primeiros {num_nos_plotar} nós do grafo fornecido.")
            nos_subgrafo = list(grafo.nodes())[:num_nos_plotar]
        else:
            print("Não foi possível selecionar nós para o subgrafo.")
            return

    sub = grafo.subgraph(nos_subgrafo).copy() 
    
    if sub.number_of_nodes() == 0:
        print("Subgrafo resultante para visualização está vazio (após seleção por grau).")
        return
    
    print(f"Visualizando subgrafo com {sub.number_of_nodes()} nós e {sub.number_of_edges()} arestas.")

    try:
        k_val = 0.5 / (sub.number_of_nodes()**0.5) if sub.number_of_nodes() > 1 else 0.5
        pos = nx.spring_layout(sub, seed=seed, k=k_val, iterations=50) 
        
        edge_widths = [1.0] * sub.number_of_edges() 
        edge_attributes = nx.get_edge_attributes(sub, 'weight')

        if edge_attributes:
            weights_list = [edge_attributes.get(edge, 1.0) for edge in sub.edges()] 
            if weights_list: 
                min_w = min(weights_list)
                max_w = max(weights_list)
                if max_w > min_w: 
                    edge_widths = [0.5 + 2.5 * ((w - min_w) / (max_w - min_w)) for w in weights_list]
        
        plt.figure(figsize=(14, 11)) 
        nx.draw(sub, pos, node_size=60, arrows=True, with_labels=False, 
                node_color='orangered', 
                edge_color='darkgrey',  
                width=edge_widths, 
                alpha=0.75,
                connectionstyle='arc3,rad=0.05'
               )
        plt.title(f"Visualização de Subgrafo (Top {sub.number_of_nodes()} Nós por Grau) - Espessura por Frequência", fontsize=14)
        plt.savefig('subgrafo_amostra_top_grau_com_pesos.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e: print(f"Erro ao visualizar subgrafo (top grau) com pesos: {e}")


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
        # plt.yscale('log', nonpositive='clip') # Mantido comentado como na sua última versão
        # plt.xscale('log', nonpositive='clip') # Mantido comentado
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('distribuicao_graus.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e: print(f"Erro ao plotar distribuição de graus: {e}")

def visualizar_metricas_nos(grafo, valores_metrica, titulo_metrica, nome_arquivo,
                            max_nodes_vis=200, k_layout=0.5, seed=42, top_n_labels=5):
    """
    Visualiza o grafo com tamanho/cor dos nós proporcional aos valores de uma métrica.
    Adiciona rótulos aos 'top_n_labels' nós dentro do subgrafo visualizado.
    """
    print(f"\n=== VISUALIZAÇÃO: {titulo_metrica} ===")
    if not valores_metrica:
        print(f"Valores da métrica '{titulo_metrica}' estão vazios. Pulando visualização.")
        return
    if grafo.number_of_nodes() == 0:
        print("Grafo vazio. Pulando visualização de métrica.")
        return

    subgrafo_para_vis = grafo
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

        node_sizes, node_colors = 200, 'skyblue'
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
            print(f"Nenhum valor da métrica '{titulo_metrica}' para os nós do subgrafo. Usando tamanho/cor padrão.")

        labels_to_draw = {}
        if top_n_labels > 0 and subgrafo_para_vis.number_of_nodes() > 0:
            sorted_nodes_in_subgraph = sorted(metric_values_subgraph_dict.items(), key=lambda item: item[1], reverse=True)
            for i, (node, score) in enumerate(sorted_nodes_in_subgraph):
                if i < top_n_labels: labels_to_draw[node] = f"{node}\n({score:.3f})"
                else: break
        
        plt.figure(figsize=(17, 13))
        nx.draw_networkx_nodes(subgrafo_para_vis, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.9)
        nx.draw_networkx_edges(subgrafo_para_vis, pos, width=0.2, alpha=0.4, edge_color='grey')
        if labels_to_draw:
            nx.draw_networkx_labels(subgrafo_para_vis, pos, labels=labels_to_draw, font_size=7, font_color='black',
                                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
        plt.title(titulo_metrica, fontsize=18)
        if metric_values_list and max_val > min_val: 
            norm = plt.Normalize(vmin=min_val, vmax=max_val)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.6, aspect=15, pad=0.02) 
            cbar.set_label(titulo_metrica.split('(')[0].strip(), rotation=270, labelpad=20, fontsize=10)
        plt.axis('off')
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Visualização '{titulo_metrica}' salva como {nome_arquivo}")
    except Exception as e:
        print(f"Erro ao visualizar métrica '{titulo_metrica}': {e}")

def exportar_gephi(grafo, comunidades_detectadas=None, nome_arquivo="twitter_grafo.gexf"):
    print("\n=== EXPORTAÇÃO PARA GEPHI ===")
    if grafo.number_of_nodes() == 0: return
    print("Preparando grafo para exportação ao Gephi...")
    grafo_para_exportar = grafo.copy()
    try:
        nx.set_node_attributes(grafo_para_exportar, nx.pagerank(grafo_para_exportar, alpha=0.85), 'pagerank')
    except Exception as e: print(f"Erro Gephi PageRank: {e}")
    try:
        nx.set_node_attributes(grafo_para_exportar, nx.degree_centrality(grafo_para_exportar), 'degree_centrality')
    except Exception as e: print(f"Erro Gephi GrauCent: {e}")

    num_nos_para_centralidades_caras = 2000
    if grafo_para_exportar.number_of_nodes() < num_nos_para_centralidades_caras:
        try:
            nx.set_node_attributes(grafo_para_exportar, nx.betweenness_centrality(grafo_para_exportar), 'betweenness_centrality')
        except Exception as e: print(f"Erro Gephi Betweenness: {e}")
        try:
            is_connected_check = nx.is_weakly_connected(grafo_para_exportar) if grafo_para_exportar.is_directed() else nx.is_connected(grafo_para_exportar)
            if is_connected_check:
                nx.set_node_attributes(grafo_para_exportar, nx.closeness_centrality(grafo_para_exportar), 'closeness_centrality')
            else: print("Grafo não conectado, não definindo Closeness para Gephi.")
        except Exception as e: print(f"Erro Gephi Closeness: {e}")
    else:
        print(f"Grafo com {grafo_para_exportar.number_of_nodes()} nós. Intermediação e Proximidade puladas na exportação Gephi (demorado).")
    try:
        nx.set_node_attributes(grafo_para_exportar, dict(grafo_para_exportar.in_degree()), 'in_degree')
        nx.set_node_attributes(grafo_para_exportar, dict(grafo_para_exportar.out_degree()), 'out_degree')
        nx.set_node_attributes(grafo_para_exportar, dict(grafo_para_exportar.degree()), 'degree')
    except Exception as e: print(f"Erro Gephi Graus: {e}")
    if comunidades_detectadas:
        node_to_community_id = {node: i for i, com_set in enumerate(comunidades_detectadas) for node in com_set if node in grafo_para_exportar}
        nx.set_node_attributes(grafo_para_exportar, node_to_community_id, 'community_id')
        print(f"{len(node_to_community_id)} nós mapeados para comunidades para Gephi.")
    try:
        nx.write_gexf(grafo_para_exportar, nome_arquivo)
        print(f"Grafo exportado com sucesso para '{nome_arquivo}'!")
    except Exception as e: print(f"ERRO CRÍTICO ao exportar para Gephi: {e}")

def visualizar_comunidades(grafo, comunidades=None, max_nodes=500):
    print("\n=== VISUALIZAÇÃO DE COMUNIDADES ===")
    if grafo.number_of_nodes() == 0: return
    print("Visualizando comunidades detectadas...")
    grafo_vis = grafo.to_undirected() if grafo.is_directed() else grafo.copy()
    num_nos_original = grafo_vis.number_of_nodes()

    if num_nos_original > max_nodes:
        print(f"Limitando visualização aos {max_nodes} nós com maior grau.")
        graus_subgrafo = dict(grafo_vis.degree())
        nos_ordenados_por_grau = sorted(graus_subgrafo.items(), key=lambda item: item[1], reverse=True)
        nos_subgrafo_vis = [no for no, _ in nos_ordenados_por_grau[:max_nodes]]
        grafo_vis = grafo_vis.subgraph(nos_subgrafo_vis)
    else:
        print(f"Visualizando comunidades em {num_nos_original} nós.")

    num_nos_vis, num_arestas_vis = grafo_vis.number_of_nodes(), grafo_vis.number_of_edges()
    print(f"Subgrafo para visualização de comunidades tem {num_nos_vis} nós e {num_arestas_vis} arestas.")
    if num_nos_vis == 0: return

    node_community_map = {}
    if comunidades:
        temp_map = {node: i for i, com_original in enumerate(comunidades) for node in com_original if node in grafo_vis}
        if temp_map:
            unique_com_ids_in_subgraph = sorted(list(set(temp_map.values())))
            id_map_for_subgraph = {orig_id: new_id for new_id, orig_id in enumerate(unique_com_ids_in_subgraph)}
            for node, orig_com_id in temp_map.items():
                node_community_map[node] = id_map_for_subgraph[orig_com_id]
    
    if not node_community_map and num_nos_vis > 0 : 
        print("Detectando comunidades diretamente no subgrafo de visualização (fallback)...")
        try:
            com_gen = greedy_modularity_communities(grafo_vis)
            comunidades_para_plotar_fallback = [set(c) for c in com_gen if c]
            for i, comm_set in enumerate(comunidades_para_plotar_fallback):
                for node in comm_set: node_community_map[node] = i
        except Exception as e: print(f"Erro ao detectar comunidades (fallback): {e}")

    print(f"Número de diferentes IDs de comunidade no subgrafo plotado: {len(set(node_community_map.values())) if node_community_map else 0}")
    color_map = [node_community_map.get(node, -1) for node in grafo_vis.nodes()] if node_community_map else 'skyblue'
    
    try:
        pos = nx.spring_layout(grafo_vis, seed=42, k=0.3)
        plt.figure(figsize=(14, 10))
        num_distinct_colors = max(1, len(set(color_map))) if isinstance(color_map, list) else 1
        nx.draw_networkx(grafo_vis, pos=pos, node_color=color_map, node_size=50,
                         with_labels=False, edge_color='gray', width=0.3, alpha=0.8,
                         cmap=plt.cm.get_cmap('rainbow', num_distinct_colors))
        plt.title(f"Comunidades Detectadas (Amostra de {num_nos_vis} nós)")
        plt.axis('off')
        plt.savefig('comunidades_detectadas.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e: print(f"Erro ao visualizar comunidades: {e}")

def identificar_potenciais_espalhadores(grafo, comunidades=None, top_n=10):
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
            is_connected_check = nx.is_weakly_connected(grafo) if grafo.is_directed() else nx.is_connected(grafo)
            if is_connected_check: closeness = nx.closeness_centrality(grafo)
        except Exception: pass
    scores = {node: sum(filter(None, [pagerank.get(node), degree_cent.get(node), betweenness.get(node), closeness.get(node)])) / 
                    sum(1 for _ in filter(None, [pagerank.get(node), degree_cent.get(node), betweenness.get(node), closeness.get(node)])) 
              if sum(1 for _ in filter(None, [pagerank.get(node), degree_cent.get(node), betweenness.get(node), closeness.get(node)])) > 0 else 0 
              for node in grafo.nodes()}
    top_spreaders = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\nTop {top_n} potenciais espalhadores (score combinado):")
    print("-" * 100)
    print(f"{'Usuário':<15} | {'Score':<10} | {'PageRank':<10} | {'Grau Cent.':<12} | {'Intermediação':<15} | {'Proximidade':<12}")
    print("-" * 100)
    for user, score_val in top_spreaders:
        print(f"{user:<15} | {score_val:.6f} | {pagerank.get(user, 0):.6f} | {degree_cent.get(user, 0):.6f} | {betweenness.get(user, 0):.6f} | {closeness.get(user, 0):.6f}")
    if comunidades:
        print("\nDistribuição dos espalhadores por comunidades:")
        node_to_com_id = {node: i for i, com_set in enumerate(comunidades) for node in com_set}
        com_spread_count = {}
        for user, _ in top_spreaders:
            if user in node_to_com_id:
                com_id = node_to_com_id[user]
                com_spread_count[com_id] = com_spread_count.get(com_id, 0) + 1
        for com_id, count in sorted(com_spread_count.items(), key=lambda x: x[1], reverse=True):
            # Adiciona checagem para evitar IndexError em comunidades[com_id] se com_id for maior que o esperado
            tamanho_da_comunidade_str = str(len(comunidades[com_id])) if com_id < len(comunidades) else 'N/A (ID fora do esperado)'
            if count > 0: print(f"Comunidade ID {com_id} (contém {tamanho_da_comunidade_str} nós): {count} espalhadores ({count/len(top_spreaders)*100:.1f}%)")
    return top_spreaders, scores

def analisar_metricas_adicionais(grafo):
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
        maior_comp_nos = max(comp_gen, key=len, default=[]) # Adicionado default
        if maior_comp_nos and len(maior_comp_nos) > 1: # Checa se maior_comp_nos não está vazio
            sub_maior = grafo.subgraph(maior_comp_nos)
            if len(maior_comp_nos) < 1000:
                print(f"Calculando diâmetro do maior componente ({len(maior_comp_nos)} nós)...")
                diam = nx.diameter(sub_maior.to_undirected() if sub_maior.is_directed() else sub_maior)
                print(f"Diâmetro do maior componente: {diam}")
            else: print(f"Diâmetro ignorado (maior comp. com {len(maior_comp_nos)} nós).")
        else: print("Não há componente grande para calcular diâmetro.")
    except Exception as e: print(f"Erro diâmetro: {e}")

def gerar_subgrafo_reduzido(grafo_original, tamanho=500, metodo='grau'): # Definição da função
    num_nos_original = grafo_original.number_of_nodes()
    if num_nos_original == 0:
        print("AVISO: Grafo original para redução está vazio.")
        return grafo_original.copy()
    tamanho_efetivo = min(tamanho, num_nos_original)
    if num_nos_original <= tamanho_efetivo:
        print(f"INFO: Grafo original com {num_nos_original} nós. Usando cópia do original.")
        return grafo_original.copy() 
    print(f"INFO: Reduzindo grafo de {num_nos_original} nós para ~{tamanho_efetivo} via '{metodo}'.")
    nos_selecionados = []
    if metodo == 'grau':
        nos_ordenados_por_grau = sorted(grafo_original.degree(), key=lambda item: item[1], reverse=True)
        nos_selecionados = [no for no, grau in nos_ordenados_por_grau[:tamanho_efetivo]]
    elif metodo == 'aleatorio':
        nos_selecionados = random.sample(list(grafo_original.nodes()), tamanho_efetivo)
    else:
        print(f"AVISO: Método '{metodo}' desconhecido. Usando aleatório.")
        nos_selecionados = random.sample(list(grafo_original.nodes()), tamanho_efetivo)
    if not nos_selecionados:
        print("ERRO: Nenhum nó selecionado para subgrafo.")
        return grafo_original.copy()
    subgrafo = grafo_original.subgraph(nos_selecionados).copy() 
    print(f"INFO: Subgrafo reduzido: {subgrafo.number_of_nodes()} nós, {subgrafo.number_of_edges()} arestas.")
    return subgrafo

def main():
    print("\n================================================")
    print("   ANÁLISE DE REDES SOCIAIS PARA DETECÇÃO DE FAKE NEWS")
    print("================================================\n")
    
    caminho_arquivo = r"C:\Users\gonca\OneDrive\Documentos\GitHub\projeto-grafos\arquivooriginal(pesado)\higgs-retweet_network.edgelist" 
    
    if not os.path.exists(caminho_arquivo):
        caminho_absoluto = os.path.abspath(caminho_arquivo)
        print(f"AVISO: Arquivo não encontrado: {caminho_absoluto}")
        caminho_arquivo_input = input("Por favor, insira o caminho correto: ")
        if not os.path.exists(caminho_arquivo_input):
            print(f"ERRO: Arquivo '{caminho_arquivo_input}' não encontrado. Encerrando.")
            sys.exit(1)
        caminho_arquivo = caminho_arquivo_input
    
    grafo_original_completo = carregar_grafo(caminho_arquivo) 
    if grafo_original_completo.number_of_nodes() == 0:
        print("O grafo original carregado está vazio. Encerrando.")
        sys.exit(1)
    print(f"\n--- Grafo Original Completo Carregado: {grafo_original_completo.number_of_nodes()} nós, {grafo_original_completo.number_of_edges()} arestas ---")

    # <<< VALOR ALTERADO AQUI >>>
    TAMANHO_SUBGRAFO_ALVO = 500  # Alterado para 500
    
    grafo_para_analise = gerar_subgrafo_reduzido(grafo_original_completo, 
                                                 tamanho=TAMANHO_SUBGRAFO_ALVO, 
                                                 metodo='grau') 
    
    if grafo_para_analise.number_of_nodes() == 0:
        print("O subgrafo reduzido está vazio. Encerrando.")
        sys.exit(1)
    print(f"--- Iniciando análise no SUBGRAFO com {grafo_para_analise.number_of_nodes()} nós e {grafo_para_analise.number_of_edges()} arestas ---")

    mostrar_estatisticas(grafo_para_analise)
    analisar_centralidades(grafo_para_analise) 
    comunidades = detectar_comunidades(grafo_para_analise)
    
    print("\nCalculando métricas do SUBGRAFO para visualização e exportação...")
    pagerank_scores, degree_centrality_scores = {}, {}
    betweenness_centrality_scores, closeness_centrality_scores = {}, {}
    
    try: pagerank_scores = nx.pagerank(grafo_para_analise, alpha=0.85)
    except Exception as e: print(f"Falha PageRank (subgrafo): {e}")
    try: degree_centrality_scores = nx.degree_centrality(grafo_para_analise)
    except Exception as e: print(f"Falha GrauCent (subgrafo): {e}")

    num_nos_para_centralidades_caras = 2000 
    if grafo_para_analise.number_of_nodes() < num_nos_para_centralidades_caras:
        try: betweenness_centrality_scores = nx.betweenness_centrality(grafo_para_analise)
        except Exception as e: print(f"Falha Betweenness (subgrafo): {e}")
        try:
            is_connected_check = nx.is_weakly_connected(grafo_para_analise) if grafo_para_analise.is_directed() else nx.is_connected(grafo_para_analise)
            if is_connected_check: closeness_centrality_scores = nx.closeness_centrality(grafo_para_analise)
            else: print("Subgrafo não conectado, Closeness não calculada.")
        except Exception as e: print(f"Falha Closeness (subgrafo): {e}")
    else:
        print(f"Subgrafo com {grafo_para_analise.number_of_nodes()} nós. Intermediação e Proximidade serão puladas (demorado).")

    top_spreaders, combined_scores = identificar_potenciais_espalhadores(grafo_para_analise, comunidades)
    analisar_metricas_adicionais(grafo_para_analise)
    
    print("\n=== GERANDO VISUALIZAÇÕES PYTHON (DO SUBGRAFO) ===")
    plotar_distribuicao_graus(grafo_para_analise) 
    visualizar_subgrafo(grafo_para_analise, tamanho=100) # Mostra top 100 por grau da amostra de 5000
    
    visualizar_comunidades(grafo_para_analise, comunidades, max_nodes=min(500, TAMANHO_SUBGRAFO_ALVO)) 

    NOS_PARA_VISUALIZACOES_METRICAS = min(250, TAMANHO_SUBGRAFO_ALVO) 
    ROTULOS_NAS_VISUALIZACOES = 7

    if pagerank_scores:
        visualizar_metricas_nos(grafo_para_analise, pagerank_scores, 
                                "Visualização de PageRank (Subgrafo)", 
                                "subgrafo_visualizacao_pagerank.png", 
                                max_nodes_vis=NOS_PARA_VISUALIZACOES_METRICAS, top_n_labels=ROTULOS_NAS_VISUALIZACOES)
    if degree_centrality_scores:
        visualizar_metricas_nos(grafo_para_analise, degree_centrality_scores, 
                                "Visualização de Grau de Centralidade (Subgrafo)", 
                                "subgrafo_visualizacao_grau_centralidade.png", 
                                max_nodes_vis=NOS_PARA_VISUALIZACOES_METRICAS, top_n_labels=ROTULOS_NAS_VISUALIZACOES)
    if betweenness_centrality_scores:
        visualizar_metricas_nos(grafo_para_analise, betweenness_centrality_scores, 
                                "Visualização de Intermediação (Subgrafo)", 
                                "subgrafo_visualizacao_intermediacao.png", 
                                max_nodes_vis=NOS_PARA_VISUALIZACOES_METRICAS, top_n_labels=ROTULOS_NAS_VISUALIZACOES)
    if closeness_centrality_scores:
        visualizar_metricas_nos(grafo_para_analise, closeness_centrality_scores, 
                                "Visualização de Proximidade (Subgrafo)", 
                                "subgrafo_visualizacao_proximidade.png", 
                                max_nodes_vis=NOS_PARA_VISUALIZACOES_METRICAS, top_n_labels=ROTULOS_NAS_VISUALIZACOES)
    
    exportar_gephi(grafo_para_analise, comunidades_detectadas=comunidades, 
                   nome_arquivo="subgrafo_para_gephi_com_comunidades.gexf") 
    
    print("\n================================================")
    print("   ANÁLISE (NO SUBGRAFO) CONCLUÍDA COM SUCESSO!")
    print("================================================")
    print("\nArquivos gerados (verifique o diretório do script):")
    print("- distribuicao_graus.png (do subgrafo)")
    print("- subgrafo_amostra_top_grau_com_pesos.png (amostra do subgrafo por grau)") 
    print("- comunidades_detectadas.png (do subgrafo)")
    if pagerank_scores: print("- subgrafo_visualizacao_pagerank.png")
    if degree_centrality_scores: print("- subgrafo_visualizacao_grau_centralidade.png")
    if betweenness_centrality_scores: print("- subgrafo_visualizacao_intermediacao.png")
    if closeness_centrality_scores: print("- subgrafo_visualizacao_proximidade.png")
    print("- subgrafo_para_gephi_com_comunidades.gexf")

if __name__ == "__main__":
    main()