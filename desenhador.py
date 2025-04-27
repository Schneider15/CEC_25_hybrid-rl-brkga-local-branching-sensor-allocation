import networkx as nx
import matplotlib.pyplot as plt
import math
import sys
import os
sys.path.append(os.path.abspath('/home/rafael/Documents/Network Sensor Allocation/CEC'))
from CEC_NWS_InstanceReader import GraphInstanceReader # Importando diretamente as classes de leitura de instância e heurística,


# Função para desenhar a alocação dos sensores
def draw_allocation(vertices_list, adjacency_list, final_solution, instance_name, best_cost, algoritmo="BRKGA", drawpath='/home/rafael/Documents/Network Sensor Allocation/CEC/Results/graph_news'):
    """Desenha a solução final com diferentes paletas de cores, considerando a lista de vértices e a lista de adjacência."""

    # Dicionário com paletas de cores
    color_palettes = {
        "Azul": {
            'X': '#1E3A8A',  # Azul Escuro
            'Y': '#3B82F6',  # Azul Vivo
            'Z': '#93C5FD'   # Azul Claro
        },
        "Verde": {
            'X': '#064E3B',  # Verde Escuro
            'Y': '#10B981',  # Verde Vivo
            'Z': '#6EE7B7'   # Verde Claro
        },
        "Vermelho": {
            'X': '#700000',  # Vermelho Escuro
            'Y': '#FF0000',  # Vermelho Vivo
            'Z': '#FF7F7F'   # Vermelho Claro
        },
        "Laranja": {
            'X': '#C2410C',  # Laranja Escuro
            'Y': '#FB923C',  # Laranja Claro
            'Z': '#F97316'   # Laranja Vivo
        }
    }

    # Seleção da paleta com base no algoritmo
    if algoritmo == "BRKGA":
        selected_palette = color_palettes["Azul"]
    elif algoritmo == "MILP":
        selected_palette = color_palettes["Vermelho"]
    else:
        selected_palette = color_palettes["Verde"]

    node_colors = []
    labels = {
        'X': 'Sensor X',
        'Y': 'Sensor Y',
        'Z': 'Sensor Z'
    }

    # Inicializa as listas para os sensores X, Y, Z
    s_x_nodes = final_solution['s_x']
    s_y_nodes = final_solution['s_y']
    s_z_nodes = final_solution['s_z']

    # Criar o grafo a partir da lista de vértices e adjacência
    G = nx.Graph()
    G.add_nodes_from(vertices_list)  # Adiciona os nós
    for node, adj_nodes in adjacency_list.items():
        for adj_node in adj_nodes:
            G.add_edge(node, adj_node)  # Adiciona as arestas

    # Atribuindo cores aos nós
    for node in G.nodes():
        if node in s_x_nodes:
            node_colors.append(selected_palette['X'])
        elif node in s_y_nodes:
            node_colors.append(selected_palette['Y'])
        elif node in s_z_nodes:
            node_colors.append(selected_palette['Z'])
        else:
            node_colors.append('gray')  # Para nós não alocados

    # Garantindo que o número de cores corresponda ao número de nós
    if len(node_colors) != len(G.nodes()):
        print(f"Erro: o número de cores ({len(node_colors)}) não corresponde ao número de nós ({len(G.nodes())}).")
        return

    # Desenhando o grafo
# Desenhando o grafo
# Desenhando o grafo
    plt.figure(figsize=(8, 6))

    # Definindo a disposição dos nós no gráfico (posicionamento)
    num_nodes = len(G.nodes())
    cols = int(math.sqrt(num_nodes))
    rows = (num_nodes + cols - 1) // cols  # Arredondar para cima

    pos = {
        node: (i % cols, rows - 1 - (i // cols))
        for i, node in enumerate(G.nodes())
    }

    # Desenhando as arestas com estilo
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.0, edge_color='gray')  # Arestas em cinza

    # Desenhando os nós com as cores especificadas
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400)

    # Criando um dicionário de rótulos para cada nó baseado no tipo de sensor
    node_labels = {}
    for node in G.nodes():
        if node in s_x_nodes:
            node_labels[node] = 'X'
        elif node in s_y_nodes:
            node_labels[node] = 'Y'
        elif node in s_z_nodes:
            node_labels[node] = 'Z'
        else:
            node_labels[node] = ''  # Para nós não alocados (se houver)

    # Desenhando os rótulos dos tipos de sensores nos nós
    # Agora, podemos especificar uma cor para os rótulos do sensor X
    for node, label in node_labels.items():
        font_color = 'black' if label == 'X' else 'black'  # Rótulo 'X' será branco, outros são pretos
        nx.draw_networkx_labels(G, pos, labels={node: label}, font_size=10, font_weight='bold', font_color=font_color)

    # Criando a legenda para os sensores
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=labels[key], 
                        markerfacecolor=selected_palette[key], markersize=10) for key in selected_palette.keys()]
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.8, 0.03))  # Ajustando a posição da legenda
    # Título com o custo final
    # plt.title(f"Best Solution {algoritmo} - {num_nodes} sensor nodes \nCost: {best_cost}")

    # Remover o retângulo (borda) ao redor do grafo
    plt.axis('off')  # Desativa os eixos e a borda ao redor

    # Salvar a imagem no diretório especificado, com o nome da paleta como sufixo
    plt.savefig(f"{drawpath}/_{instance_name}_{algoritmo}_unlabeld.png", bbox_inches='tight', pad_inches=0)
    plt.close()  # Fecha a figura para liberar memória


if __name__ == "__main__":
    # Caminho onde as instâncias estão salvas
    file_path = '/home/rafael/Documents/Network Sensor Allocation/CEC/Instances/Grids/Regular/regular_grid_64.dat'


    # Carrega a instância do grafo
    reader = GraphInstanceReader(file_path)
    reader.read_instance()
    reader.print_summary()

    # Obtém os dados do grafo
    vertices, adjacency_list, second_adjacency_list, centralities, num_vertices, num_edges, sensor_costs, sensor_ranges, vertices_list, converted_centralities, grid_size = reader.get_data()

    # Alocação dos sensores
    final_solution = {
        's_x': [10, 13, 15, 34, 37, 39, 52, 55, 57],
        's_y': [29, 35, 45],
        's_z': [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 36, 38, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 53, 54, 56, 58, 59, 60, 61, 62, 63, 64]
    }

    # Nome da instância (para o título da imagem)
    instance_name = 'regular_grid_64'

    # Custo da melhor solução (pode ser um valor arbitrário ou real dependendo da solução encontrada)
    best_cost = 94  # Exemplo de custo, altere conforme necessário

    # Diretório onde a imagem será salva
    drawpath = '/home/rafael/Documents/Network Sensor Allocation/CEC/Results/graph_news'

    # Desenha a alocação dos sensores
    draw_allocation(vertices_list, adjacency_list, final_solution, instance_name, best_cost, algoritmo="MILP", drawpath=drawpath)
