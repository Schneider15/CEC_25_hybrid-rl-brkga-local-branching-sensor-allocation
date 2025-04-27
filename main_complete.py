###############################################################################
# main_complete.py: comprehensive script for BRKGA-MP-IPR experiments
#                   using Python.
# ########################################
"""
Usage:
  main_complete.py -c <config_file> -s <seed> -r <stop_rule> \
-a <stop_arg> -t <max_time> -i <instance_file> [--no_evolution]

  main_complete.py (-h | --help)

Options:
  -c --config_file <arg>    Text file with the BRKGA-MP-IPR parameters.

  -s --seed <arg>           Seed for the random number generator.

  -r --stop_rule <arg>      Stop rule where:
                            - (G)enerations: number of evolutionary
                              generations.
                            - (I)terations: maximum number of generations
                              without improvement in the solutions.
                            - (T)arget: runs until obtains the target value.

  -a --stop_arg <arg>       Argument value for '-r'.

  -t --max_time <arg>       Maximum time in seconds.

  -i --instance_file <arg>  Instance file.

  --no_evolution      If supplied, no evolutionary operators are applied. So,
                      the algorithm becomes a simple multi-start algorithm.

  -h --help           Produce help message.
"""

from copy import deepcopy
from datetime import datetime
from os.path import basename
import time
import csv
import math
import numpy as np
import pandas as pd
import random
import os
import networkx as nx
import matplotlib.pyplot as plt

import docopt

from brkga_mp_ipr.algorithm import BrkgaMpIpr
from brkga_mp_ipr.enums  import ParsingEnum, Sense
from brkga_mp_ipr.types_io import load_configuration
from sap_decoder import SAPDecoder
from sap_decoder import SAPDecoderRefiner
from sap_decoder import main_DQN
from CEC_NWS_InstanceReader import GraphInstanceReader # Importando diretamente as classes de leitura de instância e heurística,
from CEC_NWS_Constructive_Heuristic_Allocation import SensorAllocationHeuristic
from CEC_MLPwarm_BRKGA import solve_wsn_optimization_docplex

def save_results_to_csv(results, filename="BRKGA NWS/Results/semi.csv"):
    """
    Função para salvar os resultados em um arquivo CSV, incluindo o grid_size e delta_LB.
    """
    # Verifica se o arquivo já existe para adicionar os cabeçalhos
    file_exists = os.path.exists(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)  # Usa o módulo csv para escrever no arquivo

        if not file_exists:      
            writer.writerow([
                "Instance", "Seed", "GridSize", "NumVertices", "NumEdges", 
                "InitialSolutionAllocation", "InitialSolutionCost", "NumIterationsHeuristic",
                "TimeHeuristic", "BestCentralityHeuristic", "TotalNumIterations-BRKGA", 
                "TotalElapsedTime-BRKGA", "LargeOffset-BRKGA", "LastUpdateIteration-BRKGA", 
                "LastUpdateTime-BRKGA", "Final_Sol_BRKGA", "FinalSolutionCost-BRKGA", 
                "delta_LB", "Best Cost LB", "Sensor Allocations LB", 
                "Elapsed Time Exact LB (s)", "MIP Gap LB", "TotalRLTime", "ImprovedRL"
            ])

        # Escreve os resultados organizados
        writer.writerow(results)

def get_excel_file_path(instance_name: str) -> str:
    """
    Determina o caminho do arquivo Excel com base no nome da instância.
    O arquivo é procurado no diretório 'BRKGA NWS/Results'.
    """
    base_path = "BRKGA NWS/Results"
    
    if instance_name.startswith("regular"):
        return os.path.join(base_path, "BRKGA_reg.csv")
    elif instance_name.startswith("semi_regular"):
        return os.path.join(base_path, "BRKGA_RL_SemiReg.csv")
    elif instance_name.startswith("irregular"):
        return os.path.join(base_path, "BRKGA_rl_semi_irreg.csv")
    else:
        raise ValueError(f"Instância com nome {instance_name} não corresponde a um tipo conhecido (regular, semi_regular, irregular).")

def load_initial_solution_from_heuristic(file_path: str, instance_name: str, num_vertices: int):
    """
    Carrega a solução inicial e outras informações de um arquivo Excel com base no nome da instância.
    Verifica se o tamanho do dicionário de alocação corresponde ao número de vértices.
    """
    df = pd.read_csv(file_path)
    instance_data = df[df['Instance'] == instance_name]

    if instance_data.empty:
        raise ValueError(f"A instância {instance_name} não foi encontrada na planilha.")

    initial_solution_str = instance_data.iloc[0]['InitialSolutionAllocation']
    initial_solution_cost = instance_data.iloc[0]['InitialSolutionCost']
    num_iterations_heuristic = instance_data.iloc[0]['NumIterationsHeuristic']
    time_heuristic = instance_data.iloc[0]['TimeHeuristic']
    best_centrality = instance_data.iloc[0]['BestCentralityHeuristic']
    grid_size = int(instance_data.iloc[0]['GridSize'])
    num_edges = int(instance_data.iloc[0]['NumEdges'])

    initial_solution = eval(initial_solution_str) 
    
    # Verifica se o número de vértices da solução inicial corresponde ao número de vértices esperado
    if len(initial_solution) != num_vertices:
        raise ValueError(f"O número de vértices na solução inicial ({len(initial_solution)}) não corresponde ao número de vértices esperado ({num_vertices}).")
    
    return {
        'initial_solution': initial_solution,
        'initial_solution_cost': initial_solution_cost,
        'num_iterations_heuristic': num_iterations_heuristic,
        'time_heuristic': time_heuristic,
        'best_centrality': best_centrality,
        'grid_size': grid_size,
        'num_edges': num_edges
    }

def generate_chromosome_from_solution(initial_solution, vertices_list, seed):
    """
    Gera o cromossomo a partir da solução inicial fornecida.
    """
    num_vertices = len(vertices_list)
    initial_chromosome = [0.0] * num_vertices
    random.seed(seed)
    for i, v in enumerate(vertices_list):
        if initial_solution.get(v) == 'X':
            initial_chromosome[i] = 0.5 + random.random() * 0.5  # (0.5, 1]
        else:
            initial_chromosome[i] = random.random() * 0.49  # [0, 0.5)
    
    return initial_chromosome



def draw_allocation(vertices_list, adjacency_list, final_solution, instance_name, best_cost, algoritmo="BRKGA", drawpath='Results/graph_news'):
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
    if algoritmo == "BRKGA":
        selected_palette = color_palettes["Azul"]
    elif algoritmo == "heuristic":
        selected_palette = color_palettes["Vermelho"]
    else:
        selected_palette = color_palettes["Verde"]

    node_colors = []
    labels = {
        'X': 'Sensor X',
        'Y': 'Sensor Y',
        'Z': 'Sensor Z'
    }
    s_x_nodes = [node for node, sensor in final_solution.items() if sensor == 'X']
    s_y_nodes = [node for node, sensor in final_solution.items() if sensor == 'Y']
    s_z_nodes = [node for node, sensor in final_solution.items() if sensor == 'Z']
    G = nx.Graph()
    G.add_nodes_from(vertices_list)  # Adiciona os nós
    for node, adj_nodes in adjacency_list.items():
        for adj_node in adj_nodes:
            G.add_edge(node, adj_node)  # Adiciona as arestas
    for node in G.nodes():
        if node in s_x_nodes:
            node_colors.append(selected_palette['X'])
        elif node in s_y_nodes:
            node_colors.append(selected_palette['Y'])
        elif node in s_z_nodes:
            node_colors.append(selected_palette['Z'])
        else:
            node_colors.append('gray')  # Para nós não alocados
    if len(node_colors) != len(G.nodes()):
        print(f"Erro: o número de cores ({len(node_colors)}) não corresponde ao número de nós ({len(G.nodes())}).")
        return
    plt.figure(figsize=(8, 6))
    num_nodes = len(G.nodes())
    cols = int(math.sqrt(num_nodes))
    rows = (num_nodes + cols - 1) // cols  # Arredondar para cima

    pos = {
        node: (i % cols, rows - 1 - (i // cols))
        for i, node in enumerate(G.nodes())
    }
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.0, edge_color='gray')  # Arestas em cinza
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=labels[key], 
                           markerfacecolor=selected_palette[key], markersize=10) for key in selected_palette.keys()]
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.92, 0.05))  # Ajustando a posição da legenda
    plt.title(f"Best Solution {algoritmo} - {num_nodes} sensor nodes \nCost: {best_cost}")
    plt.savefig(f"{drawpath}/_{instance_name}_{algoritmo}.png")
    plt.close()  
        

###############################################################################
# Enumerations and constants
###############################################################################

class StopRule(ParsingEnum):
    """
    Controls stop criteria. Stops either when:
    - a given number of `GENERATIONS` is given;
    - or a `TARGET` value is found;
    - or no `IMPROVEMENT` is found in a given number of iterations.
    """
    GENERATIONS = 0
    TARGET = 1
    IMPROVEMENT = 2

###############################################################################

def main() -> None:
    """
    Proceeds with the optimization. Create to avoid spread `global` keywords
    around the code.
    """

    args = docopt.docopt(__doc__)
    # print(args)

    configuration_file = args["--config_file"]
    instance_file = args["--instance_file"]
    seed = int(args["--seed"])
    stop_rule = StopRule(args["--stop_rule"])

    if stop_rule == StopRule.TARGET:
        stop_argument = float(args["--stop_arg"])
    else:
        stop_argument = int(args["--stop_arg"])

    maximum_time = float(args["--max_time"])

    if maximum_time <= 0.0:
        raise RuntimeError(f"Maximum time must be larger than 0.0. "
                           f"Given {maximum_time}.")

    perform_evolution = not args["--no_evolution"]

    ########################################
    # Load config file and show basic info.
    ########################################

    brkga_params, control_params = load_configuration(configuration_file)

    print(f"""------------------------------------------------------
> Experiment started at {datetime.now()}
> Instance: {instance_file}
> Configuration: {configuration_file}
> Algorithm Parameters:""", end="")

    if not perform_evolution:
        print(">    - Simple multi-start: on (no evolutionary operators)")
    else:
        output_string = ""
        for name, value in vars(brkga_params).items():
            output_string += f"\n>  -{name} {value}"
        for name, value in vars(control_params).items():
            output_string += f"\n>  -{name} {value}"

        print(output_string)
        print(f"""> Seed: {seed}
> Stop rule: {stop_rule}
> Stop argument: {stop_argument}
> Maximum time (s): {maximum_time}
------------------------------------------------------""")

    print(f"\n[{datetime.now()}] Reading SAP instance via GraphInstanceReader...")


    # Precisamos para gerar a SOLUÇÃO INICIAL via heurística.
    reader = GraphInstanceReader(instance_file)
    reader.read_instance()
    (
        vertices_info,
        adjacency_list,
        second_adjacency_list,
        centralities_raw,
        num_vertices,
        num_edges,
        sensor_costs,
        sensor_ranges,
        vertices_list,
        converted_centralities,
        grid_size
    ) = reader.get_data()

    print(f"Number of vertices: {num_vertices}, edges: {num_edges}")
    print("Sensor costs:", sensor_costs)

    instance_name = os.path.basename(instance_file) 
    file_path = get_excel_file_path(instance_name)  

    if os.path.exists(file_path):
        try:
            initial_data = load_initial_solution_from_heuristic(file_path, instance_name, num_vertices)
            best_solution_heur = initial_data['initial_solution']
            best_cost_heur = initial_data['initial_solution_cost']
            best_iteration = initial_data['num_iterations_heuristic']
            time_heur = initial_data['time_heuristic']
            best_centrality = initial_data['best_centrality']
            grid_size = initial_data['grid_size']
            num_edges = initial_data['num_edges']

            print(f"\n[{datetime.now()}] Solution loaded from Excel file:")
            print(f"Initial cost from file: {best_cost_heur}")
            initial_chromosome = generate_chromosome_from_solution(best_solution_heur, vertices_list, seed)
            print(f"Initial Chromosome from saved solution: {initial_chromosome}")

        except ValueError as e:
            print(f"[{datetime.now()}] Error: {e}")
            print(f"\n[{datetime.now()}] Running heuristic to generate initial solution...")
            heuristic = SensorAllocationHeuristic(
                vertices=vertices_list,
                adjacency_list=adjacency_list,
                second_adj_list=second_adjacency_list,
                sensor_costs=sensor_costs,
                centralities=converted_centralities
            )

            best_solution_heur, best_cost_heur, best_centrality, best_iteration, time_heur = heuristic.run_multi_centralities(
                instance=instance_file,
                num_iterations=1000,
                percentual=0.15,
            )
            print(f"Initial cost from heuristic: {best_cost_heur}")
            initial_chromosome = [0.0] * num_vertices
            random.seed(seed)  
            for i, v in enumerate(vertices_list):
                if best_solution_heur.get(v) == 'X':
                    initial_chromosome[i] = 0.5 + random.random() * 0.5  # (0.5, 1]
                else:
                    initial_chromosome[i] = random.random() * 0.49  # [0, 0.5)

            print(f"Initial Chromosome from heuristic: {initial_chromosome}")

    else:
        print(f"\n[{datetime.now()}] No saved solution found. Running heuristic to generate initial solution...")
        heuristic = SensorAllocationHeuristic(
            vertices=vertices_list,
            adjacency_list=adjacency_list,
            second_adj_list=second_adjacency_list,
            sensor_costs=sensor_costs,
            centralities=converted_centralities
        )

        best_solution_heur, best_cost_heur, best_centrality, best_iteration, time_heur = heuristic.run_multi_centralities(
            instance=instance_file,
            num_iterations=1000,
            percentual=0.15,
        )
        print(f"Initial cost from heuristic: {best_cost_heur}")
        # print("Initial solution from heuristic", best_solution_heur)

        # Gerar cromossomo a partir da solução heurística
        initial_chromosome = [0.0] * num_vertices
        random.seed(seed)  # Garantir reprodutibilidade
        for i, v in enumerate(vertices_list):
            if best_solution_heur.get(v) == 'X':
                initial_chromosome[i] = 0.5 + random.random() * 0.5  # (0.5, 1]
            else:
                initial_chromosome[i] = random.random() * 0.49  # [0, 0.5)

        print(f"Initial Chromosome from heuristic: {initial_chromosome}")

    #  Decoder:
    
    start_time = time.time()

    print(f"\n[{datetime.now()}] Building BRKGA data...")


    brkga_params.population_size = min(brkga_params.population_size, 10 * num_vertices)
    print(f"New population size: {brkga_params.population_size}")
    sap_decoder = SAPDecoderRefiner(instance_file, penalty_value=4.1)  # Ajuste penalty se quiser

    brkga = BrkgaMpIpr(
        decoder=sap_decoder,
        sense=Sense.MINIMIZE,
        seed=seed,
        chromosome_size = num_vertices,
        params        = brkga_params,
        evolutionary_mechanism_on = perform_evolution
    )

    random.seed(seed)
    keys = sorted([random.random() for _ in range(num_vertices)])
    brkga.set_initial_population([initial_chromosome])

    print(f"\n[{datetime.now()}] Initializing BRKGA data...")
    brkga.initialize()

    ########################################
    # Warm up
    ########################################
    print(f"\n[{datetime.now()}] Warming up...")

    bogus_alg = deepcopy(brkga)
    bogus_alg.evolve(2)
    bogus_alg.get_best_fitness()
    bogus_alg.get_best_chromosome()
    bogus_alg = None

    ########################################
    # Evolving
    ########################################

    print(f"\n[{datetime.now()}] Evolving...")
    print("* Iteration | Cost | CurrentTime")

    best_cost = best_cost_heur
    best_chromosome = initial_chromosome

    iteration = 0
    last_update_time = 0.0
    last_update_iteration = 0
    large_offset = 0
    # num_homogenities = 0
    # num_best_improvements = 0
    # num_elite_improvements = 0
    run = True
    total_rl_time = 0.0
    while run:
        iteration += 1

        # Evolves one iteration with RL refinement for the best individual
        rl_time = brkga.evolve_with_rl(current_iteration=iteration, num_generations=1)

        # Acumula o tempo total de RL
        total_rl_time += rl_time

        # Checks the current results and holds the best.
        fitness = brkga.get_best_fitness()
        if fitness < best_cost:
            last_update_time = time.time() - start_time
            update_offset = iteration - last_update_iteration

            if large_offset < update_offset:
                large_offset = update_offset

            last_update_iteration = iteration
            best_cost = fitness
            best_chromosome = brkga.get_best_chromosome()
        
            print(f"* {iteration} | {best_cost:.3f} | {last_update_time:.2f}")
        # end if

        # TODO (ceandrade): implement path relink calls here.
        # Please, see Julia version for that.

        iter_without_improvement = iteration - last_update_iteration

        # Check stop criteria.
        run = not (
            (time.time() - start_time > maximum_time)
            or
            (stop_rule == StopRule.GENERATIONS and iteration == stop_argument)
            or
            (stop_rule == StopRule.IMPROVEMENT and
             iter_without_improvement >= stop_argument)
            or
            (stop_rule == StopRule.TARGET and best_cost <= stop_argument)
        )
    # end while
    total_elapsed_time = time.time() - start_time
    total_num_iterations = iteration

    print(f"[{datetime.now()}] End of optimization\n")

    print(f"Total number of iterations: {total_num_iterations}")
    print(f"Last update iteration: {last_update_iteration}")
    print(f"Total optimization time: {total_elapsed_time:.2f}")
    print(f"Last update time: {last_update_time:.2f}")
    print(f"Large number of iterations between improvements: {large_offset}")
    final_cost, final_solution = brkga._decoder.decode(best_chromosome, rewrite=False)
    
    print(f"\nBest final cost = {final_cost:.3f}")

    print(f"\n Alocação Final = {final_solution}")

    print("\nInstance,Seed,NumVertices,TotalIterations,TotalTime,"
          "LargeOffset,LastUpdateIteration,LastUpdateTime,Cost")

    print(f"{basename(instance_file)},{seed},{num_vertices},"
          f"{total_num_iterations},{total_elapsed_time:.2f},"
          f"{large_offset},{last_update_iteration},"
          f"{last_update_time:.2f},{best_cost:.3f}")


    grid_size_class = grid_size % 3  #
    quotient = grid_size // 3  # 
    if grid_size_class == 0:
        delta_LB = (quotient**2) 
    elif grid_size_class == 1:
        delta_LB = (quotient**2 + quotient)  
    else:
        delta_LB = ((quotient + 1)**2)  


    model_LB = solve_wsn_optimization_docplex(
        vertices=reader.vertices,
        sensors=['s_x','s_y','s_z'],
        costs=sensor_costs,
        instance_name=basename(instance_file),
        num_vertices=num_vertices,
        num_edges=num_edges,
        adjacency_list=adjacency_list,
        second_adjacency_list=second_adjacency_list,
        brkga_solution=final_solution,     
        local_branching_k=delta_LB,  
        time_limit=600                 
    )

    print(f"\n---Local Branching Results ---")
    print(model_LB)

    rl_improvement_count = brkga._rl_improvement_count if hasattr(brkga, "_rl_improvement_count") else 0
    results = [
        basename(instance_file),
        seed,
        grid_size,
        num_vertices,
        num_edges,
        str(best_solution_heur),  # Alocação inicial pela heurística
        int(best_cost_heur),
        best_iteration,
        time_heur,
        best_centrality,
        total_num_iterations,
        total_elapsed_time,
        large_offset,
        last_update_iteration,
        last_update_time,
        final_solution,  # Adiciona a contagem de melhorias pelo RL,
        int(final_cost),  # Custo BRKGA
        delta_LB,
        model_LB['best_cost'],  # Melhor custo no modelo de LB
        model_LB['sensor_allocations'],  # Alocação final de sensores
        model_LB['elapsed_time'],  # Tempo gasto no modelo exato
        model_LB['mip_gap'],  # GAP do modelo exato
        total_rl_time,  # Tempo total gasto no RL
        rl_improvement_count
    ]

    # Salva os resultados no CSV
    save_results_to_csv(results)
    print(f"[{datetime.now()}] End of optimization")
    print(f"Best final cost = {final_cost:.3f}")
    print('delta_LB',delta_LB)
    print('best_solution_heur',best_solution_heur)
    print('final_solution brkga',final_solution)
    model_LB_solution = {}
    for vertex in model_LB['sensor_allocations']['s_x']:
        model_LB_solution[vertex] = 'X'
    for vertex in model_LB['sensor_allocations']['s_y']:
        model_LB_solution[vertex] = 'Y'
    for vertex in model_LB['sensor_allocations']['s_z']:
        model_LB_solution[vertex] = 'Z'


    if grid_size < 20:
        instance_name = basename(instance_file)  
        drawpath = 'BRKGA NWS/Draws'  
        draw_allocation(vertices_list, adjacency_list, best_solution_heur, instance_name, best_cost_heur, algoritmo= "heuristic", drawpath=drawpath)
        draw_allocation(vertices_list, adjacency_list, final_solution, instance_name, int(best_cost), algoritmo= "BRKGA", drawpath=drawpath)
        draw_allocation(vertices_list, adjacency_list, model_LB_solution, instance_name, int(model_LB['best_cost']), algoritmo="BRKGA_LB", drawpath=drawpath)


if __name__ == "__main__":
    main()
