from brkga_mp_ipr.types import BaseChromosome
import sys
import os
from datetime import datetime
sys.path.append(os.path.abspath('/home/rafael/Documents/Network Sensor Allocation/CEC')) # Adiciona o caminho da pasta onde os módulos auxiliares estão localizados
from CEC_NWS_InstanceReader import GraphInstanceReader # Importando diretamente as classes de leitura de instância e heurística,
from CEC_NWS_Constructive_Heuristic_Allocation import SensorAllocationHeuristic

### Importações para o RL:
import pandas as pd
import random
from pathlib import Path
import math
import time
import numpy as np
from brkga_mp_ipr.types import BaseChromosome
from torch.nn import MSELoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module, Linear, MSELoss
import torch.nn.functional as F
from torch.optim import Adam
from dataclasses import dataclass, field
import numpy as np
from torch import Tensor as T
import gym
from gym import spaces
import networkx as nx

# Classe de decodificação das soluções:
class SAPDecoder:
    """
    Decoder para o Sensor Allocation Problem (SAP).
    aloca X, Y, Z e penaliza vértices que fiquem sem cobertura.
    """

    def __init__(self, instance_path: str, penalty_value: float = 4.1):
        """
        Construtor: Lê a instância e carrega as estruturas necessárias.

        Parâmetros:
          - instance_path: caminho do arquivo de instância .dat
          - penalty_value: valor de penalização para cada vértice livre 
                          que não pode ser coberto por nenhum sensor X.
        """
        # 1) Ler a instância
        self.reader = GraphInstanceReader(instance_path)
        self.reader.read_instance()
        self.instance_file = instance_path

        # Extrair dados
        (
            self.vertices_info,            # [(v_id, 'r.c'), ...]
            self.adjacency_list,           # {v: [vizinhos]}
            self.second_adjacency_list,    # {v: [vizinhos_de_2a_ordem]}
            self.centralities_raw,         # dict com centralidades brutas
            self.num_vertices,
            self.num_edges,
            self.sensor_costs,            # [cost_X, cost_Y, cost_Z]
            self.sensor_ranges,           # ex: [0, 2, 1]
            self.vertices_list,           # ex: [1, 2, 3, ...]
            self.converted_centralities,   # {'degree centrality': {1: val, ...}, ...}
            self.grid_size
            
        ) = self.reader.get_data()

        # Armazena custos de cada tipo de sensor
        # Assumimos que sensor_costs[0] = custo X, [1] = custo Y, [2] = custo Z
        self.cost_X = self.sensor_costs[0] if len(self.sensor_costs) > 0 else 4
        self.cost_Y = self.sensor_costs[1] if len(self.sensor_costs) > 1 else 2
        self.cost_Z = self.sensor_costs[2] if len(self.sensor_costs) > 2 else 1

        # Tamanho do cromossomo = número de vértices
        self.n = self.num_vertices

        # Valor de penalização para cada vértice não coberto
        self.penalty_value = penalty_value

    def decode(self, chromosome: BaseChromosome, rewrite: bool) -> float:
        """
        Converte o cromossomo em uma solução X, Y, Z e calcula seu custo com penalização.
        
        Regras:
          - gene > 0.5 => vértice v recebe sensor X
          - gene <= 0.5 => vértice v não recebe X (ficará livre, possivelmente Y ou Z)
          - Para cada X, alocamos Z nos vizinhos imediatos, Y nos vizinhos de 2a ordem
          - Se ainda houver vértice livre que NÃO está na adj ou 2a adj de nenhum X, 
            aplicamos penalização.
        """

        if len(chromosome) != self.n:
            raise ValueError(f"Tamanho do cromossomo ({len(chromosome)}) difere do esperado ({self.n}).")

        # Exemplo de atribuição binária + proximidade (original):
        solution = {}
        allocated_X = []

        # Atribuição inicial
        for i in range(self.n):
            v = self.vertices_list[i]
            if chromosome[i] > 0.5:
                solution[v] = 'X'
                allocated_X.append(v)
            else:
                solution[v] = None

        # Alocar Z para vizinhos imediatos e Y para vizinhos de 2a ordem
        for x_vertex in allocated_X:
            # Vizinhos de 1 hop => Z
            for z_candidate in self.adjacency_list[x_vertex]:
                if solution[z_candidate] is None:
                    solution[z_candidate] = 'Z'
            # Vizinhos de 2 hops => Y
            for y_candidate in self.second_adjacency_list[x_vertex]:
                if solution[y_candidate] is None:
                    solution[y_candidate] = 'Y'

        # Penalização:
        uncovered_vertices = [v for v, s in solution.items() if s is None]
        cost = 0.0
        if uncovered_vertices:
            cost += len(uncovered_vertices)*self.penalty_value
            # Se quiser forçar esses 'None' como 'X'
            for v in uncovered_vertices:
                solution[v] = 'X'
                # Opcionalmente, reescrever gene>0.5
                chromosome[self.vertices_list.index(v)] = random.uniform(0.5,1.0)

        # Ajuste de vizinhos: rebaixar Y para Z se tiver um sensor X no range hop 1 - primeira vizinhança
        #   Se um nó está marcado como Y mas é 'vizinho imediato' de X,
        #   então trocamos para Z. 
        for v, sensor in solution.items():
            if sensor == 'Y':
                # verifica se existe algum X no adjacency
                for neigh in self.adjacency_list[v]:
                    if solution.get(neigh) == 'X':
                        # Rebaixa de Y para Z
                        solution[v] = 'Z'
                        break  # pode sair do loop de vizinhos

        # Cálculo final do custo
        final_cost = 0.0
        for v, s in solution.items():
            if s == 'X':
                final_cost += self.cost_X
            elif s == 'Y':
                final_cost += self.cost_Y
            elif s == 'Z':
                final_cost += self.cost_Z

        return final_cost, solution
    

class SAPDecoderRefiner(SAPDecoder):
    """
    Refinador para o Sensor Allocation Problem (SAP).
    Aplica o reinforcement learning (RL) após a decodificação para refinar a solução.
    """

    def __init__(self, instance_path: str, penalty_value: float = 4.1, max_iterations: int = 100):
        """
        Construtor: Inicializa o decodificador base e carrega a estrutura necessária.
        
        Parâmetros:
          - instance_path: Caminho para o arquivo da instância.
          - penalty_value: Valor de penalização para vértices não cobertos.
          - max_iterations: Número máximo de iterações do BRKGA, usado no cálculo da probabilidade.
        """
        super().__init__(instance_path, penalty_value)
        self.max_iterations = max_iterations  

    def should_use_rl(self, current_iteration: int) -> bool:
        """
        Determina se o RL deve ser usado nesta iteração, com base na fração do número total de iterações.

        Parâmetros:
          - current_iteration: Iteração atual no loop do BRKGA.
        
        Retorna:
          - True se o RL deve ser aplicado, False caso contrário.
        """
        if current_iteration == 1 or current_iteration % 250 == 0:
            return True
        else:
            return False

    def refine_solution(self, chromosome: BaseChromosome, current_iteration: int, rewrite: bool = True):
        """
        Decodifica a solução e decide se o RL deve ser aplicado com base na iteração atual.

        Parâmetros:
        - chromosome: Cromossomo para decodificação.
        - current_iteration: Iteração atual no loop do BRKGA.
        - rewrite: Indica se o cromossomo deve ser reescrito após a refinação.

        Retorna:
        - Melhor custo encontrado.
        - Melhor solução no formato de dicionário.
        - Tempo gasto no RL (ou 0.0 se o RL não for aplicado).
        """
        initial_cost, initial_solution = self.decode(chromosome, rewrite)
        use_rl = self.should_use_rl(current_iteration)
        if not use_rl:
            return initial_cost, initial_solution, 0.0, False
        num_vertices = len(self.vertices_list)
        if num_vertices < 100:
            n_episodes = math.ceil(num_vertices)  
        elif num_vertices <= 1000:
            n_episodes = math.ceil(num_vertices / 2)  
        elif num_vertices <= 5000:
            n_episodes = math.ceil(num_vertices /10)  
        else:
            n_episodes = math.ceil(num_vertices / 100) 


        rl_start_time = time.time()  
        _, rl_solution, rl_cost, _, _ = main_DQN(
            instance_path=self.instance_file,
            cost=initial_cost,
            solution=initial_solution,
            n_episodes=n_episodes,
        )
        rl_time = time.time() - rl_start_time  # Calcula o tempo de execução do RL

        # Verificar se o RL produziu uma solução melhor
        if rl_cost > 0 and rl_cost < initial_cost:
            for i, vertex in enumerate(self.vertices_list):
                if rl_solution.get(vertex) == 'X':
                    chromosome[i] = 1  # Gene correspondente a X
                else:
                    chromosome[i] = 0
                # print('rl_cost < initial_cost',rl_cost, initial_cost)  # Gene correspondente a não X
            return rl_cost, rl_solution, rl_time, True  # Indica melhora
        else:
            return initial_cost, initial_solution, rl_time, False

    
    @property
    def num_vertices_sap(self):
        """
        Retorna o número de vértices, útil para o main saber o tamanho do cromossomo.
        """
        return self.n


class SensorEnv(gym.Env):
    def __init__(self, instance_path, initial_solution):
        super(SensorEnv, self).__init__()
        print("[DEBUG] Inicializando SensorEnv e lendo instância.")
        self.reader = GraphInstanceReader(instance_path)
        self.reader.read_instance()
        (   self.vertices_info,
            self.adjacency_list,
            self.second_adjacency_list,
            self.centralities_raw,
            self.num_vertices,
            self.num_edges,
            self.sensor_costs,
            self.sensor_ranges,
            self.vertices_list,
            self.converted_centralities,
            self.grid_size
        ) = self.reader.get_data()

        self.state = [
            [0, float('inf'), self.converted_centralities.get("degree centrality", {}).get(v, 0.0)]
            for v in self.vertices_list
        ]
                
        for node, stype in initial_solution.items():
            if node in self.vertices_list:
                if stype == 0:  
                    self.state[node - 1][0] = 1  
                    self.state[node - 1][1] = 0  
                elif stype == 1:  
                    self.state[node - 1][0] = 0  
                    self.state[node - 1][1] = 2  
                elif stype == 2:  
                    self.state[node - 1][0] = 0  
                    self.state[node - 1][1] = 1  

        self.action_space = spaces.Discrete(self.num_vertices)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_vertices, 3), dtype=np.float32)
        self.sensor_costs = {0: 0, 1: 4, 2: 1, 3: 2}  
        self.penalty_invalid = -1000

    def reset(self):
        """
        Reseta o ambiente para o estado inicial.
        """

        return self._get_observation()

    def step(self, action):
        """
        Executa a ação escolhida pelo agente:
        - Aloca ou desaloca um sensor X.
        - Recalcula a conectividade e custo.
        - Aplica a lógica de atualização do estado.
        """
        if action < 0 or action >= self.action_space.n:
            print("ação",action)
            print(f"Fora do intervalo permitido (0, {self.action_space.n - 1}).")

            raise ValueError(f"Ação inválida: {action}. Fora do intervalo permitido (0, {self.action_space.n - 1}).")

        vertex = action
        self.state[vertex][0] = 1 if self.state[vertex][0] == 0 else 0
        self.state[vertex][1] = 0 if self.state[vertex][0] == 1 else float('inf')
        self._update_state()
        cost, solution = self._calculate_cost()
        reward = self._calculate_reward(cost)
        valid = self._is_solution_valid()
        done = not valid
        if not valid:
            reward = self.penalty_invalid  

        return self._get_observation(), reward, cost, solution, done, {}

    def _update_state(self):
        """
        Atualiza o estado do ambiente recalculando os range hops e alocações complementares.
        """
        for v in self.vertices_list:
            v_index = v - 1
            if self.state[v_index][0] == 1:  # Sensor X já alocado
                self.state[v_index][1] = 0  # Range hop = 0
                continue

            # Calcula range hop baseado nos vizinhos
            range_hop = float('inf')
            for neighbor in self.adjacency_list[v]:
                if self.state[neighbor - 1][0] == 1:
                    range_hop = min(range_hop, 1)
            for neighbor in self.second_adjacency_list[v]:
                if self.state[neighbor - 1][0] == 1:
                    range_hop = min(range_hop, 2)
            if range_hop == 1:
                self.state[v_index][1] = 1  # Sensor Z (vizinho de 1º grau)
            elif range_hop == 2:
                self.state[v_index][1] = 2  # Sensor Y (vizinho de 2º grau)
            else:
                self.state[v_index][1] = float('inf')  # Sem cobertura

    def _calculate_cost(self):
        """
        Calcula o custo total da solução e retorna a solução no formato de um dicionário.
        """
        cost = 0.0
        solution = {}  # Dicionário para armazenar a solução no formato {vértice: "tipo sensor"}

        for i, (alloc, range_hop, centrality) in enumerate(self.state):
            if range_hop == 0:  # Sensor X
                cost += self.sensor_costs[1]
                solution[i + 1] = 'X'  # Vértice i+1 recebe o sensor X
            elif range_hop == 1:  # Sensor Z
                cost += self.sensor_costs[2]
                solution[i + 1] = 'Z'  # Vértice i+1 recebe o sensor Z
            elif range_hop == 2:  # Sensor Y
                cost += self.sensor_costs[3]
                solution[i + 1] = 'Y'  # Vértice i+1 recebe o sensor Y
            elif range_hop == float('inf'):  # Vértice sem cobertura
                cost -= self.penalty_invalid
                solution[i + 1] = 'None'  # Nenhum sensor alocado para esse vértice
        return cost, solution

    def _calculate_reward(self, cost):
        """
        Calcula a recompensa como o inverso do custo, com penalização para desconexões.
        """
        disconnected_count = sum(1 for alloc, range_hop, centrality in self.state if range_hop == float('inf'))
        penalty = disconnected_count * 10
        return self.num_vertices / (cost) - penalty
    

    def _is_solution_valid(self):
        """
        Verifica se todos os vértices possuem cobertura.
        """
        for alloc, range_hop, _ in self.state:
            if range_hop == float('inf'):
                # print('self.state invalido', self.state)
                return False
        # print('self.state valido', self.state)
        return True

    def _get_observation(self):
        """
        Retorna o estado como uma matriz (lista de listas).
        """
        return np.array(self.state, dtype=np.float32)


class LinearQNet(Module):

    def __init__(self, learning_rate, input_dims, n_actions):
        super(LinearQNet, self).__init__()
        self.fc1 = Linear(input_dims, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, n_actions)

        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.loss = MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

@dataclass
class Agent:
    learning_rate: float
    gamma: float
    n_actions: int
    n_states: int
    epsilon: float
    eps_min: float
    eps_dec: float
    Q: dict = field(default_factory=dict)
    deep: bool = True

    def __post_init__(self):
        self.init_Q()
        self.action_space = [i for i in range(self.n_actions)]

    def init_Q(self):
        if self.deep:
            self.Q = LinearQNet(self.learning_rate, self.n_states, self.n_actions)
        else:
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    self.Q[(state, action)] = 0.0
    
    def choose_action(self, state, deep=True):
        action = None
        if deep:
            if np.random.random() > self.epsilon:
                state = torch.tensor(state, dtype=torch.float).to(self.Q.device)
                actions = self.Q.forward(state)
                action = T.argmax(actions).item()
            else:
                action = np.random.choice(self.action_space)
        else:
            if np.random.random() < self.epsilon:
                action = np.random.choice([i for i in range(self.n_actions)])
            else:
                actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
                action = np.argmax(actions)

        # Validação da ação
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Ação inválida escolhida pelo agente: {action}.")
        
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, next_state):
        if self.deep:
            self.Q.optimizer.zero_grad()
            states = torch.tensor(state, dtype=torch.float).to(self.Q.device)
            actions = torch.tensor([action], dtype=torch.int).to(self.Q.device)          
            rewards = torch.tensor(reward, dtype=torch.float).to(self.Q.device)
            next_states = torch.tensor(next_state, dtype=torch.float).to(self.Q.device)
            q_pred = self.Q.forward(states)[actions.unsqueeze(0)]
            q_next = self.Q.forward(next_states).max()
            q_target = rewards + self.gamma * q_next
            loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
            loss.backward()
            self.Q.optimizer.step()
        else:
            actions = np.array([self.Q[(next_state, a)] for a in range(self.n_actions)])
            a_max = np.argmax(actions)

            self.Q[(state, action)] += self.learning_rate * (reward +
                                                              self.gamma * self.Q[(next_state, a_max)] -
                                                              self.Q[(state, action)])
        self.decrement_epsilon()

def main_DQN(instance_path, cost, solution, n_episodes=20, max_repeated_actions=10):
    n_games = n_episodes  # Número de episódios (jogos)
    start_time = time.time()
    sensor_mapping = {'X': 0, 'Y': 1, 'Z': 2}
    initial_solution = {vertex: sensor_mapping[sensor] for vertex, sensor in solution.items()}
    print("\n--- [Solução Inicial para RL] ---")
    print(f"Solução Inicial: {initial_solution}")

    env = SensorEnv(instance_path, initial_solution)
    obs = env.reset()
    state = env._get_observation()

    # Número de ações do agente ajustado ao espaço de ação
    n_actions = env.action_space.n

    agent = Agent(
        learning_rate=0.001,
        gamma=0.9,
        epsilon=1,
        eps_min=0.01,
        eps_dec=0.999,
        n_actions=n_actions,
        n_states=state.shape[1],
        deep=True
    )

    # Inicialização dos melhores valores
    best_config = solution.copy()
    best_config_value = 1 / cost  # Inverso do custo inicial como pontuação inicial
    best_config_cost = cost
    best_reward = 0
    last_best_episode = -1

    for i in range(n_games):
        terminated = False
        obs = env.reset()
        score = 0
        repeated_count = 0
        last_action = None
        current_cost = float('inf')
        # print(f"\n[{datetime.now()}] Iniciando Episódio {i}")

        while not terminated:
            try:
                # Escolhe ação usando o agente
                action = agent.choose_action(state)
                # print(f"[DEBUG] Episódio {i}, Ação Escolhida: {action}")
                # Verifica repetição da ação
                if action == last_action:
                    repeated_count += 1
                else:
                    repeated_count = 0

                last_action = action

                # Executa a ação no ambiente
                new_obs, reward, cost_step, sol, done, _ = env.step(action)

                # Atualiza o estado
                new_state = new_obs
                score += reward
                current_cost = cost_step

                # Critério de término: se a solução for inválida
                if done:
                    # print("[DEBUG] Episódio encerrado pois a solução ficou inválida (ou desconexão).")
                    terminated = True

                # Critério de término: repetição excessiva da mesma ação
                if repeated_count >= max_repeated_actions:
                    print(f"[DEBUG] Encerrando episódio por repetição da mesma ação {max_repeated_actions} vezes.")
                    terminated = True
                    # Força uma nova ação diferente da última
                    possible_actions = [a for a in range(n_actions) if a != last_action]
                    action = random.choice(possible_actions)
                    print(f"[DEBUG] Nova ação forçada: {action}")
                    repeated_count = 0  # Reinicia o contador para a nova ação

                new_obs, reward, cost_step, sol, done, _ = env.step(action)
                score += reward
                current_cost = cost_step

                # print(f"[DEBUG] Recompensa: {reward:.4f}, Custo: {cost_step}, Terminado: {terminated}")

                # Aprendizado do agente
                agent.learn(state, action, reward, new_state)

                # Atualiza custos e configurações
                if current_cost < best_config_cost:
                    best_config_cost = current_cost
                    best_config = sol.copy()
                    best_reward = score
                    best_config_value = 1 / current_cost
                    last_best_episode = i
                    print(f"[DEBUG] Melhor custo atualizado: {best_config_cost}")
                    print(f"[DEBUG] Nova Melhor Configuração: {best_config}")

            except ValueError as e:
                print(f"[ERROR] {e}")
                continue  # Ignora ações inválidas e segue o loop

        print(f"[DEBUG] Episódio {i} concluído com custo: {current_cost}")

    # Captura o tempo final e calcula a diferença
    end_time = time.time()
    total_time = end_time - start_time

    # Resultados finais
    print("\n==========================================================")
    print(f"Melhor Pontuação: {best_config_value}")
    print(f"Melhor Configuração: {best_config}")
    print(f"Menor Custo: {best_config_cost}")
    print(f"Tempo Total de Execução: {total_time:.2f} segundos")
    print(f"Episódio com a Melhor Atualização: {last_best_episode}")
    print("==========================================================")
    return best_config_value, best_config, best_config_cost, total_time, last_best_episode

if __name__ == "__main__":
    instance_path = 'instances/regular_grid_324.dat'
    decoder = SAPDecoder(instance_path)
    refiner = SAPDecoderRefiner(instance_path)
    chromosome = [random.random() for _ in range(refiner.n)]
    chromosome = [0.04893717774179001, 0.12275241594060571, 0.07125885541986514, 0.3004385033781368, 0.4586010193078502, 0.18675250057361373, 0.4190657529802115, 0.17114804812659687, 0.3103042597912043, 0.3012684704594461, 0.2895403132499574, 0.38645657769591907, 0.24522928786231063, 0.4135620431315822, 0.41094329115305217, 0.4309749208215627, 0.16357491020647688, 0.12323660852166266, 0.03398163651074496, 0.3544562715537574, 0.07288690814085463, 0.056232419684708476, 0.4880972906677498, 0.45309482495001935, 0.05249862096518351, 0.14965135636917806, 0.6129535677079878, 0.029260096374325806, 0.37458458285861396, 0.3801951705652117, 0.09060826364813479, 0.17350014650362008, 0.3850332726971917, 0.13656222896208034, 0.19939365916021068, 0.23137671563760112, 0.3902622092415645, 0.9018641231003839, 0.06474964280497926, 0.2117139575608349, 0.7537229961072582, 0.4765431325986402, 0.22162863171478164, 0.15069351214928847, 0.01780292130539849, 0.07403830594492222, 0.11987902223050381, 0.33417753546407997, 0.39891979284753876, 0.9356537030895804, 0.05883781113619425, 0.15640557682343934, 0.7237541530861531, 0.3460555137174941, 0.4054132628741864, 0.13963126734472758, 0.1166433544828033, 0.4446483235270961, 0.027666814503337564, 0.2777212041336359, 0.11290068025819203, 0.2753372058722443, 0.07180471633139614, 0.05489383276482102, 0.36987364821834023, 0.11432588007824232, 0.32243221162423696, 0.40311676553105846, 0.03600395902930543, 0.06865553214455586, 0.3751614649341245, 0.10961635628050102, 0.44906718484431624, 0.47034193639853655, 0.4114205905220938, 0.41855653070062737, 0.14938159144990382, 0.37311367502771864, 0.21463648243069605, 0.6677614347725793, 0.35007106585899334, 0.043344563349640645, 0.26554645326835885, 0.13325001361970762, 0.020414349759538195, 0.225761376161689, 0.3362108874956331, 0.061034983134021246, 0.03252329733584742, 0.34774595878998266, 0.0209752799703221, 0.08860369602612826, 0.8334044607910129, 0.3125156797801714, 0.09307384516798747, 0.018466286788900436, 0.32621606690500543, 0.12337114420010505, 0.3198979617438751, 0.23800905642201914, 0.3744114975129786, 0.10158025280917876, 0.9885657518129871, 0.44584467687398127, 0.48264282771549744, 0.07160908506147083, 0.926850939441525, 0.4753847240680344, 0.18729295752149328, 0.4030457926563294, 0.48586484105938293, 0.25248427616747016, 0.4051196760763484, 0.44078811006381863, 0.2921766540376227, 0.15972615653483738, 0.4827898271587855, 0.1385767063272925, 0.030997823880106263, 0.28926147628336446, 0.4159388146047938, 0.47019558431866243, 0.16852261409162797, 0.38563289266292705, 0.04513815886495923, 0.19592103549732207, 0.1006903414740176, 0.04465356040319573, 0.47929052622773816, 0.3375105087694047, 0.3329828183933515, 0.05562937291561778, 0.2661897264674286, 0.1410268254399842, 0.14999758029033297, 0.2607454627373474, 0.007428069564117842, 0.4897047853734015, 0.16872230324300627, 0.47587241077141695, 0.3871045690360895, 0.25330622919825085, 0.36634620052201766, 0.29253506365906334, 0.18005200429431997, 0.4059972508831712, 0.06491386799267534, 0.4483925552913221, 0.47650867626252735, 0.46708935384736233, 0.461305789795189, 0.1186657759178944, 0.5735646342633831, 0.036728021588934276, 0.46184149758793047, 0.37636562525570194, 0.6573239198596166, 0.14268751836930155, 0.05809927441535409, 0.4461056006550829, 0.7702415194073787, 0.3799791167764832, 0.19857519125759013, 0.3628425561264414, 0.22562382536497572, 0.0719572113056011, 0.9184772735693468, 0.31098944552007607, 0.024003078895759214, 0.2007715967719089, 0.4544481944944841, 0.17413180354087981, 0.22324072702770204, 0.1509025621439971, 0.07768533326580705, 0.23112471993799935, 0.4822601662734397, 0.1533260498317951, 0.0483387389013588, 0.3638972720162094, 0.4402225038120571, 0.8443176744917427, 0.054028674030811866, 0.2680641820135305, 0.046113212462993707, 0.41514465947069423, 0.1457946294444164, 0.3405736368788304, 0.19690798056544292, 0.032604369637211215, 0.33624728450864216, 0.12731644274518375, 0.47195137226466977, 0.14772085486044373, 0.016934336831295048, 0.2949705826703623, 0.23810284159180667, 0.4642883716943966, 0.07798448145174151, 0.4824040354669869, 0.03663989959172957, 0.14120930021510278, 0.412418083942841, 0.11249221501645429, 0.12242840327307411, 0.2927778133659222, 0.04939956959502181, 0.07115986093862849, 0.419782706426655, 0.3320081492584635, 0.600324285034125, 0.15011609824153072, 0.2844721018989065, 0.33176816928508485, 0.05046606521219491, 0.24251779461767733, 0.4167659689380326, 0.4575809564411653, 0.4273556244788337, 0.4021107075029395, 0.3489769192165381, 0.3267002961382479, 0.29663473709003135, 0.10291505123123962, 0.3814005194902471, 0.9487521832326998, 0.31684240627163673, 0.05767611591116099, 0.02846683291936762, 0.17341287697752564, 0.40610942123731636, 0.2060251434812691, 0.5693472457456961, 0.346928670872496, 0.3228723656446702, 0.34387710001169036, 0.24197824215741034, 0.003144732872303777, 0.365052982127026, 0.652308743049224, 0.28894978524992593, 0.4472699662581613, 0.14896855416329274, 0.16179281410119106, 0.3469482212231045, 0.3282267859493839, 0.4421892703358389, 0.1726770772314469, 0.47510905375074663, 0.036152007554623375, 0.41103180289876234, 0.2607788867694159, 0.16229768055112961, 0.10304898048855382, 0.18144075359875297, 0.23374259644911996, 0.11364358725035606, 0.005207228912021447, 0.1950722661526424, 0.2250523777510145, 0.39855286402926626, 0.020759346238788487, 0.11123051741032303, 0.098297890745235, 0.4269450379521867, 0.04480291980801098, 0.10466694645818643, 0.1675164665264085, 0.021092872300614098, 0.18339760973979974, 0.2580256747438086, 0.5447381054674476, 0.26138528711914266, 0.4506934564588472, 0.09058941098928339, 0.416064492921136, 0.06171258715097109, 0.19854873308926607, 0.43402466535629775, 0.3238579101651041, 0.9190588247934912, 0.011404794094432798, 0.4555150520104583, 0.12996741298989842, 0.2570915383216987, 0.7579875340929234, 0.11118725387831527, 0.10228710414418661, 0.4699840492447755, 0.16156632514724817, 0.0440203586250398, 0.0300990654496711, 0.4177056763311164, 0.7090114167731947, 0.12381898526920665, 0.30574773386859433, 0.12980803861755152, 0.38292834840364814, 0.011521168993574022, 0.11167166783327163, 0.33181493422153363, 0.06618431127666713, 0.08685474839886206, 0.3800509320615949, 0.1541746186412221, 0.171731812239237, 0.44512821849991185, 0.33131664677018274, 0.22572195498868577, 0.3869436292898657, 0.48892553203738987, 0.40557398748244905, 0.1811060723149842, 0.24970494413740377, 0.4453849279088412, 0.31410808914567084, 0.21241064495572914, 0.1661300265085594, 0.281368826901179, 0.3969650272488297, 0.09016581714047452, 0.4178686400130742, 0.2779710742867103, 0.31611790846594257]
    cost, solution = decoder.decode(chromosome, rewrite=True)
    print("------ Initial Cost:", cost)
    print("------- Initial Solution", solution)
    best_cost, best_solution, rl_time, improved = refiner.refine_solution(chromosome, current_iteration=250, rewrite=True)

    # Resultado
    print("Melhor Custo:", best_cost)
    print("Melhor Solução:", best_solution)
    print("rl_time:", rl_time)
    print("improved:", improved)

    count_X = list(best_solution.values()).count('X')
    count_Y = list(best_solution.values()).count('Y')
    count_Z = list(best_solution.values()).count('Z')
    cost_X = count_X * 4
    cost_Y = count_Y * 2
    cost_Z = count_Z * 1
    total_cost = cost_X + cost_Y + cost_Z

    print(total_cost)