import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
from sklearn import metrics

from pyswarm import pso  # optimisation library

"""
functions to transform 1-array into triangular inf matrix  and vice versa 
not interresting functions : just to be adapted variables (vector) to PSO library
"""


def i2j_to_k(i, j):
    return i * (i - 1) / 2 + j


def k_to_i2j(k):
    i = int((1 + math.sqrt(1 + 8 * k)) / 2)
    j = int(k - i * (i - 1) / 2)
    return (i, j)


def d1_2_mat(d):
    l = len(d)
    (n, _) = k_to_i2j(l)
    # print(n)
    mat = np.zeros([n, n])
    for k in range(l):
        (i, j) = k_to_i2j(k)
        mat[i, j] = d[k]
        mat[j, i] = -d[k]
    return mat


def mat_2_d1(mat):
    (n, _) = np.shape(mat)
    n -= 1
    k_max = int(n * (n - 1) / 2 + n)
    liste = []
    for k in range(k_max):
        (i, j) = k_to_i2j(k)
        liste.append(mat[k_to_i2j(k)])

    return np.array(liste)


def d1_2_mat_list(d, n_Item):
    item_list = np.reshape(np.array(d), (n_Item, -1))
    l = len(item_list[0])

    (n, _) = k_to_i2j(l)

    mat = np.zeros([n_Item, n, n])

    for it in range(n_Item):

        for k in range(l):
            (i, j) = k_to_i2j(k)
            mat[it, i, j] = item_list[it, k]
            mat[it, j, i] = -item_list[it, k]
    return mat


def mat_2_d1_list(mat):
    liste_totale = []
    (n_item, n, _) = np.shape(mat)
    n -= 1
    k_max = int(n * (n - 1) / 2 + n)
    for it in range(n_item):
        liste = []
        for k in range(k_max):
            (i, j) = k_to_i2j(k)
            liste.append(mat[it, i, j])

        liste_totale += liste

    return np.array(liste_totale)


def detection_cooperation_A(env, id_agent_source, last_offer):  # ratio other agent gave / what I could give
    n_agents = env.n_agents
    agent = env.agents[id_agent_source]
    history = env.transactions_history_numpy
    if len(history) < 2:
        return np.zeros(n_agents)
    else:
        # print(history[-2])
        last_trans = history[-2].sum(1)  # what each agent gave
        # last_offer = agent.last_offers[-1]
        my_offer_max = np.sum(np.maximum(last_offer, 0))

        # print("last_trans", last_trans)
        coop_degrees = np.clip(last_trans / my_offer_max, 0, 1)

        return coop_degrees


def detection_cooperation_B(env, id_agent_source,
                            last_offer):  # ratio other agent gave ME * (N_agent-1) / what I could give
    n_agents = env.n_agents
    agent = env.agents[id_agent_source]
    history = env.transactions_history_numpy
    if len(history) < 2:
        return np.zeros(n_agents)
    else:
        # print(history[-2])
        last_trans = history[-2][:, id_agent_source] * (n_agents - 1)  # what each agent gave
        # last_offer = agent.last_offers[-1]
        my_offer_max = np.sum(np.maximum(last_offer, 0))

        # print("last_trans", last_trans)
        coop_degrees = np.clip(last_trans / my_offer_max, 0, 1)

        return coop_degrees


def TFT(alpha, r, beta=0):  # improved Tit-for-Tat with parameters alpha, r and beta
    def function(old_coop_degrees, detected_coop_degrees, r):
        delta = detected_coop_degrees - old_coop_degrees
        r = np.maximum(r + beta * delta, 0)
        output = alpha * old_coop_degrees + (1 - alpha) * (r + (1 - r) * detected_coop_degrees)
        return output, r

    return function, r


class Agent:
    def __init__(self, id_agent, n_agents, n_items, neg_algo=0):
        self.n_agents = n_agents
        self.n_items = n_items
        self.id_agent = id_agent
        self.old_coop_degrees = np.zeros(self.n_agents)
        self.ut_function = []  # history of local utiliy function
        self.last_offers = []
        tft_algo, r = neg_algo
        self.negociation_algo = tft_algo
        self.r = r * np.ones(self.n_agents)

    def coop_detection(self, env):
        n_agents = env.n_agents
        id_agent_source = self.id_agent
        if len(self.last_offers) == 0:
            return np.zeros(n_agents)
            # print("non last offers")
        else:
            # print("presence last offers")
            last_offer = self.last_offers[-1]
            output = detection_cooperation_B(env, id_agent_source, last_offer)
            return output

    def offer(self, env):  # compute offer/demand (what he needs and what he can offer)
        tran = env.optimize_localy(self.id_agent)
        self.last_offers.append(tran)
        return tran

    def negociation(self, detected_coop_degrees):  # compute new cooperation degrees
        output, r_new = self.negociation_algo(self.old_coop_degrees, detected_coop_degrees, self.r)
        self.r = r_new
        self.old_coop_degrees = output
        return output


class Environment:
    def __init__(self, n_agents, n_items, list_agents=[]):
        self.n_agents = n_agents
        self.agents = list_agents  # list of Agent objects
        self.n_items = n_items
        self.t = 0  # step
        self.state = np.zeros([n_agents, n_items])  # state is the ressources of all agents
        self.states_history = []  # history of states
        self.transactions_history = []
        # a transaction is [step t, id agent source, id agent target, id item, quantity ]
        self.transactions_history_numpy = []

        self.optimal_SW = 0  # optimal social welfare
        self.hist_SW = []  # evolution of social welfare
        self.hist_ut_agents = [[] for _ in range(self.n_agents)]  # evolution of utilities of agents

        self.hist_coop_degrees = []  # evolution of cooperation degrees

    def init_state(self, state):
        s = np.copy(state)
        self.state = s
        self.t = 0

    def next_round(self):
        self.t += 1
        self.transactions_history.append([])
        self.transactions_history_numpy.append(np.zeros([self.n_agents, self.n_agents]))

    def replace_agents(self, list_agents):
        self.agents = list_agents

    def clip_state(self, lb, ub):  # clipping states (in case of math conflict : e.g. log or power)
        self.state = np.clip(self.state, lb, ub)

    def random_init(self, mean=0, std=1):
        self.state = np.random.normal(mean, std, size=[self.n_agents, self.n_items])

    def transaction(self, id_agent_source, id_agent_target, id_item, quantity):
        self.state[id_agent_target, id_item] += quantity
        self.state[id_agent_source, id_item] -= quantity

        self.transactions_history[-1].append((self.t, id_agent_source, id_agent_target, id_item, quantity))

        # update quantities shared between agents (independently of items)
        self.transactions_history_numpy[-1][id_agent_source, id_agent_target] += quantity

    def add_transactions_np(self, state, transactions):  # change state with numpy format
        # state : array nA x nI
        # transactions : array nI x nA x nA
        (nA, nI) = np.shape(state)
        new_s = state.copy()
        for item in range(nI):
            for agent in range(nA):
                new_s[agent, item] -= transactions[item, agent, :].sum()

        return new_s

    def add_transactions_var(self, state, trans_var):  # change state with PSO library variable format
        # state : array nA x nI
        # trans_var : variable for optimisation, liste
        transactions = d1_2_mat_list(trans_var, self.n_items)
        return self.add_transactions_np(state, transactions)

    def global_utility(self, state, lb=-2.0, ub=100.0):  # compute global utility
        s = np.copy(state)
        s = np.clip(s, lb, ub)
        return -np.log(s + 2 + 1e-8).sum()

    def local_utility(self, state, id_agent):  # compute only local utility of id_agent
        s = np.copy(state)
        s = s[id_agent, :]
        s = np.clip(s, -2, 100)
        return -np.log(s + 2 + 1e-8).sum()

    def optimize_localy(self, id_agent, lb=-4, ub=4, min_cons=-1):
        """
        :param id_agent:
        :param lb: lower bound resource to give
        :param ub: upper bound resource to receive
        :param min_cons:
        :return:
        """
        (nA, nI) = self.n_agents, self.n_items
        size_var = nI  # size of variable
        lb_list = lb * np.ones(size_var)  # lower bounds
        ub_list = ub * np.ones(size_var)  # upper bounds

        s_tmp = np.copy(self.state)

        def f_opt(dx):  # utility function for PSO library
            s = np.copy(s_tmp)
            s[id_agent, :] += dx
            # local utily + penality for higher transactions
            return self.local_utility(s, id_agent) + 0.01 * np.linalg.norm(dx)

        def constraint(dx):  # defining constraints for PSO
            s = np.copy(s_tmp)
            s[id_agent, :] += dx
            s = s[id_agent]
            const_out1 = s - min_cons  # minimum of state ressources
            const_out2 = np.array([-dx.sum()])  # not giving more than having
            const_out = np.concatenate((const_out1, const_out2))

            return const_out

        xopt, fopt = pso(f_opt, f_ieqcons=constraint, lb=lb_list, ub=ub_list, maxiter=200, swarmsize=200)

        return (xopt)

    def optimize_globably(self, lb=-2.0, ub=2.0):
        (nA, nI) = self.n_agents, self.n_items
        k_max = nI * (int((nA - 1) * (nA - 2) / 2 + nA - 1))  # size of optimisation variable
        d_var = np.zeros(k_max)  # global variable for PSO library
        lb_list = lb * np.ones(k_max)  # lower bounds
        ub_list = ub * np.ones(k_max)  # upper bounds

        s = self.state.copy()

        def f_opt(dx):
            trans_var_np = d1_2_mat_list(dx, self.n_items)  # convert PSO variable into numpy variable
            new_s_tmp = self.add_transactions_np(s, trans_var_np)

            fusion_items = trans_var_np.sum(0)
            received_agents = fusion_items.sum(1)

            # sum of local utilies + penality for inequalities between agents + penality for higher transactions
            return self.global_utility(new_s_tmp) + 0.1 * np.linalg.norm(received_agents) + 0.1 * np.linalg.norm(dx)

        # def constraint(dx):
        #     trans_var_np = d1_2_mat_list(dx, self.n_items)
        #     new_s_tmp = self.add_transactions_np(s, trans_var_np)
        #
        #     const_out = new_s_tmp - min_cons
        #     # const_out2 = np.array([-dx.sum()])
        #     # const_out = np.concatenate((const_out1,const_out2))
        #
        #     return const_out

        xopt, fopt = pso(f_opt, lb_list, ub_list, maxiter=300, swarmsize=300)

        transactions = d1_2_mat_list(xopt, self.n_items)
        new_s = self.add_transactions_np(s, transactions)
        # print(new_s)

        return (transactions, new_s, self.global_utility(new_s))

    def optimal_social_welfare(self):  # update optimal social welfare
        (transactions, new_s, fopt) = self.optimize_globably()
        self.optimal_SW = fopt
        return fopt

    def get_observation(self, id_agent):
        return self.state[id_agent, :]

    def allocation(self, coop_degrees, demands):
        # according to cooperation degree and demands, compute the allocation (according to the paper)
        demands_agents = np.maximum(demands, 0)  # positive values of demands -> what agents need
        offers_agents = -np.minimum(demands, 0)  # - negative values of demands -> what agents can offer

        for it in range(self.n_items):
            for agent_source in range(self.n_agents):
                of_source = offers_agents[agent_source, it]  # what can offer agent_source for item it

                if of_source > 0:  # agent_source can give of_source for item it

                    demands_targets = np.zeros([self.n_agents])
                    parts_targets = np.zeros([self.n_agents])

                    for agent_target in range(self.n_agents):
                        dem_target = demands_agents[agent_target, it]  # demand of agent_target if < 0
                        dem_target_clip = min(dem_target, of_source)
                        demands_targets[agent_target] = dem_target_clip
                        parts_targets[agent_target] = dem_target_clip

                    total_demand = demands_targets.sum()

                    for agent_target in range(self.n_agents):

                        # according to our allocation rule
                        alloc = demands_targets[agent_target] * coop_degrees[agent_source, agent_target] * of_source

                        if total_demand != 0:
                            alloc /= total_demand
                        alloc = min(alloc, of_source)

                        self.transaction(agent_source, agent_target, it, alloc)

    def show(self, ylim_lb=-1, ylim_ub=1):
        fig, axs = plt.subplots(1, self.n_agents, figsize=(10, 10))
        x = np.arange(1, self.n_items + 1)
        for i in range(self.n_agents):
            axs[i].bar(x, self.get_observation(i), orientation='vertical')
            # axs[i].axis('equal')
            axs[i].set_title("Agent " + str(i + 1))
            axs[i].set_ylim([ylim_lb, ylim_ub])


# env = Environment(3,4, liste_agents_A)

def episode(env, verbose=1):
    # one step :
    # 1. compute offers and demands
    # 2. estimate cooperation degrees
    # 3. compute new cooperation degrees
    # 4. allocate resources according to cooperation degrees and demands

    n_agents = env.n_agents
    n_items = env.n_items
    opt_sw = env.optimal_social_welfare()
    env.next_round()

    env.states_history.append(env.state)
    current_SW = env.global_utility(env.state)
    env.hist_SW.append(current_SW)

    for i_A in range(n_agents):
        # compute local utilies for history (and then for the curves)
        uti_agent = env.local_utility(env.state, i_A)
        env.hist_ut_agents[i_A].append(uti_agent)

    env.show()

    demands = np.zeros([n_agents, n_items])

    coop_degrees = np.zeros([n_agents, n_agents])

    for i_agent in range(n_agents):
        transa = env.agents[i_agent].offer(env)
        env.agents[i_agent].last_offers.append(transa)
        # print(transa)

        if verbose == 1:
            print("Agent ", i_agent)
        coop_deg_detected = env.agents[i_agent].coop_detection(env)
        if verbose == 1:
            print("coop deg detected ", coop_deg_detected)

        coop_deg_i = env.agents[i_agent].negociation(coop_deg_detected)
        if verbose == 1:
            print("coop deg negociated ", coop_deg_i)
            print()

        demands[i_agent, :] = transa
        coop_degrees[i_agent, :] = coop_deg_i

    coop_degrees = np.clip(coop_degrees, 0, 1)

    print(coop_degrees)
    env.hist_coop_degrees.append(coop_degrees)

    env.allocation(coop_degrees, demands)


def mean_coop_degrees(matrix_list):
    if matrix_list == []:
        return []
    else:
        (n_A, _) = np.shape(matrix_list[0])
        output = [[[] for _ in range(n_A)],
                  [[] for _ in range(n_A)]]  # curves for mean receiving AND sending coop degree
        for coop_degrees_mat in matrix_list:
            rece_coop = coop_degrees_mat.sum(0)
            send_coop = coop_degrees_mat.sum(1)
            for i_A in range(n_A):
                mean_rece = (rece_coop[i_A] - coop_degrees_mat[i_A, i_A]) / (n_A - 1)
                output[0][i_A].append(mean_rece)  # receiving coop degree mean for agent i_A

                mean_send = (send_coop[i_A] - coop_degrees_mat[i_A, i_A]) / (n_A - 1)
                output[1][i_A].append(mean_send)  # receiving coop degree mean for agent i_A

        return output


def figure_utilities(sw, list_ut, output_fig, max_t, lu=2, uu=4):
    colors = ['b', 'm', 'c', 'r']
    t_max = min(len(sw), max_t)
    t = np.arange(t_max)

    fig, ax1 = plt.subplots()

    color = 'b'
    ax1.set_xlabel('Rounds', fontsize=14)
    ax1.set_ylabel('Social Welfare', color='g', fontsize=14)
    ax1.plot(t, sw[:t_max], color='g', label="Social Welfare")
    ax1.tick_params(axis='y', labelcolor='g', labelsize=14)
    plt.legend(loc=2, fontsize=13)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Individual Utility', color=color, fontsize=14)
    ax2.set_ylim(lu, uu)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)

    for i_A in range(len(list_ut)):
        color = colors[i_A]
        label = "Agent " + str(i_A + 1)
        if i_A == 5:
            label = "Egoist"
        ax2.plot(t, list_ut[i_A][:t_max], color=color, label=label)

    plt.legend(loc=4, fontsize=13)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(output_fig)
    plt.show()


def figure_coop_degrees_mean(list_coop, output_fig, max_t, lc=0, uc=1):
    colors = ['b', 'm', 'c', 'g', 'r']

    t_max = min(len(list_coop[0][0]), max_t)
    t = np.arange(t_max)

    fig, ax1 = plt.subplots()

    color = 'b'
    ax1.set_xlabel('Rounds', fontsize=14)
    ax1.set_ylabel('Mean Cooperation Degree', fontsize=14)

    ax1.set_ylim(lc, uc)

    ax1.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)

    for i_A in range(len(list_coop[0])):
        color = colors[i_A]
        if i_A != 5:
            label = "Agent " + str(i_A + 1)
        else:
            label = "Egoist"
        ax1.plot(t, list_coop[0][i_A][:t_max], color, label=label + " : receiving")
        ax1.plot(t, list_coop[1][i_A][:t_max], color + "--", label=label + " : sending")
        plt.legend(loc=4, fontsize=13)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(output_fig)
    plt.show()


def affiche(env, output, lu, uu, lc, uc):
    yA = env.hist_ut_agents[0]
    yB = env.hist_ut_agents[1]
    yC = env.hist_ut_agents[2]
    # yD = env.hist_ut_agents[3]
    y = env.hist_SW

    yA = [-x for x in yA]
    yB = [-x for x in yB]
    yC = [-x for x in yC]
    # yD = [-x for x in yD]
    y = [-x for x in y]

    mean_coop_degrees_expe = mean_coop_degrees(env.hist_coop_degrees)

    figure_utilities(y, [yA, yB, yC], 'evolution_utilities_' + output + '.svg', 10, lu, uu)

    figure_coop_degrees_mean(mean_coop_degrees_expe, 'evolution_cooperation_' + output + '.svg', 10, lc, uc)


def effiency(list_SW, SWopt):
    SW0 = list_SW[0]
    output = []

    for sw in list_SW:
        e = min((-sw + SW0) / (-SWopt + SW0), 1)
        output.append(e)

    return output


def speed(list_eff, T):
    final_efficiency = list_eff[-1]
    x = np.arange(T)
    return metrics.auc(x, list_eff[:T]) / (T * final_efficiency)


def analyse(state, beta, r, alpha, N_iter=10, verbose=1):
    # general function for run a game with parameters beta, r, alpha
    # and compute metrics : efficiency, speed, incentive-compatibility and safety

    algo = TFT(beta, r, alpha)
    egoist = TFT(1, 0, 0.5)

    n_A, n_I = np.shape(state)

    T_speed = 10

    T_speed = min(T_speed, N_iter)

    # Environments A,B and C have respectively 0, 1 and nA-1=2 egoists
    # these three simulations are for computing metrics safety and IC
    liste_agents_A = []
    liste_agents_B = []
    liste_agents_C = []
    for i in range(n_A):
        liste_agents_A.append(Agent(i, n_A, n_I, algo))

        if i == n_A - 1:
            liste_agents_B.append(Agent(i, n_A, n_I, egoist))
        else:
            liste_agents_B.append(Agent(i, n_A, n_I, algo))

        if i >= n_A - 2:
            liste_agents_C.append(Agent(i, n_A, n_I, egoist))
        else:
            liste_agents_C.append(Agent(i, n_A, n_I, algo))

    envA = Environment(n_A, n_I, liste_agents_A)
    envA.init_state(state)

    envB = Environment(n_A, n_I, liste_agents_B)
    envB.init_state(state)

    envC = Environment(n_A, n_I, liste_agents_C)
    envC.init_state(state)

    tra, ns, swopt = envA.optimize_globably()

    for i_ep in range(N_iter):
        print("Env A, episode ", i_ep)
        episode(envA, verbose)

    for i_ep in range(N_iter):
        print("Env B, episode ", i_ep)
        episode(envB, verbose)

    for i_ep in range(N_iter):
        print("Env C, episode ", i_ep)
        episode(envC, verbose)

    effic = effiency(envA.hist_SW, swopt)

    ef = effic[-1]

    sp = speed(effic, T_speed)

    ut_max_pers_nices = -1 * envA.hist_ut_agents[2][-1]  # positive
    ut_pers_3egoist = -1 * envA.hist_ut_agents[0][0]  # positive
    ut_pers_1egoist = -1 * envB.hist_ut_agents[2][-1]  # positive
    ut_pers_2egoist = -1 * envC.hist_ut_agents[0][-1]  # positive

    print(ut_max_pers_nices, ut_pers_1egoist, ut_pers_2egoist, ut_pers_3egoist)

    ic = (ut_max_pers_nices - ut_pers_1egoist)
    sf = (ut_pers_2egoist - ut_pers_3egoist)

    return [[beta, r, alpha], [envA, envB, envC], ef, sp, ut_max_pers_nices, ut_pers_1egoist, ut_pers_2egoist,
            ut_pers_3egoist]


def all_expe(liste_bra, state_E):
    output = []
    for x in liste_bra:
        beta, r, alpha = x
        N_iter = 15
        a = analyse(state_E, beta, r, alpha, N_iter)
        output.append(a)

    return output
