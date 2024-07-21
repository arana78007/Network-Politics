import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import sympy as sp
import scipy
from sympy import Function, lambdify, Derivative, latex
from scipy.integrate import odeint



def inhomogenousEr(n,p,r):
    size1 = n
    G = nx.DiGraph()
    nodes_community_1 = range(size1)  
    nodes_community_2 = range(size1,size1+size1)  
    G.add_nodes_from(nodes_community_1, community=1)  
    G.add_nodes_from(nodes_community_2, community=2)
    #The above adds the nodes to the network
    for i in range(size1+size1):
        for j in range(i+1,size1+size1):
            if i==j:
                continue
            #no self loops
            else:
                if i<size1 and j<size1:
                    random1 = random.random()
                    random2 = random.random()
                    if p[0]>random1:
                        G.add_edge(i,j,color='lightblue',weight = 1)
                    if p[0]>random2:
                        G.add_edge(j,i,color='lightblue',weight = 1)
                    #Independently add links inside community 1 and gives a colour
                elif i<size1 and j>=size1 :
                    random1 = random.random()
                    random2 = random.random()
                    if i==0 and j==n:
                        G.add_edge(i,j,color='darkslategray',weight = r)
                        G.add_edge(j,i,color='darkslategray',weight = r)
                    #THIS IS THE LEADER CONNECTION CONDITION
                    else:
                        if p[1]>random1:
                            G.add_edge(i,j,color='darkslategray', weight = r)
                        if p[1]>random2:
                            G.add_edge(j,i,color='darkslategray',weight = r)
                    #Independently adds links between community 1 and 2 with weight r and gives a colour
                else:
                    random1 = random.random()
                    random2 = random.random()
                    if p[0]>random1:
                        G.add_edge(i,j,color='tomato', weight = 1)
                    if p[0]>random2:
                        G.add_edge(j,i,color='tomato', weight = 1)
                    #Independently adds links inside community 2, and gives a colour
    return G
#Function iterates over upper triangle ONLY of adjacency as to min iterations no, and sets links both ways independently to do other half.


x = inhomogenousEr(25,[0.25,0.05],0.5)
colors = nx.get_edge_attributes(x,'color').values()
node_colors = ['black' if x.nodes[node]['community'] == 1 else 'grey' for node in x.nodes()]
#sets node colours
for u, v, data in x.edges(data=True):
    data['weight'] *= 2
#to scale thickness of edges according to weight
weights = [data['weight'] for u,v, data in x.edges(data=True)]
pos = nx.spectral_layout(x)
#calculates to spectral layout of nodes

nx.draw(x , pos, node_size=25, node_color=node_colors, edge_color = colors, width = weights)
ax = plt.gca()
ax.margins(0.20)
plt.title(label="Example Graph")
plt.show()
# Above draws the graph


def ZeroOneMatrix(n):
    block_zeros = np.zeros((n, n),dtype = int)
    block_ones = np.ones((n, n),dtype = int)
    matrix_np = np.block([[block_zeros, block_ones], [block_ones, block_zeros]])
    zeroone = sp.Matrix(matrix_np)
    return zeroone
#Creates the matrix X used in the solution.

R = sp.Symbol('R')
zeta = sp.Symbol('Z')
t = sp.Symbol('t')
a = sp.Symbol('a')
alpha1 = sp.Symbol('al_1')
alpha2 = sp.Symbol('al_2')
#makes the sympy symbols we will use
           
def Utility(n,evidence):
    neta = sp.Matrix([evidence for _ in range(2*n)])
    sigma = sp.Matrix([Function(f'x_{i}')(t) for i in range(2*n)])
    rows = 2*n
    cols = 2*n
    W = sp.Matrix(2*n, 2*n, lambda i, j: Function(f'W_{i}_{j}')(t) if i != j else 0)
    com1_sigma = [sigma[i] for i in range(n)]
    com2_sigma = [sigma[i] for i in range(n,(2*n))]
    com1_sum = sp.Add(*com1_sigma)
    com2_sum = sp.Add(*com2_sigma)
    U = neta.dot(sigma)-(R/2)*(sigma.dot(sigma))+zeta*(((sigma.T)*W).dot(sigma)) + (a/((W*ZeroOneMatrix(n)).trace()))*(sp.Add(com1_sum,-1*com2_sum))   
    return U
#above just calculates a symbolic utility function for a fixed level of evidence

def gradopn(utility,n,bool):
    sigma = sp.Matrix([Function(f'x_{i}')(t) for i in range(2*n)])
    derivatives = sp.diff(utility, sigma)
    #Takes derivative wrt the vector
    ddt = sp.diff(sigma, t)
    if bool == True:
        gradlearn = sp.Add(alpha1*derivatives,-ddt) 
    else:
        gradlearn = alpha1*derivatives 
    return gradlearn
#Assign boolean depending on if you want the differntial term or not

print(sp.latex(gradopn(Utility(1,1),1,True)))

def gradweight(utility,n,bool):
    sigma = sp.Matrix([Function(f'x_{i}')(t) for i in range(2*n)])
    W = sp.Matrix(2*n, 2*n, lambda i, j: Function(f'W_{i}_{j}')(t) if i != j else 0)
    com1_sigma = [sigma[i] for i in range(n)]
    com2_sigma = [sigma[i] for i in range(n,(2*n))]
    com1_sum = sp.Add(*com1_sigma)
    com2_sum = sp.Add(*com2_sigma)
    firstterm = sp.Matrix((zeta*sp.tensorproduct(sigma,sigma)).reshape(2*n,2*n))
    #calulates tensor product and shapes it back into matrix foorm from the list output
    secondterm = (-a*(sp.Add(com1_sum,-1*com2_sum))/((W*ZeroOneMatrix(n)).trace())**2)*(ZeroOneMatrix(n))
    #calculates the devation/convergence preference term derivative
    ddt = sp.diff(W,t)
    temp = alpha2*(firstterm+secondterm)
    for i in range(temp.rows):
        temp[i, i] = 0
        #set diagnonals to zero as no self loops
    if bool == True:
        gradlearn = temp - ddt
    else:
        gradlearn = temp
    return gradlearn
#Assign boolean for differntial
print(sp.latex(gradweight(Utility(1,1),1,True)))


def symbolicsolattempt(utility,n):
    t = sp.Symbol('t')
    eqns = gradopn(utility,n,True).tolist() + gradweight(utility,n,True).tolist()
    flattened_eqns = [item for eqn in eqns for item in eqn]
    filtered_eqns = [x for x in flattened_eqns if x != 0]
    #Makes list of equations without 0's 
    sigma = sp.Matrix([Function(f'x_{i}')(t) for i in range(2*n)]).tolist()
    W = sp.Matrix(2*n, 2*n, lambda i, j: Function(f'W_{i}_{j}')(t) if i != j else 0).tolist()
    vars = sigma + W
    flattened_vars = [item for var in vars for item in var]
    filtered_vars = [x for x in flattened_vars if x != 0]
    #does the same for variables
    sol = sp.solvers.ode.systems.dsolve_system(filtered_eqns,filtered_vars,t)
    #attempts solution
    return sol

#symbolicsolattempt(Utility(1,1),1)
#this stops the code, hence commented and numerical solutions are found instead

time = np.linspace(0, 40, 101)
#defines time period we are solving over
     
def numericalsolution(intopn,utility,n,p,relweight,Z1,a1,R1,alp):
    eqns = gradopn(utility,n,False).tolist() + gradweight(utility,n,False).tolist()
    flattened_eqns = [item for eqn in eqns for item in eqn]
    #makes list of equations removing null equations
    sigma = sp.Matrix([Function(f'x_{i}')(t) for i in range(2*n)]).tolist()
    W = sp.Matrix(2*n, 2*n, lambda i, j: Function(f'W_{i}_{j}')(t) if i != j else 0).tolist()
    vars = sigma + W
    flattened_vars = [item for var in vars for item in var]
    filtered_vars = [x for x in flattened_vars if x != 0]
    #makes list of variables the same ways as eqns removing null variables (0's on diagonal)
    function_list_subs = []
    for func in flattened_eqns:
        if func.has(alpha1):
            function_list_subs.append(func.subs({zeta:Z1,alpha1:alp,R:R1,a:a1}))
            continue
        elif func.has(alpha2):
            function_list_subs.append(func.subs({zeta:Z1,alpha2:alp,a:a1}))
    #substitute constants into each equation
    adjacencyinitalcond = nx.adjacency_matrix(inhomogenousEr(n,p,relweight))
    listintconopn = [intopn for _ in range(2*n)]
    listintcon = (adjacencyinitalcond.toarray()[~np.eye(adjacencyinitalcond.toarray().shape[0],dtype=bool)].reshape(adjacencyinitalcond.toarray().shape[0],-1)).flatten()
    #lists the intial conditions for the weights that is preappended with the opinion inital conditions below
    for i, element in enumerate(listintconopn):
        listintcon = np.insert(listintcon, i, element)
    floatintcon =  [float(entry) for entry in listintcon]
    expr_without_t = sp.sympify(str(function_list_subs).replace("(t)","")) 
    vars_without_t = sp.sympify(str(filtered_vars).replace("(t)",""))
    #remove function notation for scipy library
    lambdified_functions =  [lambdify((vars_without_t), equations) for equations in expr_without_t]
    #now lamdify to convert symobolic expression into something odeint can handle
    numsol = odeint(lambda y, time: [func(*y) for func in lambdified_functions], floatintcon,time)
    return numsol
  
simulationno = 20
#adjust this as you please
#the rest of the code simply loops and averages the solution array which is then plotted
#note that the last two graphs are done first, then the final two are done.
sol1list = []
sol2list = []
sol3list = []
sol4list = []


k = 0
while k < simulationno:
        sol = numericalsolution(1,Utility(5,1),5,[0.35,0.15],0.8,0.05,0.50,1.2,0.1)
        sol3list.append(sol)
        k+=1
sol3 = np.mean(sol3list,axis=0)

plt.figure(figsize=(8, 6))
for i in range(1,5):
    plt.plot(time, sol3[:, i], label=f'Variable {i+1}', color='blue')
    plt.plot(time, sol3[:, 5+i], label=f'Variable {i+1}', color='red')
plt.show()

plt.figure(figsize=(8, 6))
for i in range(10,100):
    plt.plot(time, sol3[:, i], label=f'Variable {i+1}', color='purple')
plt.show()

o = 0
while o < simulationno:
        sol = numericalsolution(1,Utility(5,1),5,[0.35,0.15],2.5,0.05,0.50,1.2,0.1)
        sol4list.append(sol)
        o+=1
sol4 = np.mean(sol4list,axis=0)

plt.figure(figsize=(8, 6))
for i in range(1,5):
    plt.plot(time, sol4[:, i], label=f'Variable {i+1}', color='blue')
    plt.plot(time, sol4[:, 5+i], label=f'Variable {i+1}', color='red')
plt.show()

plt.figure(figsize=(8, 6))
for i in range(10,100):
    plt.plot(time, sol4[:, i], label=f'Variable {i+1}', color='purple')
plt.show()

i = 0
while i < simulationno:
        sol = numericalsolution(1,Utility(5,1),5,[0.35,0.075],0.8,0.05,2,1.2,0.05)
        sol1list.append(sol)
        i+=1
        
sol1 = np.mean(sol1list,axis=0)

plt.figure(figsize=(8, 6))
for i in range(1,5):
    plt.plot(time, sol1[:, i], label=f'Variable {i+1}', color='blue')
    plt.plot(time, sol1[:, 5+i], label=f'Variable {i+1}', color='red')
plt.show()

plt.figure(figsize=(8, 6))
for i in range(10,100):
    plt.plot(time, sol1[:, i], label=f'Variable {i+1}', color='purple')
plt.show()

j = 0
while j < simulationno:
        sol = numericalsolution(1,Utility(5,1),5,[0.35,0.15],0.8,0.05,2,1.2,0.05)
        sol2list.append(sol)
        j+=1
sol2 = np.mean(sol2list,axis=0)

plt.figure(figsize=(8, 6))
for i in range(1,5):
    plt.plot(time, sol2[:, i], label=f'Variable {i+1}', color='blue')
    plt.plot(time, sol2[:, 5+i], label=f'Variable {i+1}', color='red')
plt.show()

plt.figure(figsize=(8, 6))
for i in range(10,100):
    plt.plot(time, sol2[:, i], label=f'Variable {i+1}', color='purple')
plt.show()