"""
Visualize Genetic Algorithm to find a maximum point in a function.

"""
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy.core.fromnumeric import shape
from tqdm import tqdm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# Clear the terminal window
print("\033c")

VR = np.array([8, 8, 12, 10])    # !Modify: Variables Resolution
POP_SIZE = 200           # !Modify: population size
CROSS_RATE = 0.8         # !Modify: mating probability (DNA crossover)
MUTATION_RATE = 0.003    # !Modify: mutation probability
N_GENERATIONS = 400     # !Modify: Number of iteration
BOUNDs = [[25, 35],        # !Modify: Range of each variable
          [350, 500],
          [50000, 80000],
          [1500, 2700]]

NoV = shape(VR)[0]   # Number of Variables
DNA_L = np.cumsum(VR)   # variable end in chromosome
DNA_F = np.zeros(NoV)   # variable start in chromosome
DNA_F[1:NoV] = DNA_L[0:NoV-1]
DNA_SIZE = sum(VR)      # Total size of chromosome
pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))
pop_best_real = np.random.randint(2, size=(NoV))


# For printing result, since the population is 1, we need a different index for variables
# Result = True: when we are calculating the final result
# Result = False: otherwise
# !Modify: the function and variables need to be modified
# This GA code finds the maximum, so to find the minimum, inverse the function 1/F()
def F(Vars,Result): 
    if Result != True:
        X = np.float32(Vars[:,0])   #mu
        Y = np.float32(Vars[:,1])   #ms
        U = np.float32(Vars[:,2])   #K
        W = np.float32(Vars[:,3])   #C
    else:
        X = np.float32(Vars[0]) #mu
        Y = np.float32(Vars[1]) #ms
        U = np.float32(Vars[2]) #K
        W = np.float32(Vars[3]) #C


    # Constants
    R = 30
    V = 6.5e-6
    Kt = 50000
    # to find the maximum of this function
    return 1/np.sqrt(np.pi*R*V*((Kt*W/(2*Y**3/2*U**1/2))+((X+Y)*U**2/(2*W*Y**2))))


# find non-zero fitness for selection
def get_fitness(F_values): return F_values + 1e-3 - np.min(F_values)

# convert binary DNA for X to decimal and normalize it to a range(0, 5)
def translateDNA(pop): 
    popReal = np.random.randint(2, size=(POP_SIZE, NoV))
    for VariableNo in range(NoV):
        popVar = np.array(pop[:, np.intc(DNA_F[VariableNo]):np.intc(DNA_L[VariableNo])])
        popRealTemp = popVar.dot(2 ** np.arange((VR[VariableNo]))[::-1]) / float(2**(
            VR[VariableNo])-1) * (BOUNDs[VariableNo][1]-BOUNDs[VariableNo][0]) + BOUNDs[VariableNo][0]
        popReal[:,VariableNo] = popRealTemp
    return popReal

def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx] 

def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent

def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


#######################
# plt.ion()       # Turn the interactive plotting mode on
# X = np.linspace(0, 5, 100)     
# Y = np.linspace(0, 5, 100)     
# X, Y = np.meshgrid(X, Y) 

# Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
# fig = pyplot.figure() 
# ax = Axes3D(fig)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#   cmap=cm.nipy_spectral, linewidth=0.08,
#   antialiased=True)    
#########################

# Turn the interactive plotting mode on
plt.ion() 
x = np.linspace(0, N_GENERATIONS, N_GENERATIONS)
#plt.plot(x, F(x))

for i in tqdm(range(N_GENERATIONS)):
    # compute function value by extracting DNA
    F_values = F(translateDNA(pop),False)

    # sca = ax.scatter(translateDNA_X(pop), translateDNA_Y(pop), F_values, s=100, c='black'); plt.pause(0.1)
    # Following if statement is to remove the points on plotted curve in previous generation
    # if i < (N_GENERATIONS - 1): sca.remove()   
    # GA part (evolution)
    fitness = get_fitness(F_values)
    pop_best_DNA = pop[np.argmax(fitness), :]
    # something about plotting
    sca = plt.scatter(i, 1/F_values[np.argmax(fitness)], s=20, lw=1, c='red', alpha=0.5)
    plt.pause(0.05)
    #if i < (N_GENERATIONS - 1):
    #sca.remove()


    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child

    # ! Uncomment this if you want to always carry your best Chromosome to next generation
    # Saving the best Chromosome
    #pop[POP_SIZE-1, :] = pop_best_DNA


#Calculate and display the optimal result 
pop_best_DNA = pop[np.argmax(fitness),:]
pop_best_real = translateDNA(pop)[np.argmax(fitness), :]

print("Best real Parameter values: ", pop_best_real)
#print("Function value for Minimization: ", 1/F(np.transpose(pop_best_real), True))
#print("Function value for Maximization: ", F(np.transpose(pop_best_real), True))
print("Objective Function value: ", 1/F(np.transpose(pop_best_real), True))

plt.ioff()  # Turn the interactive plotting mode off
plt.show()

