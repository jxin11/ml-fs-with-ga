import random
import numpy as np
from pandas import read_csv
from deap import base, creator, tools, algorithms
from elitismFunction_Guards import eaSimpleWithElitism
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

class Zoo:
    """This class encapsulates the Friedman1 test for a regressor
    """

    DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data'
    NUM_FOLDS = 5

    def __init__(self, randomSeed):
        """
        :param randomSeed: random seed value used for reproducible results
        """
        self.randomSeed = randomSeed

        # read the dataset, skipping the first columns (animal name):
        self.data = read_csv(self.DATASET_URL, header=None, usecols=range(1, 18))

        # separate to input features and resulting category (last column):
        self.X = self.data.iloc[:, 0:16]
        self.y = self.data.iloc[:, 16]

        # split the data, creating a group of training/validation sets to be used in the k-fold validation process:
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS) #, random_state=self.randomSeed)

        self.classifier = DecisionTreeClassifier(random_state=self.randomSeed)

    def __len__(self):
        """
        :return: the total number of features used in this classification problem
        """
        return self.X.shape[1]

    def getMeanAccuracy(self, zeroOneList):
        """
        returns the mean accuracy measure of the calssifier, calculated using k-fold validation process,
        using the features selected by the zeroOneList
        :param zeroOneList: a list of binary values corresponding the features in the dataset. A value of '1'
        represents selecting the corresponding feature, while a value of '0' means that the feature is dropped.
        :return: the mean accuracy measure of the calssifier when using the features selected by the zeroOneList
        """

        # drop the dataset columns that correspond to the unselected features:
        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX = self.X.drop(self.X.columns[zeroIndices], axis=1)

        # perform k-fold validation and determine the accuracy measure of the classifier:
        cv_results = model_selection.cross_val_score(self.classifier, currentX, self.y, cv=self.kfold, scoring='accuracy')

        # return mean accuracy:
        return cv_results.mean()

# create a problem instance:
zoo = Zoo(randomSeed=42)

### Param
CHROMOSOME = len(zoo)            # n of guards
POP_SIZE = 120                   # adjust to improve/worsen algo performance
P_CROSSOVER = 0.9                # can be adjusted
P_MUTATION = 0.02                # can be adjusted
MAX_GEN = 50
FEATURE_PENALTY_FACTOR = 0.02

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

### Register param to toolbox
toolbox = base.Toolbox()
    ### Register a function to create the structure of the chromosome
toolbox.register("Binary", random.randint, 0, 1)  # func name, opr, args
    ### Create a class which determines the evaluation mathod (maximize/minimize)   --> Fitness class (newClass, baseClass, attr)
    ### weights can be any real number and only the *sign* is used to determine if a maximization or minimization is done
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    ### Create a class to create an individual + fitness
creator.create("Individual", list, fitness=creator.FitnessMax)    # attr of class Individual: fitness, every individual composed by a list & has a fitness
toolbox.register("IndividualCreator", tools.initRepeat, creator.Individual, toolbox.Binary, CHROMOSOME)  # initRepeat: (container, func, n)
toolbox.register("PopulationCreator", tools.initRepeat, list, toolbox.IndividualCreator)  # n individual --> population

### Create fitness func
def FitnessFunction(individual):
    individual = [1 if x>0 else 0 for x in individual]
    numFeatureUsed = sum(individual)
    if numFeatureUsed == 0:
        return 0.0,
    else:
        accuracy = zoo.getMeanAccuracy(individual)
        return accuracy - FEATURE_PENALTY_FACTOR * numFeatureUsed ,
    # return zoo.getMeanAccuracy(individual),

### Register genetic opr to the toolbox 
### (selection, crossover, mutation, evaluation-fitness)
toolbox.register("evaluate", FitnessFunction)
    # Select the best individual among *tournsize* randomly chosen individuals, *k* times. The list returned contains references to the input *individuals*.
    # selTournament: (individuals, k, tournsize, fit_attr="fitness") -> list
# toolbox.register("select", tools.selBest, fit_attr="fitness")
toolbox.register("select", tools.selTournament, tournsize=2)   # 3 individuals compete & select winner
    # Executes a two-point crossover on the input sequence individuals. The two individuals are modified in place and both keep their original length
    # cxTwoPoint: (ind1, ind2) -> tuple
toolbox.register("mate", tools.cxTwoPoint)  
    # mutFlipBit: (individual, indpb) -> tuple
toolbox.register("mutate", tools.mutFlipBit, indpb=1/POP_SIZE)


# testing the class:
def main():

    allOnes = [1] * len(zoo)    
    print("-- All features selected: ", allOnes, ", accuracy = ", zoo.getMeanAccuracy(allOnes))

    population = toolbox.PopulationCreator(n=POP_SIZE)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)   # Object that compiles statistics on a list of arbitrary objects.
    stats.register("max", np.max)   # max fitness
    stats.register("avg", np.mean)  # avg fitness

    hof = tools.HallOfFame(1)
    # Without elitism
    # population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GEN, 
    #                                           stats=stats, halloffame=hof, verbose=True)
    # With elitism
    population, logbook = eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GEN, 
                                              stats=stats, halloffame=hof, verbose=True)
    maxFitnessValue, meanFitnessValue = logbook.select('max', 'avg')

    print("\nSolution: \n")
    print("-- Features selected: ", hof[0], ", accuracy = ", zoo.getMeanAccuracy(hof[0]))
    

    plt.plot(maxFitnessValue, color='red', label='max')
    plt.plot(meanFitnessValue, color='green', label='mean')
    plt.xlabel('Max/average fitness')
    plt.ylabel('Max/average fitness over generation')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

