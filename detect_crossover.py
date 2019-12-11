import csv
from scipy.optimize import curve_fit
import numpy as np
import math
import random
import time

# Altered mutation() and crossover()

# Returns the chi squared score testing for epistatic relationship between snp1 and snp2
# Fits data using logistic regression
# Compares epistatic model with null model
# Takes as input the id number of two snps and the data
def score_epistasis(data, snps, table):
    snp1 = snps[0]
    snp2 = snps[1]
    # check if score already calculated
    if (table[snp1][snp2] != None):
        return table[snp1][snp2]

    # return least score if snps are the same
    ### UPDATE if scoring function changes
    if ( snp1 == snp2 ):
        return 0.0
    
    prevalence = compute_prevalence(data, snp1, snp2)
    snp_data = []
    # compile snp values (xdata)
    for sample in data:
        # exclude from data if disease prevalence is zero
        if ( prevalence[sample[snp1]][sample[snp2]] > 0 ):
            snp_data.append([sample[snp1], sample[snp2]])

    # compile log odds for each snp pair (ydata)
    odds = []
    for sample in snp_data:
        p = prevalence[sample[0]][sample[1]]
        odds.append(log_odds(p))

    # fit parameters using epistatic and null model
    epi_params, epi_cov = curve_fit(epistasis_function, np.transpose(snp_data), odds)
    null_params, null_cov = curve_fit(null_function, np.transpose(snp_data), odds)
    #print(epi_params)
    #print(null_params)
    
    # calculate chi squared between epi params and null_params, 4 df
    chi_sq = 0.0
    for i, param in enumerate(epi_params):
        if ( i >= len(null_params) ):
            chi_sq += param**2
        else:
            chi_sq += (param - null_params[i])**2 

    table[snp1][snp2]  = chi_sq / np.sum(np.absolute(null_params))
    table[snp2][snp1]  = chi_sq / np.sum(np.absolute(null_params))
    return chi_sq / np.sum(np.absolute(null_params))
'''
    # Recalculate log odds prevalence using non-epistatic parameters and epistatic params
    prevalence_null = [ [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    prevalence_epi =  [ [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # loop over all genotype combos
    for x in range(0, 3):
        for y in range(0, 3):
            prevalence_null[x][y] = null_function([x, y], null_params[0], null_params[1], null_params[2])
            prevalence_epi[x][y] = epistasis_function([x, y], epi_params[0], epi_params[1], epi_params[2], epi_params[3])


    # convert prevalence to log odds
    prevalence_obs = prevalence.copy()
    for i in range(0, 3):
        for j in range(0, 3):
            if (prevalence_obs[i][j] > 0):
                prevalence_obs[i][j] = log_odds(prevalence_obs[i][j])
            else:
                prevalence_obs[i][j] = None

#    print(prevalence_obs)
#    print(prevalence_null)
#    print(prevalence_epi)
    # Compute chi squared
    chi_sq = 0.0
    for i in range(0, 3):
        for j in range(0, 3):
            if ( not math.isnan(prevalence[i][j]) ):
                chi_sq += ( (prevalence[i][j] - prevalence_null[i][j])**2 / math.fabs(prevalence_null[i][j]) )
    
    return chi_sq
    '''

def log_odds(p):
    return math.log(p / (1.0 - p))

# Function to model epistatic interaction for 2-order epistasis
# Takes snp_values [snp1, snp2]
# Rest of inputs are parameters to be estimated
# beta3 parameter is the epistatic interaction coefficient
def epistasis_function(snp_values, alpha, beta1, beta2, beta3):
    return alpha + beta1*snp_values[0] + beta2*snp_values[1] + beta3*snp_values[0]*snp_values[1]

# Function to model the null model: no epistasis
def null_function(snp_values, alpha, beta1, beta2):
    return alpha + beta1*snp_values[0] + beta2*snp_values[1]

# Return a table with the prevalence of the disease given the values of snp1 and snp2 
def compute_prevalence(data, snp1, snp2):
    prevalence = [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ] # rows = snp1 value, columns = snp2 value
    num_cases = 0
    for sample in data:
        snp1_value = sample[snp1]
        snp2_value = sample[snp2]
        if ( sample[-1] == 1 ):
            prevalence[snp1_value][snp2_value] += sample[-1]
            num_cases += 1
            
    # divide by numer of cases samples to get probability
    return np.true_divide(prevalence, len(data)) 


# Returns a list of candidate vectors for epistasis using genetic algorithm
def DESeeker(data, iterations, pop_size, order, num_snps, scaling_factor, cr_rate):
    # generate initial random population
    population = []
    for i in range(pop_size):
        sample = random.sample(range(num_snps), order)
        sample.append(0.1 + random.random() * 0.9)
        sample.append(random.random())
        population.append(sample)

    # store table of scores to avoid recomputation
    score_table = np.full((num_snps, num_snps), None)
    
    # generations
    candidates = []
    for j in range(iterations):
        #print("iteration num: ", j)
        best_vector = population[0]
        best_score = score_epistasis(data, best_vector[:order], score_table)
        next_gen = []
        for i in range(pop_size):
            mutated_vector = mutation(population[i], population, num_snps, scaling_factor)
            crossover_vector = crossover(population[i], mutated_vector, num_snps, cr_rate)
            selected_vector = selection(data, population[i], crossover_vector, score_table)
            next_gen.append(selected_vector)
            if (score_epistasis(data, selected_vector[:order], score_table) > score_epistasis(data, best_vector[:order], score_table)):
                #best_vector = selected_vector
                #best_score = score_epistasis(data, best_vector, score_table)
                best_vector, best_score = hill_climb(selected_vector, population, pop_size, num_snps, cr_rate, score_table)
        if (best_vector[:-2] not in candidates):
            candidates.append(best_vector[:-2])
       # print(best_vector, best_score, np.count_nonzero(score_table == None))
        population = next_gen
    return candidates, np.count_nonzero(score_table == None)

# return the new best vector and its score after hill climbing
def hill_climb(best_vector, population, pop_size, num_snps, cr_probability, table):
    current_best = best_vector.copy()
    for i in range(pop_size):
        random_vector = random.choice(population)[:-2]
        cross = crossover(current_best, random_vector, num_snps, cr_probability)
        if (score_epistasis(data, cross, table) > score_epistasis(data, current_best, table)):
            current_best = cross
    return current_best, score_epistasis(data, current_best, table)


# Random seeker
def RandomSeeker(data, iterations, order, num_snps):
    score_table = np.full((num_snps, num_snps), None)

    best_vector = random.sample(range(num_snps), order)
    for i in range(iterations):
        random_vector = random.sample(range(num_snps), 2)
        if ( score_epistasis(data, random_vector, score_table) > score_epistasis(data, best_vector, score_table) ):
            best_vector = random_vector

    print("Score", score_epistasis(data, best_vector, score_table))
    return best_vector

# testing; select random mutations 
def mutation(target, population, num_snps, scaling_factor):
    return random.sample(range(num_snps), 2)
    
def mutation2(target, population, num_snps, scaling_factor):
    scaling_factor = target[-2]
    # select three random vectors from population
    vector_nums = random.sample(range(len(population)), 3)
    v1 = population[vector_nums[0]].copy()
    v2 = population[vector_nums[1]].copy()
    v3 = population[vector_nums[2]].copy()

    v_diff = []
    for i in range(len(v3)):
        v_diff.append(scaling_factor * (v3[i] - v2[i]))

    for i in range(len(v1)):
        v1[i] = v1[i] + int(round(v_diff[i]))
    # fix out of bounds

    for i in range(len(v1)):
        if (v1[i] < 0):
            v1[i] *= -1
        if (v1[i] >= num_snps):
            v1[i] = random.randint(0, num_snps-1)
        
    return v1

# Just return mutant vector with parameters attached
def crossover(target, mutant, num_snps, cr_probability):
    cross = mutant.copy()
    # append the parameters
    cross.append(target[-2])
    cross.append(target[-1])
    return cross

def selection(data, target, crossed, table):
    if ( score_epistasis(data, crossed[:-2], table) > score_epistasis(data, target[:-2], table) ):
        if ( random.random() < 0.1 ):
            crossed[-2] = 0.1 + random.random() * 0.9
        if ( random.random() < 0.1 ):
            crossed[-1] = random.random()
        return crossed
    else:
        if ( random.random() < 0.1 ):
            target[-2] = 0.1 + random.random() * 0.9
        if ( random.random() < 0.1 ):
            target[-1] = random.random()
        return target
    


for model in [100, 200, 300, 400, 500]:
    power = 0.0
    percentage_explored = 0.0
    total_time = 0.0
    num_snps = model
    num_files = 101
    if ( model >= 400 ):
        num_files = 51
    total_time = 0.0
    for file_num in range(1, num_files):
        data_file = open("model%d/model%d_EDM-1_%03d.txt" %(model, model, file_num))
        reader = csv.reader(data_file, delimiter="\t")
        next(reader)
        data = [] # in 2D array form; each row is a different patient
        for line in reader:
            data.append(list(map(int, line))) # convert to ints

        start = time.time()
        candidate_vectors, pairs_explored = DESeeker(data, 25 * int(num_snps / 100), 500 * int(num_snps / 100), 2, num_snps, 0.9, 0.8)
        if ([num_snps-1, num_snps-2] in candidate_vectors or [num_snps-2, num_snps-1] in candidate_vectors):
            power += 1
        end = time.time()
        total_time += (end - start)
        percentage_explored += (pairs_explored / (num_snps**2))
    power = power / (num_files - 1)
    average_time = total_time / (num_files - 1)
    percentage_explored = percentage_explored / (num_files - 1)
    print("%%d SNPs\tpower: %d\ttime: %d\t, percentage explored: %d" %(model, power, average_time, percentag_explored))

'''
data_file = open("model3/model3_Models.txt_EDM-1_001.txt")
reader = csv.reader(data_file, delimiter="\t")
next(reader)
data = [] # in 2D array form; each row is a different patient
for line in reader:
    data.append(list(map(int, line))) # convert to ints

dummy_table = np.full((2000, 2000), None)
#print(score_epistasis(data, [1998, 1999], dummy_table))
#print(score_epistasis(data, [34, 31], dummy_table))
candidates = {}
for snp1 in range(0, 2000):
    for snp2 in range(snp1+1, 2000):
        score = score_epistasis(data, [snp1, snp2], dummy_table)
        #print(score)
        if ( len(candidates) < 5 ):
            candidates[(snp1, snp2)] = score
        elif ( score > min(candidates.values()) ):
            del candidates[min(candidates, key=candidates.get)]
            candidates[(snp1, snp2)] = score
            
print(candidates)
'''

# Single test
'''
data_file = open("model100/model100_EDM-1_002.txt")
reader = csv.reader(data_file, delimiter="\t")
next(reader)
data = [] # in 2D array form; each row is a different patient
for line in reader:
    data.append(list(map(int, line))) # convert to ints
num_snps = 100

#print(score_epistasis(data, [num_snps-1, num_snps-2], dummy_table))
start = time.time()
#print(RandomSeeker(data, 1000, 2, 100))
print(DESeeker(data, 25 * int(num_snps / 100), 500 * int(num_snps / 100), 2, num_snps, 0.9, 0.8)) 
print( (time.time() - start))
'''
'''
# Multiple tests
power = 0.0
num_snps = 100
for file_num in range(1, 51):
    data_file = open("model2/model2_EDM-1_%03d.txt" %file_num)
    reader = csv.reader(data_file, delimiter="\t")
    next(reader)
    data = [] # in 2D array form; each row is a different patient
    for line in reader:
        data.append(list(map(int, line))) # convert to ints

    candidates = DESeeker(data, 25, 250, 2,  num_snps, 0.9, 0.8) # set N and M to 500 and 500 later
    if ([num_snps-1, num_snps-2] in candidates or [num_snps-2, num_snps-1] in candidates):
        power += 1
print(power / 50)
'''
'''
for file_num in range(1, 101):
data_file = open("model2/model2_EDM-1_%03d.txt" %file_num)
reader = csv.reader(data_file, delimiter="\t")
next(reader)
data = [] # in 2D array form; each row is a different patient
for line in reader:
     data.append(list(map(int, line))) # convert to ints
        
     print("File " + str(file_num) + ": " , DESeeker(data, 25, 250, 2, 100, 0.9, 0.8)) # set N and M to 500 and 500 later
        
dummy_table = np.full((100, 100), None)
print(dummy_table)
print(score_epistasis(data, [28, 10], dummy_table))
print(score_epistasis(data, [98, 99], dummy_table))

candidates = {}
for snp1 in range(0, 100):
    for snp2 in range(snp1+1, 100):
        score = score_epistasis(data, [snp1, snp2], dummy_table)
        if ( len(candidates) < 5 ):
            candidates[(snp1, snp2)] = score
        elif ( score > min(candidates.values()) ):
            del candidates[min(candidates, key=candidates.get)]
            candidates[(snp1, snp2)] = score
            
print(candidates)
'''
'''
for file_num in range(1, 51):
    data_file = open("model2/model2_EDM-1_%03d.txt" %file_num)
    reader = csv.reader(data_file, delimiter="\t")
    next(reader)
    data = [] # in 2D array form; each row is a different patient
    for line in reader:
        data.append(list(map(int, line))) # convert to ints

    # find five pairs of snps most likely
    candidates = {}
    for snp1 in range(0, 100):
        for snp2 in range(snp1+1, 100):
            score = score_epistasis(data, [snp1, snp2])
            if ( len(candidates) < 5 ):
                candidates[(snp1, snp2)] = score
            elif ( score > min(candidates.values()) ):
                del candidates[min(candidates, key=candidates.get)]
                candidates[(snp1, snp2)] = score
                
    print("File " + str(file_num) + ": " + str(candidates))
    print("Max" + str(max(candidates, key=candidates.get)))

'''    

