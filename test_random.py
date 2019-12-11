import csv
from scipy.optimize import curve_fit
import numpy as np
import math
import random
import time

# Implement a first choice hill climb instead of a genetic algorithm

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
    # for now, just return the epistatic interaction param
    #return epi_params[3]

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


    return current_best, score_epistasis(data, current_best, table)


# Random seeker
def RandomSeeker(data, iterations, order, num_snps):
    score_table = np.full((num_snps, num_snps), None)
    best_vector = random.sample(range(num_snps), order)
    for i in range(iterations):
        random_vector = random.sample(range(num_snps), 2)
        if ( score_epistasis(data, random_vector, score_table) > score_epistasis(data, best_vector, score_table) ):
            best_vector = random_vector

#    print("Score", score_epistasis(data, best_vector, score_table))
    return best_vector, np.count_nonzero(score_table == None)


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
        
        result, pairs_explored = RandomSeeker(data, 5000 * int(num_snps / 100), 2, num_snps)
        if (result == [num_snps-1, num_snps-2] or result == [num_snps-2, num_snps-1]):
            power += 1
        end = time.time()
        total_time += (end - start)
        percentage_explored += (pairs_explored / (num_snps**2))
    power = power / (num_files - 1)
    average_time = total_time / (num_files - 1)
    percentage_explored = percentage_explored / (num_files - 1)
    print("%%d SNPs\tpower: %d\ttime: %d\t, percentage explored: %d" %(model, power, average_time, percentag_explored))

# Single test
'''
data_file = open("model200/model200_EDM-1_002.txt")
reader = csv.reader(data_file, delimiter="\t")
next(reader)
data = [] # in 2D array form; each row is a different patient
for line in reader:
    data.append(list(map(int, line))) # convert to ints
num_snps = 200
start = time.time()
print(RandomSeeker(data, 5000 * int(num_snps / 100), 2, num_snps))
print(time.time() - start)
#print(DESeeker(data, 25, 250, 2, 100, 0.9, 0.8)) # set N and M to 500 and 500 later
'''
# Multiple tests
'''
power = 0.0
num_snps = 100
for file_num in range(1, 51):
    data_file = open("model2/model2_EDM-1_%03d.txt" %file_num)
    reader = csv.reader(data_file, delimiter="\t")
    next(reader)
    data = [] # in 2D array form; each row is a different patient
    for line in reader:
        data.append(list(map(int, line))) # convert to ints

    result = RandomSeeker(data, 2000, 2, 100)
    if ( result == [num_snps-2, num_snps-1] or result == [num_snps-1, num_snps-2] ):
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
for file_num in range(1, 101):
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
