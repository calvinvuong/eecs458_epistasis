import csv
from scipy.optimize import curve_fit
import numpy as np
import math


# Returns the chi squared score testing for epistatic relationship between snp1 and snp2
# Fits data using logistic regression
# Compares epistatic model with null model
# Takes as input the id number of two snps and the data
def score_epistasis(data, snp1, snp2):
    prevalence = compute_prevalence(data, snp1, snp2)
    snp_data = []
    # compile snp values (xdata)
    for sample in data:
        # exclude from data if disease prevalence is zero
        if ( prevalence[sample[snp1]][sample[snp2]] > 0 ):
            snp_data.append([sample[snp1], sample[snp2]])

    # compile log odds for each snp pair (ydata)
    log_odds = []
    for sample in snp_data:
        p = prevalence[sample[0]][sample[1]]
        log_odds.append(math.log(p / (1.0 - p)))

    # fit parameters using epistatic and null model
    epi_params, epi_cov = curve_fit(epistasis_function, np.transpose(snp_data), log_odds)
    null_params, null_cov = curve_fit(null_function, np.transpose(snp_data), log_odds)
        
    # calculate chi squared between epi params and null_params, 4 df
    chi_sq = 0.0
    for i, param in enumerate(epi_params):
        if ( i >= len(null_params) ):
            chi_sq += param**2
        else:
            chi_sq += (param - null_params[i])**2 

    return chi_sq / np.sum(np.absolute(null_params))
    '''
    # recalculate log odds prevalence using non-epistatic parameters and epistatic params
    prevalence_null = [ [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    prevalence_epi =  [ [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # loop over all genotype combos
    for x in range(0, 3):
        for y in range(0, 3):
            prevalence_null[x][y] = null_function([x, y], null_params[0], null_params[1], null_params[2])
            prevalence_epi[x][y] = epistasis_function([x, y], epi_params[0], epi_params[1], epi_params[2], epi_params[3])

    prevalence_obs = prevalence.copy()
    for i in range(0, 3):
        for j in range(0, 3):
            if (prevalence_obs[i][j] > 0):
                prevalence_obs[i][j] = math.log(prevalence_obs[i][j] / (1.0 - prevalence_obs[i][j]))
            else:
                prevalence_obs[i][j] = None
    print(prevalence_obs)
    print(prevalence_null)
    print(prevalence_epi)
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
            score = score_epistasis(data, snp1, snp2)
            if ( len(candidates) < 5 ):
                candidates[(snp1, snp2)] = score
            elif ( score > min(candidates.values()) ):
                del candidates[min(candidates, key=candidates.get)]
                candidates[(snp1, snp2)] = score
                
    print("File " + str(file_num) + ": " + str(candidates))
    print("Max" + str(max(candidates, key=candidates.get)))
