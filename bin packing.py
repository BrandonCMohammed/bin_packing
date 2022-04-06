import numpy as np #Numpy is a useful library for number manipulation

#The number of items to be stored
num_items = 15 

#Since we don't have specific numbers for the weights, a random set of integers will be generated in the range of 3 to 7
#*item_weight_range = 3 7, it does not including 7
item_weight_range = [3, 7]
weights = list(np.random.randint(*item_weight_range, num_items))

#We will now establish the bin capacity, We do not have a specific number right now so the mean of the weights will be found,
#Then multiplied by 10 and casted to an int to remove any decimal points.
bin_capacity = int(10 * np.mean(weights))
print("Problem: pack a total weight of {} into bins of capacity {}.".format(sum(weights), bin_capacity)) 


#Instantiating our CQM
from dimod import ConstrainedQuadraticModel
cqm = ConstrainedQuadraticModel()

#Now we can create our objective function, The formula will be in the pdf
#Binary is used to create BinaryQuadraticModules
from dimod import Binary

#This creates a list of 
#BQMs with linear keys of bin_used_{j} and values of 1
bin_used = [Binary(f'bin_used_{j}') for j in range(num_items)]
#print(bin_used[0])

cqm.set_objective(sum(bin_used))

#Constrains

#Now we are creating the first constrain. Each item can go in one bin only. This creates a list of 
#BQMs with linear keys of item_{i}_in_bin_{j} and values of 1 
item_in_bin = [[Binary(f'item_{i}_in_bin_{j}') for j in range(num_items)] for i in range(num_items)]
#print(item_in_bin[0])

#for each item i in item in bin, we add the constraint
for i in range(num_items):
    one_bin_per_item = cqm.add_constraint(sum(item_in_bin[i]) == 1, label=f'item_placing_{i}')
    #print(one_bin_per_item)

#This is constraint two. 
for j in range(num_items):
    bin_up_to_capacity = cqm.add_constraint(
        sum(weights[i] * item_in_bin[i][j] for i in range(num_items)) - bin_used[j] * bin_capacity <= 0,
        label=f'capacity_bin_{j}')

    print(bin_up_to_capacity)

len(cqm.variables)

"""
from dwave.system import LeapHybridCQMSampler
sampler = LeapHybridCQMSampler()  

sampleset = sampler.sample_cqm(cqm,
                               time_limit=180,
                               label="SDK Examples - Bin Packing")  
feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)  
if len(feasible_sampleset):      
    best = feasible_sampleset.first
    print("{} feasible solutions of {}.".format(len(feasible_sampleset), len(sampleset)))

selected_bins = [key for key, val in best.sample.items() if 'bin_used' in key and val]   
print("{} bins are used.".format(len(selected_bins))) 

def get_indices(name):
    return [int(digs) for digs in name.split('_') if digs.isdigit()]


for bin in selected_bins: 
                           
    in_bin = [key for key, val in best.sample.items() if
            "_in_bin" in key and
            get_indices(key)[1] == get_indices(bin)[0]
            and val]
    b = get_indices(in_bin[0])[1]
    w = [weights[get_indices(item)[0]] for item in in_bin]
    print("Bin {} has weights {} for a total of {}.".format(b, w, sum(w)))
"""