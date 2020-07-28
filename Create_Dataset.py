import numpy as np
import pandas as pd
import gc
from numpy.random import rand

nd = pd.read_csv('One Encoded and Reg.csv').sort_values('Timestamp', ascending=False).drop(columns=['Timestamp', 'Label']).drop(columns=['Unnamed: 0'])
gc.collect()
nd_np = np.array(nd)
random = rand(1, 1200000)*(len(nd)-50)
random = random.astype(int)
random = list(set(random[0]))
labels = pd.read_csv('The Right Labels.csv')
labels = labels.drop(columns=['Unnamed: 0'])
labels = np.array(labels)
for i in range(len(labels)):
    if labels[i] == 'ddos':
        labels[i] = 1
    else:
        labels[i] = 0

sequences = []
labels_random = []
labels_random.append(labels[random[0]].astype(np.int8))
count = 0
for i in random:
    sequences.append(nd_np[i:i+50, :])
    labels_random.append(labels[i].astype(np.int8))
    print((count+1)/len(random))
    count+=1

labels_random = np.array(labels_random)
sequences = np.array(sequences)

#np.save('Sequences_Random_50.npy', sequences)
#np.save('Labels_Random_50.npy', labels_random)