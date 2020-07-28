import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import gc
from sklearn import preprocessing

#Replace the string in read_csv with the path to the dataset in your computer
nd = pd.DataFrame(pd.read_csv('datasets_179276_403373_ddos_imbalanced_unbalaced_20_80_dataset (1).csv').sort_values('Timestamp', ascending=False), 
                            columns = ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Flow Duration', 'Fwd PSH Flags',
                                      'Bwd PSH Flags', 'Fwd Header Len', 'Bwd Header Len', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
                                      'PSH Flag Cnt','ACK Flag Cnt','URG Flag Cnt','CWE Flag Count', 'ECE Flag Cnt', 'Label'])
gc.collect()

s = set()
for i in nd['Dst IP']:
    s.add(i)
print(len(s))
s_a = set()
s = list(s)
s_df = pd.DataFrame(nd['Dst IP'])
print(s_df.head())
for i in range(len(s)):
    if len(s_df[s_df['Dst IP'] == s[i]])/len(nd) >= .05:
        s_a.add(s[i])
        print('IP Found')
    s_df = s_df[s_df['Dst IP'] != s[i]]
    print(str(i/len(s)))
print(len(s_a))
Dst_IP = s_a.copy()
del s_a
del s
del s_df
gc.collect()


s = set()
for i in nd['Dst Port']:
    s.add(i)
print(len(s))
s_a = set()
s = list(s)
s_df = pd.DataFrame(nd['Dst Port'])
print(s_df.head())
for i in range(len(s)):
    if len(s_df[s_df['Dst Port'] == s[i]])/len(nd) >= .05:
        s_a.add(s[i])
        print('Port Found')
    s_df = s_df[s_df['Dst Port'] != s[i]]
    print(str(i/len(s)))
print(len(s_a))
Dst_Port = s_a.copy()
del s_a
del s
del s_df
gc.collect()

s = set()
for i in nd['Src IP']:
    s.add(i)
print(len(s))
s_a = set()
s = list(s)
s_df = pd.DataFrame(nd['Src IP'])
print(s_df.head())
for i in range(len(s)):
    if len(s_df[s_df['Src IP'] == s[i]])/len(nd) >= .01:
        s_a.add(s[i])
        print('IP Found')
    s_df = s_df[s_df['Src IP'] != s[i]]
    print(str(i/len(s)))
print(len(s_a))
Src_IP = s_a.copy()
del s_a
del s
del s_df
gc.collect()

s = set()
for i in nd['Src Port']:
    s.add(i)
print(len(s))
s_a = set()
s = list(s)
s_df = pd.DataFrame(nd['Src Port'])
print(s_df.head())
for i in range(len(s)):
    if len(s_df[s_df['Src Port'] == s[i]])/len(nd) >= .05:
        s_a.add(s[i])
        print('Port Found')
    s_df = s_df[s_df['Src Port'] != s[i]]
    print(str(i/len(s)))
print(len(s_a))
Src_Port = s_a.copy() 
del s_a
del s
del s_df
gc.collect()

Src_IP_Encode_0 = []
Src_IP_Encode_1 = []
Src_IP_Encode_2 = []
Dst_IP_Encode_0 = []
Dst_IP_Encode_1 = []
Dst_IP_Encode_2 = []
Src_Port_Encode_0 = []
Src_Port_Encode_1 = []
Dst_Port_Encode_0 = []
Dst_Port_Encode_1 = []
Dst_Port_Encode_2 = []
Dst_Port_Encode_3 = []

#Endoding Src IP

for i in nd['Src IP']:
    if i == Src_IP[0]:
        Src_IP_Encode_0.append(1)
    else:
        Src_IP_Encode_0.append(0)
for i in nd['Src IP']:
    if i == Src_IP[1]:
        Src_IP_Encode_1.append(1)
    else:
        Src_IP_Encode_1.append(0)
for i in nd['Src IP']:
    if i == Src_IP[2]:
        Src_IP_Encode_2.append(1)
    else:
        Src_IP_Encode_2.append(0)

#Encoding Dst IP
for i in nd['Dst IP']:
    if i == Dst_IP[0]:
        Dst_IP_Encode_0.append(1)
    else:
        Dst_IP_Encode_0.append(0)

for i in nd['Dst IP']:
    if i == Dst_IP[1]:
        Dst_IP_Encode_1.append(1)
    else:
        Dst_IP_Encode_1.append(0)

for i in nd['Dst IP']:
    if i == Dst_IP[2]:
        Dst_IP_Encode_2.append(1)
    else:
        Dst_IP_Encode_2.append(0)

#Encoding Src Port
for i in nd['Src Port']:
    if i == Src_Port[0]:
        Src_Port_Encode_0.append(1)
    else:
        Src_Port_Encode_0.append(0)

for i in nd['Src Port']:
    if i == Src_Port[1]:
        Src_Port_Encode_1.append(1)
    else:
        Src_Port_Encode_1.append(0)

#Encoding Dst Port
for i in nd['Dst Port']:
    if i == Dst_Port[0]:
        Dst_Port_Encode_0.append(1)
    else:
        Dst_Port_Encode_0.append(0)

for i in nd['Dst Port']:
    if i == Dst_Port[1]:
        Dst_Port_Encode_1.append(1)
    else:
        Dst_Port_Encode_1.append(0)
for i in nd['Dst Port']:
    if i == Dst_Port[2]:
        Dst_Port_Encode_2.append(1)
    else:
        Dst_Port_Encode_2.append(0)

for i in nd['Dst Port']:
    if i == Dst_Port[3]:
        Dst_Port_Encode_3.append(1)
    else:
        Dst_Port_Encode_3.append(0)

nd['Src IP Encode 0'] = Src_IP_Encode_0
nd['Src IP Encode 1'] = Src_IP_Encode_1
nd['Src IP Encode 2'] = Src_IP_Encode_2
nd['Dst IP Encode 0'] = Dst_IP_Encode_0
nd['Dst IP Encode 1'] = Dst_IP_Encode_1
nd['Dst IP Encode 2'] = Dst_IP_Encode_2
nd['Src Port Encode 0'] = Src_Port_Encode_0
nd['Src Port Encode 1'] = Src_Port_Encode_1
nd['Src Port Encode 0'] = Src_Port_Encode_0
nd['Dst Port Encode 1'] = Dst_Port_Encode_1
nd['Dst Port Encode 2'] = Dst_Port_Encode_2
nd['Dst Port Encode 3'] = Dst_Port_Encode_3

protocol_0 = []
protocol_6 = []
protocol_17 = []
for i in nd['Protocol']:
    if i == 0:
        protocol_0.append(1)
    else:
        protocol_0.append(0)
for i in nd['Protocol']:
    if i == 6:
        protocol_6.append(1)
    else:
        protocol_6.append(0)
for i in nd['Protocol']:
    if i == 17:
        protocol_17.append(1)
    else:
        protocol_17.append(0)

nd['Protocol 0'] = protocol_0
nd['Protocol 6'] = protocol_6
nd['Portocol 17'] = protocol_17
nd = nd.drop(columns = ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol'])

Fwd_Len = [nd['Fwd Header Len']]
Bwd_Len = [nd['Bwd Header Len']]
Flow_Duration = [nd['Flow Duration']]

Fwd_Len_Norm = preprocessing.normalize(np.array(Fwd_Len))
Bwd_Len_Norm = preprocessing.normalize(np.array(Bwd_Len))
Flow_Duration_Norm = preprocessing.normalize(np.array(Flow_Duration))
nd['Fwd Header Len'] = Fwd_Len_Norm[0]
nd['Bwd Header Len'] = Bwd_Len_Norm[0]
nd['Flow Duration'] = Flow_Duration_Norm[0]
gc.collect()
nd.to_csv("One Encoded and Reg.csv")
labels = pd.DataFrame(nd['Label'])
labels.to_csv("The Right Labels.csv")