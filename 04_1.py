import torch
import pandas
import os

os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file = os.path.join('..','data','house_tiny.csv')

with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pandas.read_csv(data_file)
print(data)
input,output = data.iloc[:,0:2],data.iloc[:,2]
input.iloc[:,0] = input.iloc[:,0].fillna(input.iloc[:,0].mean())
input = pandas.get_dummies(input,dummy_na=True,dtype=type(0))
print(input)

X,Y = torch.tensor(input.values),torch.tensor(output.values)
print(X,Y)