import sys
sys.path.append('/home/gpeled/workspace/pytorch/python3-pytorch/lib/python3.5/site-packages/')
#print ('\n'.join(sys.path))
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def update_view( x, new_y ) :
    #pl.set_ydata(new_y)
    #plt.draw()
    plt.plot(x, new_y, 'g-')
    plt.show(block=False)


dim=50
x_values = [i for i in range(dim)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values_pure = [0.25 * i + 10  for i in x_values]
y_train = np.array(y_values_pure, dtype=np.float32)
noise = np.random.randn( *y_train.shape )
noise = noise / 2
y_train += noise
y_train = y_train.reshape(-1, 1)

plt.plot(x_train,y_train,'rx')
plt.plot(x_train,y_values_pure,'b-')
#flat = np.ones(dim)
#pl, = plt.plot(x_train,flat,'g-')
plt.ylabel('Y')
plt.show(block=False)

'''
CREATE MODEL CLASS
'''


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


'''
INSTANTIATE MODEL CLASS
'''
input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)

#######################
#  USE GPU FOR MODEL  #
#######################

#model.cuda()

'''
INSTANTIATE LOSS CLASS
'''

criterion = nn.MSELoss()

'''
INSTANTIATE OPTIMIZER CLASS
'''

learning_rate = 0.0001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
TRAIN THE MODEL
'''
epochs = 50
for epoch in range(epochs):
    epoch += 1
    # Convert numpy array to torch Variable

    #if not (epoch % 50) :
    #    print ("Reducing learning rate")

    #######################
    #  USE GPU FOR MODEL  #
    #######################
    #inputs = Variable(torch.from_numpy(x_train).cuda())
    inputs = Variable(torch.from_numpy(x_train))

    #######################
    #  USE GPU FOR MODEL  #
    #######################
    #labels = Variable(torch.from_numpy(y_train).cuda())
    labels = Variable(torch.from_numpy(y_train))

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward to get output
    outputs = model(inputs)

    p = list(model.parameters())
    a = p[0].data.numpy()[0][0]
    b = p[1].data.numpy()[0]
    predicted_line = [a * i + b  for i in x_values]
    update_view(x_values,predicted_line)
    # Calculate Loss
    loss = criterion(outputs, labels)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    # Logging
    print('epoch {}, loss {} a={} b={}'.format(epoch, loss.data[0],a,b) )
    if(loss.data[0]< (1/1000000)) :
        print("Reached a small enough loss. Stopping.")
        break
print ("Done - displaying the plot and waiting")
plt.show()
