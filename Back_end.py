import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np



class1 = []; class2 = []; class3 = []
X = []
num1 = defaultdict(int)
num2 = defaultdict(int)
num3 = defaultdict(int)

Y1 = np.array([1, 0, 0])
Y2 = np.array([0, 1, 0])
Y3 = np.array([0, 0, 1])

Y1 = Y1.reshape((3,1))
Y2 = Y2.reshape((3,1))
Y3 = Y3.reshape((3,1))

weights = []
test=[]
hiddenLayers = 0
Activation = ''
Output_weights=[]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_activation(x):
    if(Activation=='  Sigmoid'):
        return derivative_sigmoid(x)
    else:
        return derivative_tanH(x)

def derivative_sigmoid(x):
    x = np.array(x)
    xdash = 1 - x
    return np.multiply(x,xdash)

def derivative_tanH(x):
    x = np.array(x)
    return 1-x**2

def Reading_file():
    file = open('IrisData.txt', 'r')
    File_Read=1
    for line in file:
        List = line.split(',')
        if List[4][0:11]=='Iris-setosa':
            class1.append(List[0:5])
        elif List[4][0:15]=='Iris-versicolor':
            class2.append(List[0:5])
        elif List[4][0:14]=='Iris-virginica':
            class3.append(List[0:5])

def Train_Sample():


    global X
    X=[]
    Reading_file()
    i=0;
    global num1,num2,num3
    num1 = defaultdict(int)
    num2 = defaultdict(int)
    num3 = defaultdict(int)
    while i < 30 :
        idx=np.random.randint(0,49)
        if (num1[idx]==0):
            X.append(class1[idx])
            num1[idx]=1
            i += 1
        else:
            i = i;

    i = 0;
    while i < 30:
        idx = np.random.randint(0, 49)
        if (num2[idx] == 0):
            X.append(class2[idx])
            num2[idx] = 1
            i += 1
        else:
            i = i;
    i = 0;
    while i < 30:
        idx = np.random.randint(0, 49)
        if (num3[idx] == 0):
            X.append(class3[idx])
            num3[idx] = 1
            i += 1
        else:
            i = i;

def Test_Sample():
    global test
    test = []

    for i in range(50):
        if ( num1[i]==0):
            test.append(class1[i][0:4])
    for i in range(50):
        if(num2[i]==0):
            test.append(class2[i][0:4])
    for i in range(50):
        if (num3[i] == 0):
            test.append(class3[i][0:4])

def Backprobagation_network(bias , epochs, learning_rate, hiddenL, neurons , mse_stopping,activationFunction):
    #print(bias,epochs,learning_rate,hiddenLayers,neurons,mse_stopping,activationFunction)
    global Activation

    Activation = activationFunction
    neuronList = neurons.split(',')
    global weights , Output_weights , hiddenLayers
    hiddenLayers = hiddenL
    weights = []
    Forward = []
    gradients = []

    for i in range(hiddenLayers):
        if i==0:
            weight = np.random.uniform(low=0.0, high=1.0, size=(int(neuronList[i]),4))
        else:
            weight = np.random.uniform(low=0.0, high=1.0, size=(int(neuronList[i]), int(neuronList[i-1])))
        weights.append(weight)
    Output_weights=np.random.uniform(low=0.0 , high=1.0 , size = (3 , int(neuronList[hiddenLayers-1])))
    Output_weights= np.array(Output_weights,  dtype=float)
    for e in range(epochs):
        for i in range(90):
            Input = np.array(X[i][0:4], dtype=float)
            Input = Input.reshape((4,1))
            Forward = []
            gradients = []
            # print('input shape' , Input.shape)
            #print('Input',Input , '\n' , 'shape' , Input.shape)
            # #####  Forward ######

            #print('weights' , weights , '\n' , 'shape' ,'cant print shape')
            #print('Output Weight' , Output_weights , '\n' , 'shape' , Output_weights.shape)

            for j in range(hiddenLayers):
                if j==0 :
                    # print(type(Input).dtype , type(weights[j].dtype ))
                    net = np.dot(weights[j] , Input)
                else:
                    net = np.dot(weights[j] , Forward[j-1])
                if(activationFunction=='  Sigmoid'):
                    net_act = sigmoid(net)
                else:
                    net_act = np.tanh(net)
                Forward.append(net_act)

            #print('Forward' , Forward , '\n' , 'shape' , 'cant :(')

            Y_hat = np.dot(Output_weights, Forward[hiddenLayers-1])

            #print('Y_hat' , Y_hat , '\n' , 'shape',Y_hat.shape)

            if (activationFunction == '  Sigmoid'):
                Y_hat = sigmoid(Y_hat)
            else:
                Y_hat = np.tanh(Y_hat)
            #if i==0 or i==30 or i==60:
            #    print('Y_hat after activation', Y_hat , '\n', 'shape' , Y_hat.shape)

                #####   Backward  ######
            if i <30: #class1
                gradient = (Y1 - Y_hat) * derivative_activation(Y_hat)
                # print(gradient.shape, 'gradiennnnnnnnnnnt ')
                gradients.append(gradient)
            elif i<60:
                gradient = (Y2 - Y_hat) * derivative_activation(Y_hat)
                gradients.append(gradient)
            else:
                gradient = (Y3 - Y_hat) * derivative_activation(Y_hat)
                gradients.append(gradient)
            for j in reversed(range(hiddenLayers)):
                if j == hiddenLayers-1:
                    # print('shapes0' , Output_weights.shape , gradients[0].shape)
                    gradient = derivative_activation(Forward[j])* np.dot(Output_weights.T,gradients[0])
                else:
                    # print('shapes' , weights[j+1].shape , gradients[hiddenLayers-j-1].shape)
                    gradient = derivative_activation(Forward[j])* np.dot(weights[j+1].T, gradients[hiddenLayers-j-1])
                gradients.append(gradient)
            #print('gradients' , gradients , '\n' , 'shape' ,' cant')

            ######  Update  ########
            for j in range(hiddenLayers):
                if j==0:
                    # print(gradients[hiddenLayers-j].shape,(Input.T).shape)
                    weights[0]=weights[0]+(learning_rate* np.dot(gradients[hiddenLayers-j],Input.T))
                else:
                    # print('errorrr ',weights[j].shape  , 'hoohoh', np.dot(gradients[hiddenLayers-j],Forward[j-1].T).shape )
                    # print(weights[j])
                    # print('ok')
                    weights[j]=weights[j]+(learning_rate* np.dot(gradients[hiddenLayers-j],Forward[j-1].T))
            Output_weights = Output_weights + (learning_rate*np.dot(gradients[0], Forward[hiddenLayers-1].T))

            #print('weights afte update', weights, '\n' , 'shape cant')
            #print('Output Weight after update', Output_weights, '\n' , 'shape cant')
    Test()


def Test():
    global hiddenLayers , Activation , Output_weights
    #print('weights',weights)
    for i in range(60):
            Input = np.array(test[i], dtype=float)
            Input = Input.reshape((4, 1))
            Forward=[]
     #       print('input' , Input)
            for j in range(hiddenLayers):
                if j == 0:
                    #print(type(Input).dtype, type(weights[j].dtype))
                    net = np.dot(weights[j], Input)
                else:
                    net = np.dot(weights[j], Forward[j - 1])
                if (Activation == '  Sigmoid'):
                    net_act = sigmoid(net)
                else:
                    net_act = np.tanh(net)
                Forward.append(net_act)

            #print('Forward Test', Forward, '\n')

            Y_hat = np.dot(Output_weights, Forward[hiddenLayers - 1])

            #print('Y_hat', Y_hat, '\n')

            if (Activation == '  Sigmoid'):
                Y_hat = sigmoid(Y_hat)
            else:
                Y_hat = np.tanh(Y_hat)

            #if i<20:
            print(Y_hat, i)

