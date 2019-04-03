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
Bias=[]

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
    global weights , Output_weights , hiddenLayers , Bias , Forward , gradients
    hiddenLayers = hiddenL
    for i in range(hiddenLayers):
        if i==0:
            weight = np.random.uniform(low=0.0, high=1.0, size=(int(neuronList[i]),4))
            B = np.random.uniform(low= 0.0 , high=1.0 , size=(int(neuronList[i]) , 1))
            Bias.append(B)
        else:
            weight = np.random.uniform(low=0.0, high=1.0, size=(int(neuronList[i]), int(neuronList[i-1])))
            B = np.random.uniform(low=0.0, high=1.0, size=(int(neuronList[i]), 1))
            Bias.append(B)
        weights.append(weight)
    Output_weights=np.random.uniform(low=0.0 , high=1.0 , size = (3 , int(neuronList[hiddenLayers-1])))
    B = np.random.uniform(low=0.0, high=1.0, size=(3, 1))
    Bias.append(B)
    Output_weights= np.array(Output_weights,  dtype=float)

    for e in range(epochs):
        for i in range(90):
            Input = np.array(X[i][0:4], dtype=float)
            Input = Input.reshape((4,1))
            Forward = []
            gradients = []

            for j in range(hiddenLayers):
                if j==0 :
                    net = np.dot(weights[j] , Input) + Bias[j]
                else:
                    net = np.dot(weights[j] , Forward[j-1]) + Bias[j]
                if(activationFunction=='  Sigmoid'):
                    net_act = sigmoid(net)
                else:
                    net_act = np.tanh(net)
                Forward.append(net_act)


            Y_hat = np.dot(Output_weights, Forward[hiddenLayers-1]) + Bias[hiddenLayers]

            if (activationFunction == '  Sigmoid'):
                Y_hat = sigmoid(Y_hat)
            else:
                Y_hat = np.tanh(Y_hat)
                #####   Backward  ######
            if i <30: #class1
                gradient = (Y1 - Y_hat) * derivative_activation(Y_hat)
                gradients.append(gradient)
            elif i<60:
                gradient = (Y2 - Y_hat) * derivative_activation(Y_hat)
                gradients.append(gradient)
            else:
                gradient = (Y3 - Y_hat) * derivative_activation(Y_hat)
                gradients.append(gradient)
            for j in reversed(range(hiddenLayers)):
                if j == hiddenLayers-1:
                    gradient = derivative_activation(Forward[j])* np.dot(Output_weights.T,gradients[0])
                else:
                    gradient = derivative_activation(Forward[j])* np.dot(weights[j+1].T, gradients[hiddenLayers-j-1])
                gradients.append(gradient)

            ######  Update  ########
            for j in range(hiddenLayers):
                if j==0:
                    weights[0]=weights[0]+(learning_rate* gradients[hiddenLayers-j] *Input.T)
                    Bias[0]=Bias[0]+(learning_rate* gradients[hiddenLayers-j])
                else:
                    weights[j]=weights[j]+(learning_rate* gradients[hiddenLayers-j]*Forward[j-1].T)
                    Bias[j] = Bias[j] + (learning_rate * gradients[hiddenLayers - j])
            Output_weights = Output_weights + (learning_rate*gradients[0]* Forward[hiddenLayers-1].T)
            Bias[hiddenLayers]=Bias[hiddenLayers]+(learning_rate*gradients[0])
    Test()


def Test():
    global hiddenLayers , Activation , Output_weights ,Bias , weights
    Class1 = 0
    Class2 = 0
    Class3 = 0
    for i in range(60):
            Input = np.array(test[i], dtype=float)
            Input = Input.reshape((4, 1))
            Forward=[]
            for j in range(hiddenLayers):
                if j == 0:
                    net = np.dot(weights[j], Input) + Bias[j]
                else:
                    net = np.dot(weights[j], Forward[j - 1]) + Bias[j]
                if (Activation == '  Sigmoid'):
                    net_act = sigmoid(net)
                else:
                    net_act = np.tanh(net)
                Forward.append(net_act)


            Y_hat = np.dot(Output_weights, Forward[hiddenLayers - 1]) + Bias[hiddenLayers]

            #print('Y_hat', Y_hat, '\n')

            if (Activation == '  Sigmoid'):
                Y_hat = sigmoid(Y_hat)
            else:
                Y_hat = np.tanh(Y_hat)


            print(Y_hat,i)
            if(max(Y_hat[0] , Y_hat[1] , Y_hat[2]) == Y_hat[0]):
                #print('Class1' , i+1)
                Class1+=1
            if (max(Y_hat[0], Y_hat[1], Y_hat[2]) == Y_hat[1]):
                #print('Class2', i+1)
                Class2+=1
            if (max(Y_hat[0], Y_hat[1], Y_hat[2]) == Y_hat[2]):
                #print('Class3',i+1)
                Class3+=1


    error = 0
    if(Class1<20):
        error+= 20-Class1
    if(Class2<20):
        error+= 20-Class2
    if(Class3<20):
        error+= 20-Class3
    print('Finished')
    Accuracy = abs(60 - error)/60
    Accuracy*=100

    print(Class1,Class2,Class3)
    print(Accuracy)

