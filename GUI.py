import tkinter as tk
from tkinter import *
from tkinter import messagebox
import Back_end


#import Back_end
X_window = '600x'
Y_window = '300'
window  = tk.Tk()

Activation=0


window.title('Task 1')


Activation_List = tk.Listbox(window , height=3, width=8 ,selectmode=tk.SINGLE)
Activation_List.place(x=2,y=50)
Activation_List.insert(1, "  Sigmoid")
Activation_List.insert(2, "  TanH")
Activation_List.place(x=30,y=60)

Activation_Label = tk.Label(text='Activation Function')
Activation_Label.place(x=20 ,y = 20)

def Button_Activation():
    clicked = Activation_List.curselection()
    global Activation
    Activation = Activation_List.get(clicked[0])
    print(Activation)

Activation_button = tk.Button(window,text='Choose' , command= Button_Activation)
Activation_button.place(x=30 , y = 130)




def NN():
    Back_end.Train_Sample()
    Back_end.Test_Sample()
    Back_end.Backprobagation_network(int(CheckVar.get()), int(Epoch_textbox.get()),
                           float(LearningRate_textbox.get()), int(HiddenLayers_textbox.get()),
                           Neurons_textbox.get(), float(MSE_textbox.get()),Activation)



# Train_button = tk.Button(window,text='Train' , command= button3)
# Train_button.place(x=200 , y=250)

plot_button = tk.Button(window,text='Train' , command=NN)
plot_button.place(x=250 , y=250)

CheckVar = IntVar()
Check = Checkbutton(window, text = "Add Bias", variable = CheckVar, \
                 onvalue = 1, offvalue = 0, height=5, \
                 width = 20, )
Check.place(x=450 , y=30)


LearningRate_textbox = Entry(window,width=10)
LearningRate_textbox.place(x=330 , y=20)
LearningLabel = tk.Label(text='Learning Rate')
LearningLabel.place(x=200 , y=20)

Epoch_textbox = Entry(window,width=10)
Epoch_textbox.place(x=330 , y=60)
Epochs = tk.Label(text='Epochs Number')
Epochs.place(x=200 , y=60)


MSE_textbox = Entry(window,width=10)
MSE_textbox.place(x=330 , y=100)
MSE_Label = tk.Label(text='MSE')
MSE_Label.place(x=200 , y=100)


HiddenLayers_textbox = Entry(window,width=10)
HiddenLayers_textbox.place(x=330 , y=140)
HiddenLayers_Label = tk.Label(text='Hidden Layers')
HiddenLayers_Label.place(x=200 , y=140)

Neurons_textbox = Entry(window,width=10)
Neurons_textbox.place(x=330 , y=180)
Neurons_Label = tk.Label(text='Neurons')
Neurons_Label.place(x=200 , y=180)



window.geometry(X_window + Y_window)
window.mainloop()

