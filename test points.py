
# -*- coding: utf-8 -*-

from microNN    import MicroNN
from tkinter    import *
from threading  import *

# ----------------------------------------------------------------

etaLR           = 0.5   # Learning rate

width           = 800   # Window/Canvas width
height          = 500   # Window/Canvas height
examples        = [ ]

# ----------------------------------------------------------------

def rgb2hex(rgb):
    return '#%02x%02x%02x' % rgb 

# ----------------------------------------------------------------

def addExample(x, y) :
    examples.append((x, y))
    can.create_oval( x-7, y-7, x+7, y+7,
                     fill    = '#3366AA',
                     outline = '#AA3366',
                     width   = 2 )

# ----------------------------------------------------------------

class processThread(Thread) :

    def run(self) :
        evt  = Event()
        line = None
        while not evt.wait(0.010) :
            if len(examples) :
                for i in range(10) :
                    for ex in examples :
                        microNN.Learn([ex[0]], [ex[1]])
                pts = [ ]
                for x in range(0, width, 5) :
                    y = microNN.Predict([x])[0]
                    pts.append((x, y))
                can.delete(line)
                line = can.create_line(pts, fill='#3366AA')

# ----------------------------------------------------------------

def onCanvasClick(evt) :
    addExample(evt.x, evt.y)

# ----------------------------------------------------------------

X_Value  = MicroNN.IntValueType(0, width)
Y_Value  = MicroNN.IntValueType(0, height)

microNN              = MicroNN()
microNN.LearningRate = etaLR

microNN.AddInputLayer  ( dimensions  = MicroNN.Init1D(1),
                         shape       = MicroNN.ValueShape(X_Value) )

microNN.AddLayer       ( dimensions  = MicroNN.Init1D(15),
                         shape       = MicroNN.Shape.Neuron,
                         activation  = MicroNN.Activation.Gaussian,
                         initializer = MicroNN.LogisticInitializer(MicroNN.Initializer.HeUniform),
                         connStruct  = MicroNN.FullyConnected )

microNN.AddLayer       ( dimensions  = MicroNN.Init1D(15),
                         shape       = MicroNN.Shape.Neuron,
                         activation  = MicroNN.Activation.Gaussian,
                         initializer = MicroNN.LogisticInitializer(MicroNN.Initializer.HeUniform),
                         connStruct  = MicroNN.FullyConnected )

microNN.AddLayer       ( dimensions  = MicroNN.Init1D(1),
                         shape       = MicroNN.ValueShape(Y_Value),
                         activation  = MicroNN.Activation.Sigmoid,
                         initializer = MicroNN.LogisticInitializer(MicroNN.Initializer.XavierUniform),
                         connStruct  = MicroNN.FullyConnected )

microNN.InitWeights()

mainWindow = Tk()
mainWindow.title('microNN - test points')
mainWindow.geometry('%sx%s' % (width, height))
mainWindow.resizable(False, False)

can = Canvas( mainWindow,
              width       = width,
              height      = height,
              bg          = 'white',
              borderwidth = 0 )
can.bind('<Button-1>', onCanvasClick)
can.pack()

pc = processThread()
pc.daemon = True
pc.start()

mainWindow.mainloop()
