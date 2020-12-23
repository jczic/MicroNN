
# -*- coding: utf-8 -*-

from microNN    import MicroNN
from tkinter    import *
from PIL        import Image, ImageDraw, ImageTk
from random     import random

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
    col = ( round(random()*255),
            round(random()*255),
            round(random()*255) )
    examples.append( ( [x/width, y/height], [col] ) )
    can.create_oval( x-15, y-15, x+15, y+15,
                     fill  = rgb2hex(col),
                     width = 0 )

# ----------------------------------------------------------------

def process() :
    global photoBuffer
    if len(examples) :
        for i in range(10) :
            for ex in examples :
                microNN.Learn(ex[0], ex[1])
        for i in range(70) :
            x    = random()
            y    = random()
            col  = microNN.Predict([x, y])[0]
            x   *= width
            y   *= height
            r    = round(random()*6) + 1
            drawBuffer.ellipse((x-r, y-r, x+r, y+r), fill=col)
        photoBuffer = ImageTk.PhotoImage(imgBuffer)
        can.create_image(0, 0, anchor=NW, image=photoBuffer)
    mainWindow.after(10, process)

# ----------------------------------------------------------------

def onCanvasClick(evt) :
    addExample(evt.x, evt.y)

# ----------------------------------------------------------------

microNN              = MicroNN()
microNN.LearningRate = etaLR

microNN.AddInputLayer  ( dimensions  = MicroNN.Init1D(2),
                         shape       = MicroNN.Shape.Neuron )

microNN.AddLayer       ( dimensions  = MicroNN.Init1D(10),
                         shape       = MicroNN.Shape.Neuron,
                         activation  = MicroNN.Activation.Gaussian,
                         initializer = MicroNN.LogisticInitializer(MicroNN.Initializer.HeUniform),
                         connStruct  = MicroNN.FullyConnected )

microNN.AddLayer       ( dimensions  = MicroNN.Init1D(5),
                         shape       = MicroNN.Shape.Neuron,
                         activation  = MicroNN.Activation.Sigmoid,
                         initializer = MicroNN.LogisticInitializer(MicroNN.Initializer.XavierUniform),
                         connStruct  = MicroNN.FullyConnected )

microNN.AddLayer       ( dimensions  = MicroNN.Init1D(5),
                         shape       = MicroNN.Shape.Neuron,
                         activation  = MicroNN.Activation.Sigmoid,
                         initializer = MicroNN.LogisticInitializer(MicroNN.Initializer.XavierUniform),
                         connStruct  = MicroNN.FullyConnected )

microNN.AddLayer       ( dimensions  = MicroNN.Init1D(1),
                         shape       = MicroNN.Shape.Color,
                         activation  = MicroNN.Activation.Sigmoid,
                         initializer = MicroNN.LogisticInitializer(MicroNN.Initializer.XavierUniform),
                         connStruct  = MicroNN.FullyConnected )

microNN.InitWeights()

mainWindow = Tk()
mainWindow.title('microNN - test colors')
mainWindow.geometry('%sx%s' % (width, height))
mainWindow.resizable(False, False)

can = Canvas( mainWindow,
              width       = width,
              height      = height,
              bg          = 'white',
              borderwidth = 0 )
can.bind('<Button-1>', onCanvasClick)
can.pack()

imgBuffer   = Image.new('RGB', (width, height), (255, 255, 255))
drawBuffer  = ImageDraw.Draw(imgBuffer)
photoBuffer = None
process()

mainWindow.mainloop()
