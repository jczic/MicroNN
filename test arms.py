
# -*- coding: utf-8 -*-

from microNN    import MicroNN
from math       import sin, cos, pi, atan2, sqrt
from tkinter    import *
from PIL        import Image, ImageDraw, ImageTk
from random     import random

# ----------------------------------------------------------------

etaLR           = 0.3   # Learning rate

width           = 500   # Window/Canvas width
height          = 500   # Window/Canvas height

maxDistance     = 500
r1              = 70
r2              = 70
r3              = 25

# ----------------------------------------------------------------

def rgb2hex(rgb):
    return '#%02x%02x%02x' % rgb 

# ----------------------------------------------------------------

class Arm :

    def __init__(self, can, x, y) :
        self._can = can
        self._x   = x
        self._y   = y
        self._a1  = None
        self._a2  = None
        self._a3  = None
        can.create_rectangle( x-10, y-10, x+10, y+10,
                              fill  = rgb2hex((100, 150, 255)),
                              width = 2 )

    def canRemove(self) :
        self._can.delete(self._a1)
        self._can.delete(self._a2)
        self._can.delete(self._a3)

    def canDraw(self, ballX, ballY) :
        ballX -= self._x
        ballY -= self._y
        dist  = sqrt((ballX**2)+(ballY**2)) / maxDistance
        angle = atan2(ballY, ballX)
        res   = microNN.Predict( [ dist*2-1, cos(angle), sin(angle) ] )
        self.canRemove()
        refX  = self._x
        refY  = self._y
        angle = atan2(res[1], res[0])
        a1X   = cos(angle) * r1
        a1Y   = sin(angle) * r1
        self._a1 = can.create_line( refX, refY, refX+a1X, refY+a1Y,
                                    fill  = rgb2hex((100, 150, 255)),
                                    width = 6 )
        refX += a1X
        refY += a1Y
        angle = angle + atan2(res[3], res[2])
        a2X   = cos(angle) * r2
        a2Y   = sin(angle) * r2
        self._a2 = can.create_line( refX, refY, refX+a2X, refY+a2Y,
                                    fill  = rgb2hex((150, 200, 255)),
                                    width = 5 )
        refX += a2X
        refY += a2Y
        angle = angle + atan2(res[5], res[4])
        a3X   = cos(angle) * r3
        a3Y   = sin(angle) * r3
        self._a3 = can.create_line( refX, refY, refX+a3X, refY+a3Y,
                                    fill  = rgb2hex((150, 200, 255)),
                                    width = 3 )

# ----------------------------------------------------------------

def processXY(x, y) :
    global ball
    can.delete(ball)
    ball = can.create_oval( x-15, y-15, x+15, y+15,
                            fill  = rgb2hex((255, 100, 150)),
                            width = 1 )
    arm1.canDraw(x, y)
    arm2.canDraw(x, y)
    arm3.canDraw(x, y)
    arm4.canDraw(x, y)
    arm5.canDraw(x, y)

# ----------------------------------------------------------------

def onCanvasClick(evt) :
    processXY(evt.x, evt.y)

# ----------------------------------------------------------------

microNN              = MicroNN()
microNN.LearningRate = etaLR

microNN.AddInputLayer  ( dimensions  = MicroNN.Init1D(3),
                         shape       = MicroNN.Shape.Neuron )

microNN.AddLayer       ( dimensions  = MicroNN.Init1D(15),
                         shape       = MicroNN.Shape.Neuron,
                         activation  = MicroNN.Activation.LeakyReLU,
                         initializer = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform),
                         connStruct  = MicroNN.FullyConnected )

microNN.AddLayer       ( dimensions  = MicroNN.Init1D(15),
                         shape       = MicroNN.Shape.Neuron,
                         activation  = MicroNN.Activation.LeakyReLU,
                         initializer = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform),
                         connStruct  = MicroNN.FullyConnected )

microNN.AddLayer       ( dimensions  = MicroNN.Init1D(6),
                         shape       = MicroNN.ValueShape(MicroNN.FloatValueType(-1, 1)),
                         activation  = MicroNN.Activation.Sigmoid,
                         initializer = MicroNN.LogisticInitializer(MicroNN.Initializer.XavierUniform),
                         connStruct  = MicroNN.FullyConnected )

microNN.InitWeights()

for i in range(1000) :
    degArm1 = random() * 360
    degArm2 = random() * 160
    degArm3 = random() * 50
    radArm1 = degArm1 * pi / 180
    radArm2 = degArm2 * pi / 180
    radArm3 = degArm3 * pi / 180
    ballX   = (cos(radArm1)*r1) \
            + (cos(radArm1+radArm2)*r2) \
            + (cos(radArm1+radArm2+radArm3)*r3)
    ballY   = (sin(radArm1)*r1) \
            + (sin(radArm1+radArm2)*r2) \
            + (sin(radArm1+radArm2+radArm3)*r3)
    dist    = sqrt((ballX**2)+(ballY**2)) / maxDistance
    angle   = atan2(ballY, ballX)
    microNN.AddExample( [ dist*2-1, cos(angle), sin(angle) ],
                        [ cos(radArm1), sin(radArm1),
                          cos(radArm2), sin(radArm2),
                          cos(radArm3), sin(radArm3) ] )

try :
    microNN.LearnExamples(minibatchSize=10)
    print(' --> Ok.')
except KeyboardInterrupt :
    print(' --> Aborted!')
print()

ball = None

mainWindow = Tk()
mainWindow.title('microNN - test arms')
mainWindow.geometry('%sx%s' % (width, height))
mainWindow.resizable(False, False)

can = Canvas( mainWindow,
              width       = width,
              height      = height,
              bg          = 'white',
              borderwidth = 0 )
#can.bind('<Button-1>', onCanvasClick)
can.pack()

arm1 = Arm(can, 250, 250)
arm2 = Arm(can, 100, 100)
arm3 = Arm(can, 400, 100)
arm4 = Arm(can, 100, 400)
arm5 = Arm(can, 400, 400)

ballCurX = 200
ballCurY = 100
ballDirX = 3
ballDirY = 3

def process() :
    global ballCurX, ballCurY, ballDirX, ballDirY
    ballCurX += ballDirX
    ballCurY += ballDirY
    if ballCurX <= 15 or ballCurX >= width-15 :
        ballDirX = -ballDirX + (random()-0.5)
    if ballCurY <= 15 or ballCurY >= height-15 :
        ballDirY = -ballDirY + (random()-0.5)
    processXY(ballCurX, ballCurY)
    mainWindow.after(10, process)

process()
mainWindow.mainloop()
