
# -*- coding: utf-8 -*-

from microNN import MicroNN
from PIL     import Image, ImageDraw
from random  import random

# ----------------------------------------------------------------

etaLR           = 3.0   # Learning rate
explCount       = 100   # Number of examples
batchSize       = 5     # Size of minibatchs
evalCount       = 25    # Number of evaluations

w               = 32    # Sides size of images
s               = 10    # Sides size of circles

# ----------------------------------------------------------------

microNN              = MicroNN()
microNN.LearningRate = etaLR

microNN.AddInputLayer  ( dimensions    = MicroNN.Init2D(w, w),
                         shape         = MicroNN.Shape.Neuron )

microNN.AddConv2DLayer ( filtersCount  = 5,
                         filtersDepth  = 3,
                         convSize      = 5,
                         stride        = 2,
                         shape         = MicroNN.Shape.Neuron,
                         activation    = MicroNN.Activation.LeakyReLU,
                         initializer   = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform) )

microNN.AddConv2DLayer ( filtersCount  = 3,
                         filtersDepth  = 2,
                         convSize      = 3,
                         stride        = 2,
                         shape         = MicroNN.Shape.Neuron,
                         activation    = MicroNN.Activation.LeakyReLU,
                         initializer   = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform) )

microNN.AddLayer       ( dimensions    = MicroNN.Init1D(15),
                         shape         = MicroNN.Shape.Neuron,
                         activation    = MicroNN.Activation.LeakyReLU,
                         initializer   = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform),
                         connStruct    = MicroNN.FullyConnected )

microNN.AddLayer       ( dimensions    = MicroNN.Init1D(4),
                         shape         = MicroNN.Shape.Neuron,
                         activation    = MicroNN.Activation.Sigmoid,
                         initializer   = MicroNN.LogisticInitializer(MicroNN.Initializer.XavierUniform),
                         connStruct    = MicroNN.FullyConnected )

microNN.InitWeights()

print()
print('MicroNN:')
print('  - Layers      : %s' % microNN.LayersCount)
print('  - Neurons     : %s' % microNN.NeuronsCount)
print('  - Connections : %s' % microNN.ConnectionsCount)
print()

# ----------------------------------------------------------------

# Function: Gets an image with random circle position and bounds,
def getEx() :
    x      = int(random() * (w-s))
    y      = int(random() * (w-s))
    img    = Image.new('RGB', (w, w), (0, 0, 0))
    draw   = ImageDraw.Draw(img)
    draw.ellipse((x, y, x+s, y+s), fill=(255, 255, 255))
    px     = [ [ img.getpixel((x, y))[0]/255-0.5 for y in range(w) ] for x in range(w) ]
    bounds = [ x/w, y/w, (x+s)/w, (y+s)/w ]
    return px, bounds

# ----------------------------------------------------------------

# 1) Learning from examples,
for i in range(explCount) :
    px, bounds = getEx()
    microNN.AddExample(px, bounds)
try :
    microNN.LearnExamples(minibatchSize=batchSize)
    print(' --> Ok.')
except KeyboardInterrupt :
    print(' --> Aborted!')
print()

# ----------------------------------------------------------------

# 2) Tests from evaluations and saves the PNG image of results,
img  = Image.new('RGB', (w, evalCount*(w+2)), (255, 255, 255))
draw = ImageDraw.Draw(img)
for i in range(evalCount) :
    px, __  = getEx()    
    bounds  = microNN.Predict(px)
    for x in range(len(px)) :
        for y in range(len(px[x])) :
            col = round( (px[x][y]+0.5) * 255 )
            img.putpixel((x, i*(w+2)+y), (col, col, col))
    draw.rectangle( ( round(bounds[0]*w),
                      i*(w+2)+round(bounds[1]*w),
                      round(bounds[2]*w),
                      i*(w+2)+round(bounds[3]*w) ),
                    outline=(255, 0, 0) )
img.save('bound conv.png')
print('Image saved!')
print()



