
# -*- coding: utf-8 -*-

from microNN    import MicroNN
from PIL        import Image, ImageDraw
from random     import random

# ----------------------------------------------------------------

etaLR           = 3.0   # Learning rate
explCount       = 50    # Number of examples
batchSize       = 10    # Size of minibatchs
evalCount       = 25    # Number of evaluations

w               = 32    # Sides size of images
s               = 10    # Sides size of rectangles

# ----------------------------------------------------------------

microNN              = MicroNN()
microNN.LearningRate = etaLR

l1 = microNN.AddInputLayer    ( dimensions    = MicroNN.Init2D(w, w),
                                shape         = MicroNN.Shape.Byte )

l2 = microNN.AddConv2DLayer   ( filtersCount  = 5,
                                filtersDepth  = 3,
                                convSize      = 3,
                                stride        = 2,
                                shape         = MicroNN.Shape.Neuron,
                                activation    = MicroNN.Activation.LeakyReLU,
                                initializer   = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform) )

l3 = microNN.AddConv2DLayer   ( filtersCount  = 3,
                                filtersDepth  = 1,
                                convSize      = 3,
                                stride        = 2,
                                shape         = MicroNN.Shape.Neuron,
                                activation    = MicroNN.Activation.LeakyReLU,
                                initializer   = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform) )

l4 = microNN.AddDeconv2DLayer ( filtersCount  = 5,
                                filtersDepth  = 3,
                                convSize      = 3,
                                deconvSize    = 2,
                                shape         = MicroNN.Shape.Neuron,
                                activation    = MicroNN.Activation.LeakyReLU,
                                initializer   = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform) )

l5 = microNN.AddDeconv2DLayer ( filtersCount  = 3,
                                filtersDepth  = 1,
                                convSize      = 3,
                                deconvSize    = 2,
                                shape         = MicroNN.Shape.Byte,
                                activation    = MicroNN.Activation.Sigmoid,
                                initializer   = MicroNN.LogisticInitializer(MicroNN.Initializer.XavierUniform) )

microNN.InitWeights()

print()
print('MicroNN:')
print('  - Layers      : %s' % microNN.LayersCount)
print('  - Neurons     : %s' % microNN.NeuronsCount)
print('  - Connections : %s' % microNN.ConnectionsCount)
print()

# ----------------------------------------------------------------

# Function: Gets an image with random rectangle position,
def getPx() :
    x    = int(random() * (w-s))
    y    = int(random() * (w-s))
    img  = Image.new('RGB', (w, w), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle((x, y, x+s, y+s), fill=(255, 255, 255))
    return [ [ img.getpixel((x, y))[0] for y in range(w) ] for x in range(w) ]

# ----------------------------------------------------------------

# Function: Draws 2D values of a layer output in image,
def drawValues(layer, depthIdx, img, posX, posY) :
    values = layer.GetOutputValues()
    valMin = None
    valMax = None
    # Calculation of limits for the calibration of values,
    for x in range(len(values)) :
        for y in range(len(values[x])) :
            val    = values[x][y] if depthIdx is None else values[x][y][depthIdx]
            valMin = val if valMin is None else min(val, valMin)
            valMax = val if valMax is None else max(val, valMax)
    # Draws the calibrated values of the layer output from specified depth,
    for x in range(len(values)) :
        for y in range(len(values[x])) :
            val = values[x][y] if depthIdx is None else values[x][y][depthIdx]
            col = round( (val-valMin) * 255 / (valMax-valMin) )
            img.putpixel((posX+x, posY+y), (col, col, col))

# ----------------------------------------------------------------

# 1) Learning from examples,
for i in range(explCount) :
    px   = getPx()
    inpt = [ [   px[x][y]   for y in range(w)] for x in range(w) ]
    outp = [ [ [ px[x][y] ] for y in range(w)] for x in range(w) ]
    microNN.AddExample(inpt, outp)
try :
    microNN.LearnExamples(minibatchSize=batchSize)
    print('\n ---> Ok!')
except KeyboardInterrupt :
    print('\n ---> Aborted by user...')
except Exception as ex :
    print('\n ---> Aborted by exception: %s...' % ex)
print()

# ----------------------------------------------------------------

# 2) Tests from evaluations and saves the PNG image of results,
img = Image.new('RGB', (5*(w+2), evalCount*(w+2)), (255, 255, 255))
for i in range(evalCount) :
    # Feed forward,
    px    = getPx()
    inpt  = [[px[x][y] for y in range(w)] for x in range(w)]
    outp  = microNN.Predict(inpt)
    # Draws values of layers outputs,
    drawValues(l1, None, img,       0, i*(w+2))
    drawValues(l2,    0, img,     w+2, i*(w+2))
    drawValues(l3,    0, img, 2*(w+2), i*(w+2))
    drawValues(l4,    0, img, 3*(w+2), i*(w+2))
    drawValues(l5,    0, img, 4*(w+2), i*(w+2))
img.save('autoencoder conv.png')
print('Image saved!')
print()
