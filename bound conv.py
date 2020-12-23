
from microNN import MicroNN
from PIL     import Image, ImageDraw
from random  import random

w = 64
s = 10

microNN = MicroNN()

microNN.AddInputLayer  ( dimensions  = MicroNN.Init2D(w, w),
                         shape       = MicroNN.Shape.Neuron )

microNN.AddConv2DLayer ( filtersCount  = 5,
                         filtersDepth  = 3,
                         convSize      = 5,
                         stride        = 2,
                         shape         = MicroNN.Shape.Neuron,
                         activation    = MicroNN.Activation.LeakyReLU,
                         initializer   = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform) )

microNN.AddConv2DLayer ( filtersCount  = 5,
                         filtersDepth  = 3,
                         convSize      = 5,
                         stride        = 2,
                         shape         = MicroNN.Shape.Neuron,
                         activation    = MicroNN.Activation.LeakyReLU,
                         initializer   = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform) )

microNN.AddConv2DLayer ( filtersCount  = 3,
                         filtersDepth  = 2,
                         convSize      = 5,
                         stride        = 2,
                         shape         = MicroNN.Shape.Neuron,
                         activation    = MicroNN.Activation.LeakyReLU,
                         initializer   = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform) )

microNN.AddLayer       ( dimensions  = MicroNN.Init1D(10),
                         shape       = MicroNN.Shape.Neuron,
                         activation  = MicroNN.Activation.LeakyReLU,
                         initializer = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform),
                         connStruct  = MicroNN.FullyConnected )

microNN.AddLayer       ( dimensions  = MicroNN.Init1D(4),
                         shape       = MicroNN.Shape.Neuron,
                         activation  = MicroNN.Activation.TanH,
                         initializer = MicroNN.LogisticInitializer(MicroNN.Initializer.HeNormal),
                         connStruct  = MicroNN.FullyConnected )

microNN.InitWeights()

print()
print('MicroNN:')
print('  - Layers      : %s' % microNN.LayersCount)
print('  - Neurons     : %s' % microNN.NeuronsCount)
print('  - Connections : %s' % microNN.ConnectionsCount)
print()

def getEx() :
    x     = int(random() * (w-s))
    y     = int(random() * (w-s))
    img   = Image.new('RGB', (w, w), (0, 0, 0))
    draw  = ImageDraw.Draw(img)
    draw.ellipse((x, y, x+s, y+s), fill=(255, 255, 255))
    px    = [ [ img.getpixel((x, y))[0]/255-0.5 for y in range(w) ] for x in range(w) ]
    bound = [ x/w, y/w, (x+s)/w, (y+s)/w ]
    return px, bound

for i in range(100) :
    px, bound = getEx()
    microNN.AddExample(px, bound)

try :
    microNN.LearnExamples(minibatchSize=5)
    print(' --> Ok.')
except KeyboardInterrupt :
    print(' --> Aborted!')

print()

evalCount = 25

img  = Image.new('RGB', (w, evalCount*(w+2)), (255, 255, 255))
draw = ImageDraw.Draw(img)
for i in range(evalCount) :
    px, __ = getEx()    
    bound  = microNN.Predict(px)
    for x in range(len(px)) :
        for y in range(len(px[x])) :
            col = round( (px[x][y]+0.5) * 255 )
            img.putpixel((x, i*(w+2)+y), (col, col, col))
    draw.rectangle( ( round(bound[0]*w),
                      i*(w+2)+round(bound[1]*w),
                      round(bound[2]*w),
                      i*(w+2)+round(bound[3]*w) ),
                    outline=(255, 0, 0) )
img.save('bound conv.png')
print('Image saved!')
print()



