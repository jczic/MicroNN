
from microNN import MicroNN

microNN = MicroNN()

microNN.AddInputLayer  ( dimensions = MicroNN.Init1D(shapesCount=2),
                         shape      = MicroNN.Shape.Bool )

microNN.AddLayer       ( dimensions = MicroNN.Init1D(shapesCount=2),
                         shape      = MicroNN.Shape.Neuron,
                         activation = MicroNN.GaussianActivation(),
                         connStruct = MicroNN.FullyConnected )

microNN.AddOutputLayer ( dimensions = MicroNN.Init1D(shapesCount=1),
                         shape      = MicroNN.Shape.Bool,
                         activation = MicroNN.LeakyReLUActivation(),
                         connStruct = MicroNN.FullyConnected )

#microNN = MicroNN.LoadFromJSONFile('XOR.json')

print()
print('MicroNN :')
print('  - Layers      : %s' % microNN.LayersCount)
print('  - Neurons     : %s' % microNN.NeuronsCount)
print('  - Connections : %s' % microNN.ConnectionsCount)
print()

microNN.AddExample( [False, False], [False] )
microNN.AddExample( [False, True ], [True ] )
microNN.AddExample( [True , True ], [False] )
microNN.AddExample( [True , False], [True ] )
microNN.LearnExamples()

#microNN.SaveToJSONFile("XOR.json")

print()
print( "LEARNED :" )
print( "  - False XOR False = %s" % microNN.Predict([False, False])[0] )
print( "  - False XOR True  = %s" % microNN.Predict([False, True] )[0] )
print( "  - True  XOR True  = %s" % microNN.Predict([True , True] )[0] )
print( "  - True  XOR False = %s" % microNN.Predict([True , False])[0] )
print()

