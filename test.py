
from microNN import MicroNN

microNN = MicroNN()

l1 = microNN.AddInputLayer  ( dimensions  = MicroNN.Init1D(2),
                              shape       = MicroNN.Shape.Bool )

l2 = microNN.AddLayer       ( dimensions  = MicroNN.Init1D(2),
                              shape       = MicroNN.Shape.Neuron,
                              activation  = MicroNN.Activation.Gaussian,
                              initializer = MicroNN.LogisticInitializer(MicroNN.Initializer.HeUniform),
                              connStruct  = MicroNN.FullyConnected )

l3 = microNN.AddOutputLayer ( dimensions  = MicroNN.Init1D(1),
                              shape       = MicroNN.Shape.Bool,
                              activation  = MicroNN.Activation.Heaviside,
                              initializer = MicroNN.ReLUInitializer(MicroNN.Initializer.HeUniform),
                              connStruct  = MicroNN.FullyConnected )

microNN.InitWeights()

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
learned = microNN.LearnExamples()

#microNN.SaveToJSONFile("XOR.json")

print()
print( "LEARNED [%s] :" % learned)
print( "  - False XOR False = %s" % microNN.Predict([False, False])[0] )
print( "  - False XOR True  = %s" % microNN.Predict([False, True] )[0] )
print( "  - True  XOR True  = %s" % microNN.Predict([True , True] )[0] )
print( "  - True  XOR False = %s" % microNN.Predict([True , False])[0] )
print()

