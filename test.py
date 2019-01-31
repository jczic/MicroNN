
from microNN import MicroNN

"""
microNN = MicroNN()

l1 = MicroNN.InputLayer(microNN, [100, 100], MicroNN.ColorShape())
l2 = MicroNN.Layer(microNN, [30, 30], MicroNN.ColorShape(), 'sigmoid', MicroNN.LocallyConnStruct(MicroNN.LocallyConnStruct.ROLE_DECREASE, 2))
l3 = MicroNN.Layer(microNN, [10, 10], MicroNN.ColorShape(), 'sigmoid', MicroNN.LocallyConnStruct(MicroNN.LocallyConnStruct.ROLE_DECREASE, 1))
l4 = MicroNN.Layer(microNN, [2, 2], MicroNN.ValueShape(MicroNN.PercentValueType()), 'sigmoid', MicroNN.LocallyConnStruct(MicroNN.LocallyConnStruct.ROLE_DECREASE))
l5 = MicroNN.OutputLayer(microNN, [100], MicroNN.ValueShape(MicroNN.PercentValueType()), 'sigmoid', MicroNN.FullyConnStruct())

print()
print('-------------------------------------')
print(microNN.SaveToJSONFile("test2.json"))
print('-------------------------------------')
print()
"""

microNN = MicroNN()

microNN.AddInputLayer  ( dimensions     = MicroNN.Init1D(shapesCount=2),
                         shape          = MicroNN.Shape.Bool )

microNN.AddLayer       ( dimensions     = MicroNN.Init1D(shapesCount=2),
                         shape          = MicroNN.Shape.Neuron,
                         activationFunc = MicroNN.ActFunctions.Gaussian,
                         connStruct     = MicroNN.FullyConnected )

microNN.AddOutputLayer ( dimensions     = MicroNN.Init1D(shapesCount=1),
                         shape          = MicroNN.Shape.Bool,
                         activationFunc = MicroNN.ActFunctions.LeakyReLU,
                         connStruct     = MicroNN.FullyConnected )

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

print()
print( "LEARNED :" )
print( "  - False XOR False = %s" % microNN.Predict([False, False])[0] )
print( "  - False XOR True  = %s" % microNN.Predict([False, True] )[0] )
print( "  - True  XOR True  = %s" % microNN.Predict([True , True] )[0] )
print( "  - True  XOR False = %s" % microNN.Predict([True , False])[0] )
print()

