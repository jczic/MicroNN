"""
The MIT License (MIT)
Copyright © 2020 Jean-Christophe Bos (jczic.bos@gmail.com)
"""


from   math import inf, sqrt, pi, exp, ceil, log, sin, cos
from   time import time
import random
import json

# -------------------------------------------------------------------------
# --( Class : MicroNNException )-------------------------------------------
# -------------------------------------------------------------------------

class MicroNNException(Exception) :
    pass

# -------------------------------------------------------------------------
# --( Class : MicroNN )----------------------------------------------------
# -------------------------------------------------------------------------

class MicroNN :

    VERSION                        = '1.0.5'

    DEFAULT_LEARNING_RATE          = 1.0
    DEFAULT_PLASTICITY_STRENGTHING = 1.0

    # -------------------------------------------------------------------------
    # --( Class : ValueTypeException )-----------------------------------------
    # -------------------------------------------------------------------------

    class ValueTypeException(Exception) :
        pass

    # -------------------------------------------------------------------------
    # --( Abstract class : ValueType )-----------------------------------------
    # -------------------------------------------------------------------------

    class ValueType :

        # -[ Constructor ]--------------------------------------

        def __init__(self) :
            if type(self) is MicroNN.ValueType :
                raise MicroNN.ValueTypeException('"ValueType" is an abstract class and cannot be instancied.')

        # -[ Methods ]------------------------------------------

        def FromAnalog(self, value) :
            raise MicroNN.ValueTypeException('"FromAnalog" method must be implemented.')

        # ------------------------------------------------------

        def ToAnalog(self, value) :
            raise MicroNN.ValueTypeException('"ToAnalog" method must be implemented.')

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            raise MicroNN.ValueTypeException('"GetAsDataObject" method must be implemented.')

        # ------------------------------------------------------

        @staticmethod
        def CreateFromDataObject(o) :
            try :
                valType = o['Type']
                if   valType == 'NeuronValueType' :
                    return MicroNN.NeuronValueType()
                elif valType == 'FloatValueType' :
                    return MicroNN.FloatValueType(o['MinValue'], o['MaxValue'])
                elif valType == 'IntValueType' :
                    return MicroNN.IntValueType(o['MinValue'], o['MaxValue'])
                elif valType == 'BoolValueType' :
                    return MicroNN.BoolValueType()
                elif valType == 'ByteValueType' :
                    return MicroNN.ByteValueType()
                elif valType == 'PercentValueType' :
                    return MicroNN.PercentValueType()
                else :
                    raise Exception()
            except :
                raise MicroNN.ValueTypeException('Data object is not valid.')

    # -------------------------------------------------------------------------
    # --( Class : NeuronValueType )--------------------------------------------
    # -------------------------------------------------------------------------

    class NeuronValueType(ValueType) :

        # -[ Methods ]------------------------------------------

        def FromAnalog(self, value) :
            return value

        # ------------------------------------------------------

        def ToAnalog(self, value) :
            try :
                return float(value)
            except :
                raise MicroNN.ValueTypeException('Value is not correct.')

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Type' : 'NeuronValueType' }

    # -------------------------------------------------------------------------
    # --( Class : FloatValueType )---------------------------------------------
    # -------------------------------------------------------------------------

    class FloatValueType(ValueType) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, minValue, maxValue) :
            if type(minValue) not in (float, int) or \
               type(maxValue) not in (float, int) or \
               maxValue <= minValue :
                raise MicroNN.ValueTypeException('"minvalue" and "maxvalue" are not correct.')
            super().__init__()
            self._minValue = minValue
            self._maxValue = maxValue

        # -[ Methods ]------------------------------------------

        def FromAnalog(self, value) :
            return self._minValue + ( value * (self._maxValue - self._minValue) )

        # ------------------------------------------------------

        def ToAnalog(self, value) :
            if type(value) not in (float, int) :
                raise MicroNN.ValueTypeException('Value must be of "float" or "int" type.')
            if value < self._minValue or value > self._maxValue :
                raise MicroNN.ValueTypeException('Value must be >= %s and <= %s.' % (self._minValue, self._maxValue))
            return float(value - self._minValue) / (self._maxValue - self._minValue)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return {
                'Type'     : 'FloatValueType',
                'MinValue' : self._minValue,
                'MaxValue' : self._maxValue
            }

    # -------------------------------------------------------------------------
    # --( Class : IntValueType )-----------------------------------------------
    # -------------------------------------------------------------------------

    class IntValueType(FloatValueType) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, minValue, maxValue) :
            if type(minValue) is not int or type(maxValue) is not int :
                raise MicroNN.ValueTypeException('"minvalue" and "maxvalue" are not correct.')
            super().__init__(minValue, maxValue)

        # -[ Methods ]------------------------------------------

        def FromAnalog(self, value) :
            return round(super().FromAnalog(value))

        # ------------------------------------------------------

        def ToAnalog(self, value) :
            if type(value) is not int :
                raise MicroNN.ValueTypeException('Value must be of "int" type.')
            return super().ToAnalog(value)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return {
                'Type'     : 'IntValueType',
                'MinValue' : int(self._minValue),
                'MaxValue' : int(self._maxValue)
            }

    # -------------------------------------------------------------------------
    # --( Class : BoolValueType )----------------------------------------------
    # -------------------------------------------------------------------------

    class BoolValueType(IntValueType) :

        # -[ Constructor ]--------------------------------------

        def __init__(self) :
            super().__init__(0, 1)

        # -[ Methods ]------------------------------------------

        def FromAnalog(self, value) :
            return bool(super().FromAnalog(value))

        # ------------------------------------------------------

        def ToAnalog(self, value) :
            if type(value) is not bool :
                raise MicroNN.ValueTypeException('Value must be of "bool" type.')
            return super().ToAnalog(int(value))

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Type' : 'BoolValueType' }

    # -------------------------------------------------------------------------
    # --( Class : ByteValueType )----------------------------------------------
    # -------------------------------------------------------------------------

    class ByteValueType(IntValueType) :

        # -[ Constructor ]--------------------------------------

        def __init__(self) :
            super().__init__(0, 255)

        # -[ Methods ]------------------------------------------

        def GetAsDataObject(self) :
            return { 'Type' : 'ByteValueType' }

    # -------------------------------------------------------------------------
    # --( Class : PercentValueType )-------------------------------------------
    # -------------------------------------------------------------------------

    class PercentValueType(FloatValueType) :

        # -[ Constructor ]--------------------------------------

        def __init__(self) :
            super().__init__(0, 100)

        # -[ Methods ]------------------------------------------

        def GetAsDataObject(self) :
            return { 'Type' : 'PercentValueType' }

    # -------------------------------------------------------------------------
    # --( Class : ShapeException )---------------------------------------------
    # -------------------------------------------------------------------------

    class ShapeException(Exception) :
        pass

    # -------------------------------------------------------------------------
    # --( Abstract class : Shape )---------------------------------------------
    # -------------------------------------------------------------------------

    class Shape :

        # -[ Constructor ]--------------------------------------

        def __init__(self, flattenLen, valueType=None) :
            if type(self) is MicroNN.Shape :
                raise MicroNN.ShapeException('"Shape" is an abstract class and cannot be instancied.')
            if valueType is not None and not isinstance(valueType, MicroNN.ValueType) :
                raise MicroNN.ShapeException('"valueType" is not correct.')
            self._flattenLen = flattenLen
            self._valueType  = valueType if valueType else MicroNN.NeuronValueType()

        # -[ Methods ]------------------------------------------

        def Flatten(self, data) :
            raise MicroNN.ShapeException('"Flatten" method must be implemented.')

        # ------------------------------------------------------

        def Unflatten(self, data) :
            raise MicroNN.ShapeException('"Unflatten" method must be implemented.')

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            raise MicroNN.ShapeException('"GetAsDataObject" method must be implemented.')

        # ------------------------------------------------------

        @staticmethod
        def CreateFromDataObject(o) :
            try :
                shapeType = o['Type']
                if 'ValueType' in o :
                    valueType = MicroNN.ValueType.CreateFromDataObject(o['ValueType'])
                if   shapeType == 'ValueShape' :
                    return MicroNN.ValueShape(valueType)
                elif shapeType == 'VectorShape' :
                    return MicroNN.VectorShape(o['Size'], valueType)
                elif shapeType == 'ColorShape' :
                    return MicroNN.ColorShape()
                elif shapeType == 'Matrix2DShape' :
                    return MicroNN.Matrix2DShape(o['XSize'], o['YSize'], valueType)
                elif shapeType == 'Matrix3DShape' :
                    return MicroNN.Matrix3DShape(o['XSize'], o['YSize'], o['ZSize'], valueType)
                else :
                    raise Exception()
            except :
                raise MicroNN.ShapeException('Data object is not valid.')

        # -[ Properties ]---------------------------------------

        @property
        def FlattenLen(self) :
            return self._flattenLen

        @property
        def ValueType(self) :
            return self._valueType

    # -------------------------------------------------------------------------
    # --( Class : ValueShape )-------------------------------------------------
    # -------------------------------------------------------------------------

    class ValueShape(Shape) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, valueType=None) :
            super().__init__(1, valueType)

        # -[ Methods ]------------------------------------------

        def Flatten(self, data) :
            return [ self._valueType.ToAnalog(data) ]

        # ------------------------------------------------------

        def Unflatten(self, data) :
            return self._valueType.FromAnalog(data[0])

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return {
                'Type'      : 'ValueShape',
                'ValueType' : self._valueType.GetAsDataObject()
            }

    # -------------------------------------------------------------------------
    # --( Class : VectorShape )------------------------------------------------
    # -------------------------------------------------------------------------

    class VectorShape(Shape) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, size, valueType=None) :
            if type(size) is not int or size <= 0 :
                raise MicroNN.ShapeException('"size" must be of "int" type greater than zero.')
            super().__init__(size, valueType)
            self._size = size

        # -[ Methods ]------------------------------------------

        def Flatten(self, data) :
            if type(data) not in (list, tuple) or len(data) != self._size :
                raise MicroNN.ShapeException('VectorShape values must be a list or a tuple of size %s.' % self._size)
            return [ self._valueType.ToAnalog(d) for d in data ]

        # ------------------------------------------------------

        def Unflatten(self, data) :
            return [ self._valueType.FromAnalog(d) for d in data ]

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return {
                'Type'      : 'VectorShape',
                'Size'      : self._size,
                'ValueType' : self._valueType.GetAsDataObject()
            }

    # -------------------------------------------------------------------------
    # --( Class : ColorShape )-------------------------------------------------
    # -------------------------------------------------------------------------

    class ColorShape(Shape) :

        # -[ Constructor ]--------------------------------------

        def __init__(self) :
            super().__init__(3, MicroNN.ByteValueType())

        # -[ Methods ]------------------------------------------

        def Flatten(self, data) :
            if type(data) is not tuple or len(data) != 3 :
                raise MicroNN.ShapeException('ColorShape values must be a tuple of (r, g, b).')
            return [ self._valueType.ToAnalog(d) for d in data ]

        # ------------------------------------------------------

        def Unflatten(self, data) :
            return tuple( self._valueType.FromAnalog(d) for d in data )

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Type' : 'ColorShape' }

    # -------------------------------------------------------------------------
    # --( Class : Matrix2DShape )----------------------------------------------
    # -------------------------------------------------------------------------

    class Matrix2DShape(Shape) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, xSize, ySize, valueType=None) :
            if type(xSize) is not int or xSize <= 0 or \
               type(ySize) is not int or ySize <= 0 :
                raise MicroNN.ShapeException('"xSize" and "ySize" must be of "int" type greater than zero.')
            super().__init__(xSize*ySize, valueType)
            self._xSize = xSize
            self._ySize = ySize

        # -[ Methods ]------------------------------------------

        def _matrixErr(self) :
            raise MicroNN.ShapeException( 'Matrix2DShape values must be a list or a tuple 2D matrix of %sx%s.'
                                          % (self._xSize, self._ySize) )

        # ------------------------------------------------------

        def Flatten(self, data) :
            if type(data) not in (list, tuple) or len(data) != self._xSize :
                self._matrixErr()
            for dataX in data :
                if type(dataX) not in (list, tuple) or len(dataX) != self._ySize :
                    self._matrixErr()
            return [ self._valueType.ToAnalog(data[x][y])
                     for x in range(self._xSize)
                     for y in range(self._ySize) ]

        # ------------------------------------------------------

        def Unflatten(self, data) :
            return [ [ self._valueType.FromAnalog(data[ x*self._ySize + y ])
                       for y in range(self._ySize) ]
                     for x in range(self._xSize) ]

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return {
                'Type'      : 'Matrix2DShape',
                'XSize'     : self._xSize,
                'YSize'     : self._ySize,
                'ValueType' : self._valueType.GetAsDataObject()
            }

    # -------------------------------------------------------------------------
    # --( Class : Matrix3DShape )----------------------------------------------
    # -------------------------------------------------------------------------

    class Matrix3DShape(Shape) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, xSize, ySize, zSize, valueType=None) :
            if type(xSize) is not int or xSize <= 0 or \
               type(ySize) is not int or ySize <= 0 or \
               type(zSize) is not int or zSize <= 0 :
                raise MicroNN.ShapeException('"xSize", "ySize" and "zSize" must be of "int" type greater than zero.')
            super().__init__(xSize*ySize*zSize, valueType)
            self._xSize = xSize
            self._ySize = ySize
            self._zSize = zSize

        # -[ Methods ]------------------------------------------

        def _matrixErr(self) :
            raise MicroNN.ShapeException( 'Matrix3DShape values must be a list or a tuple 3D matrix of %sx%sx%s.'
                                          % (self._xSize, self._ySize, self._zSize) )

        # ------------------------------------------------------

        def Flatten(self, data) :
            if type(data) not in (list, tuple) or len(data) != self._xSize :
                self._matrixErr()
            for dataX in data :
                if type(dataX) not in (list, tuple) or len(dataX) != self._ySize :
                    self._matrixErr()
                for dataY in dataX :
                    if type(dataY) not in (list, tuple) or len(dataY) != self._zSize :
                        self._matrixErr()
            return [ self._valueType.ToAnalog(data[x][y][z])
                     for x in range(self._xSize)
                     for y in range(self._ySize)
                     for z in range(self._zSize) ]

        # ------------------------------------------------------

        def Unflatten(self, data) :
            return [ [ [ self._valueType.FromAnalog(
                         data[ x*self._ySize*self._zSize + y*self._zSize + z ] )
                         for z in range(self._zSize) ]
                       for y in range(self._ySize) ]
                     for x in range(self._xSize) ]

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return {
                'Type'      : 'Matrix3DShape',
                'XSize'     : self._xSize,
                'YSize'     : self._ySize,
                'ZSize'     : self._zSize,
                'ValueType' : self._valueType.GetAsDataObject()
            }

    # -------------------------------------------------------------------------
    # --( Class : ConnectionException )----------------------------------------
    # -------------------------------------------------------------------------

    class ConnectionException(Exception) :
        pass

    # -------------------------------------------------------------------------
    # --( Class : Connection )-------------------------------------------------
    # -------------------------------------------------------------------------

    class Connection :

        # -[ Constructor ]--------------------------------------

        def __init__(self, neuronSrc, neuronDst, weight=None) :
            if not isinstance(neuronSrc, MicroNN.Neuron) or \
               not isinstance(neuronDst, MicroNN.Neuron) :
                raise MicroNN.ConnectionException('"neuronSrc" and "neuronDst" must be of Neuron type.')
            neuronSrc.OutputConnections.append(self)
            neuronDst.InputConnections.append(self)
            self._neuronSrc      = neuronSrc
            self._neuronDst      = neuronDst
            self._deltaError     = 0.0
            self._weight         = weight if weight else 0.0
            self._momentumWeight = 0.0

        # -[ Methods ]------------------------------------------

        def BackPropagateSignalError(self, signalError) :
            self._neuronSrc.Error += signalError * self._weight
            self._deltaError      += signalError * self._neuronSrc.Output

        # ------------------------------------------------------

        def UpdateWeight(self, batchSize, learningRate, plasticityStrengthing) :
            deltaWeight           = (learningRate/batchSize) * self._deltaError
            self._weight         -= deltaWeight \
                                  + self._momentumWeight * plasticityStrengthing
            self._momentumWeight  = deltaWeight
            self._deltaError      = 0.0

        # -[ Properties ]---------------------------------------

        @property
        def NeuronSrc(self) :
            return self._neuronSrc

        @property
        def NeuronDst(self) :
            return self._neuronDst

        @property
        def Weight(self) :
            return self._weight
        @Weight.setter
        def Weight(self, value) :
            if not isinstance(value, float) :
                raise MicroNN.ConnectionException('"value" must be of "float" type.')
            self._weight = value

    # -------------------------------------------------------------------------
    # --( Class : NeuronException )--------------------------------------------
    # -------------------------------------------------------------------------

    class NeuronException(Exception) :
        pass

    # -------------------------------------------------------------------------
    # --( Class : Neuron )-----------------------------------------------------
    # -------------------------------------------------------------------------

    class Neuron :

        # -[ Constructor ]--------------------------------------

        def __init__(self, index, activation=None) :
            if not isinstance(index, int) :
                raise MicroNN.NeuronException('"index" must be of "int" type.')
            if activation is not None and not isinstance(activation, MicroNN.Activation) :
                raise MicroNN.NeuronException('"activation" must be of Activation type or "None.')
            self._index               = index
            self._activation          = activation
            self._inputConnections    = [ ]
            self._outputConnections   = [ ]
            self._input               = 0.0
            self._output              = 0.0
            self._error               = 0.0

        # -[ Methods ]------------------------------------------

        def ComputeInput(self) :
            #self._error = 0.0
            self._input = 0.0
            for conn in self._inputConnections :
                self._input += conn.NeuronSrc.Output * conn.Weight

        # ------------------------------------------------------

        def ComputeOutput(self) :
            if not self._activation :
                raise MicroNN.NeuronException('No activation available for this neuron.')
            self._output = self._activation.Get(self)

        # ------------------------------------------------------

        def SetErrorFromTarget(self, targetValue) :
            if type(targetValue) not in (float, int) :
                raise MicroNN.NeuronException('"targetValue" must be of "float" or "int" type.')
            self._error = self._output - float(targetValue)

        # ------------------------------------------------------

        def GetSignalError(self) :
            if not self._activation :
                raise MicroNN.NeuronException('No activation available for this neuron.')
            return self._error * self._activation.GetDerivative(self)

        # ------------------------------------------------------

        def BackPropagateError(self) :
            signalError = self.GetSignalError()
            for conn in self._inputConnections :
                conn.BackPropagateSignalError(signalError)
            self._error = 0.0

        # ------------------------------------------------------

        def UpdateConnectionsWeight(self, batchSize, learningRate, plasticityStrengthing) :
            for conn in self._inputConnections :
                conn.UpdateWeight(batchSize, learningRate, plasticityStrengthing)

        # ------------------------------------------------------

        def GetInputConnsAsDataObject(self) :
            o = [ ]
            for conn in self._inputConnections :
                o.append([conn.NeuronSrc.Index, conn.Weight])
            return o

        # -[ Properties ]---------------------------------------

        @property
        def InputConnections(self) :
            return self._inputConnections

        @property
        def InputConnectionsCount(self) :
            return len(self._inputConnections)

        @property
        def OutputConnections(self) :
            return self._outputConnections

        @property
        def OutputConnectionsCount(self) :
            return len(self._outputConnections)

        @property
        def Input(self) :
            return self._input
        @Input.setter
        def Input(self, value) :
            if type(value) not in (float, int) :
                raise MicroNN.NeuronException('"value" must be of "float" or "int" type.')
            self._input = float(value)

        @property
        def Output(self) :
            return self._output
        @Output.setter
        def Output(self, value) :
            if type(value) not in (float, int) :
                raise MicroNN.NeuronException('"value" must be of "float" or "int" type.')
            self._output = float(value)

        @property
        def Error(self) :
            return self._error
        @Error.setter
        def Error(self, value) :
            if type(value) not in (float, int) :
                raise MicroNN.NeuronException('"value" must be of "float" or "int" type.')
            self._error = float(value)

        @property
        def Index(self) :
            return self._index

    # -------------------------------------------------------------------------
    # --( Class : Bias )-------------------------------------------------------
    # -------------------------------------------------------------------------

    class Bias(Neuron) :
 
        # -[ Constructor ]--------------------------------------
 
        def __init__(self, index, value=1.0) :
            super().__init__(index)
            self.Value = value
 
        # -[ Properties ]---------------------------------------
 
        @property
        def Value(self) :
            return self._output
        @Value.setter
        def Value(self, value) :
            self._output = value

    # -------------------------------------------------------------------------
    # --( Class : LayerException )---------------------------------------------
    # -------------------------------------------------------------------------

    class LayerException(Exception) :
        pass

    # -------------------------------------------------------------------------
    # --( Abstract class : BaseLayer )-----------------------------------------
    # -------------------------------------------------------------------------
 
    class BaseLayer :
 
        # -[ Constructor ]--------------------------------------
 
        def __init__( self,
                      parentMicroNN,
                      dimensions,
                      shape,
                      activation    = None,
                      initializer   = None,
                      biasValue     = 1.0 ) :
            if type(self) is MicroNN.BaseLayer :
                raise MicroNN.LayerException('"BaseLayer" is an abstract class and cannot be instancied.')
            if not isinstance(parentMicroNN, MicroNN) :
                raise MicroNN.LayerException('"parentMicroNN" must be of MicroNN type.')
            if type(dimensions) not in (list, tuple) or len(dimensions) == 0 :
                raise MicroNN.LayerException('"dimensions" must be a not empty list or tuple.')
            for dimSize in dimensions :
                if type(dimSize) is not int or dimSize <= 0 :
                    raise MicroNN.LayerException('"dimensions" must contain only "int" types greater than zero.')
            if not isinstance(shape, MicroNN.Shape) :
                raise MicroNN.LayerException('"shape" must be of Shape type.')
            if activation is not None :
                if isinstance(self, MicroNN.InputLayer) :
                    raise MicroNN.LayerException('"activation" must be "None" for an input layer.')
                if not isinstance(activation, MicroNN.Activation) :
                    raise MicroNN.LayerException('"activation" must be of Activation type.')
                aMin, aMax = activation.GetRangeValues()
                if aMax-aMin == inf and type(shape.ValueType) is not MicroNN.NeuronValueType :
                    raise MicroNN.LayerException( 'Shape value type (%s) is not compatible with activation (%s).' 
                                                  % (type(shape.ValueType).__name__, type(activation).__name__) )
            elif not isinstance(self, MicroNN.InputLayer) :
                raise MicroNN.LayerException('"activation" must be defined for this layer type.')
            if initializer is not None and not isinstance(initializer, MicroNN.Initializer) :
                raise MicroNN.LayerException('"initializer" must be "None" or of Initializer type.')
            if parentMicroNN.Layers :
                topLayer = parentMicroNN.Layers[len(parentMicroNN.Layers)-1]
            else :
                topLayer = None
            if topLayer is not None :
                if isinstance(self, MicroNN.InputLayer) :
                    raise MicroNN.LayerException('No layer must be present to add an input layer.')
                if isinstance(topLayer, MicroNN.OutputLayer) :
                    raise MicroNN.LayerException('No layer can be added after an output layer.')
            elif not isinstance(self, MicroNN.InputLayer) :
                raise MicroNN.LayerException('Only an input layer can be added as first layer.')
            self._parentMicroNN  = parentMicroNN
            self._dimensions     = dimensions
            self._shape          = shape
            self._activation     = activation
            self._initializer    = initializer
            self._topLayer       = topLayer
            self._bottomLayer    = None
            if not isinstance(self, MicroNN.InputLayer) and biasValue is not None :
                if type(biasValue) not in (float, int) :
                    raise MicroNN.LayerException('"biasValue" must be "None" or of "float" or "int" type.')
                self._bias = MicroNN.Bias(index=-1, value=biasValue)
            else :
                self._bias = None
            self._subDimNrnCount = self._getsubDimNrnCount()
            self._neuronsCount   = 0
            self._neurons        = self._recurCreateNeurons()
            self._inputConnCount = 0
            if topLayer :
                topLayer._bottomLayer = self
            parentMicroNN.Layers.append(self)

        # -[ Methods ]------------------------------------------

        def _getsubDimNrnCount(self) :
            subDim = [None] * self.DimensionsCount
            dimIdx = self.DimensionsCount-1
            count  = self._shape.FlattenLen
            while dimIdx >= 0 :
                subDim[dimIdx]  = count
                count          *= self._dimensions[dimIdx]
                dimIdx         -= 1
            return subDim

        # ------------------------------------------------------

        def _recurCreateNeurons(self, dimIdx=0) :
            if dimIdx < self.DimensionsCount :
                dim = [ ]
                for i in range(self._dimensions[dimIdx]) :
                    dim.append(self._recurCreateNeurons(dimIdx+1))
                return dim
            else :
                neurons = [ ]
                for i in range(self._shape.FlattenLen) :
                    n = MicroNN.Neuron( index      = self._neuronsCount,
                                        activation = self._activation )
                    self._neuronsCount += 1
                    if self._bias :
                        MicroNN.Connection(self._bias, n)
                    neurons.append(n)
                return neurons

        # ------------------------------------------------------

        def GetNeuronByIndex(self, index) :
            try :
                neurons = self._neurons
                for subDimCount in self._subDimNrnCount :
                    neurons = neurons[index // subDimCount]
                    index   = index % subDimCount
                return neurons[index]
            except :
                return None

        # ------------------------------------------------------

        def _recurGetNeuronsList(self, neurons, neuronsList, dimIdx=0) :
            if dimIdx < self.DimensionsCount :
                for i in range(self._dimensions[dimIdx]) :
                    self._recurGetNeuronsList(neurons[i], neuronsList, dimIdx+1)
            else :
                for i in range(self._shape.FlattenLen) :
                    neuronsList.append(neurons[i])

        # ------------------------------------------------------

        def GetNeuronsList(self) :
            neuronsList = [ ]
            self._recurGetNeuronsList(self._neurons, neuronsList)
            return neuronsList

        # ------------------------------------------------------

        def InitWeights(self) :
            raise MicroNN.LayerException('"InitWeights" method must be implemented.')

        # ------------------------------------------------------

        def ComputeInput(self) :
            raise MicroNN.LayerException('"ComputeInput" method must be implemented.')

        # ------------------------------------------------------

        def ComputeOutput(self) :
            raise MicroNN.LayerException('"ComputeOutput" method must be implemented.')

        # ------------------------------------------------------

        def _recurBackPropagateError(self, neurons, dimIdx=0) :
            if dimIdx < self.DimensionsCount :
                for i in range(self._dimensions[dimIdx]) :
                    self._recurBackPropagateError(neurons[i], dimIdx+1)
            else :
                for i in range(self._shape.FlattenLen) :
                    neurons[i].BackPropagateError()

        # ------------------------------------------------------

        def BackPropagateError(self) :
            self._recurBackPropagateError(self._neurons)

        # ------------------------------------------------------

        def _recurUpdateConnectionsWeight(self, batchSize, neurons, dimIdx=0) :
            if dimIdx < self.DimensionsCount :
                for i in range(self._dimensions[dimIdx]) :
                    self._recurUpdateConnectionsWeight(batchSize, neurons[i], dimIdx+1)
            else :
                for i in range(self._shape.FlattenLen) :
                    neurons[i].UpdateConnectionsWeight( batchSize,
                                                        self._parentMicroNN.LearningRate,
                                                        self._parentMicroNN.PlasticityStrengthing )

        # ------------------------------------------------------

        def UpdateConnectionsWeight(self, batchSize) :
            if type(batchSize) is not int or batchSize <= 0 :
                raise MicroNN.LayerException('"batchSize" must be of "int" type greater than zero.')
            self._recurUpdateConnectionsWeight(batchSize, self._neurons)

        # ------------------------------------------------------

        def _recurGetOutputValues(self, neurons, dimIdx=0) :
            if dimIdx < self.DimensionsCount :
                dim = [ ]
                for i in range(self._dimensions[dimIdx]) :
                    dim.append(self._recurGetOutputValues(neurons[i], dimIdx+1))
                return dim
            else :
                scaled = (type(self._shape.ValueType) is not MicroNN.NeuronValueType)
                if scaled :
                    aMin, aMax = self._activation.GetRangeValues()
                flattenValues = [ ]
                for i in range(self._shape.FlattenLen) :
                    if scaled :
                        v = float(neurons[i].Output-aMin) / (aMax-aMin)
                    else :
                        v = neurons[i].Output
                    flattenValues.append(v)
                return self._shape.Unflatten(flattenValues)

        # ------------------------------------------------------

        def GetOutputValues(self) :
            return self._recurGetOutputValues(self._neurons)

        # ------------------------------------------------------

        def _recurSumSquareError(self, neurons, dimIdx=0) :
            x = 0.0
            if dimIdx < self.DimensionsCount :
                for i in range(self._dimensions[dimIdx]) :
                    x += self._recurSumSquareError(neurons[i], dimIdx+1)
            else :
                for i in range(self._shape.FlattenLen) :
                    x += neurons[i].Error ** 2
            return x

        # ------------------------------------------------------

        def GetMeanSquareError(self) :
            return self._recurSumSquareError(self._neurons) / self._neuronsCount

        # ------------------------------------------------------

        def GetMeanSquareErrorAsPercent(self) :
            return round( self.GetMeanSquareError() * 100 * 1000 ) / 1000

        # ------------------------------------------------------

        def _recurSumAbsoluteError(self, neurons, dimIdx=0) :
            x = 0.0
            if dimIdx < self.DimensionsCount :
                for i in range(self._dimensions[dimIdx]) :
                    x += self._recurSumAbsoluteError(neurons[i], dimIdx+1)
            else :
                for i in range(self._shape.FlattenLen) :
                    x += abs(neurons[i].Error)
            return x

        # ------------------------------------------------------

        def GetMeanAbsoluteError(self) :
            return self._recurSumAbsoluteError(self._neurons) / self._neuronsCount

        # ------------------------------------------------------

        def GetMeanAbsoluteErrorAsPercent(self) :
            return round( self.GetMeanAbsoluteError() * 100 * 1000 ) / 1000

        # ------------------------------------------------------

        def _recurGetConnsDataObject(self, neurons, connsArray=None, dimIdx=0) :
            if connsArray is None :
                connsArray = [ ]
            if dimIdx < self.DimensionsCount :
                for i in range(self._dimensions[dimIdx]) :
                    self._recurGetConnsDataObject(neurons[i], connsArray, dimIdx+1)
            else :
                for i in range(self._shape.FlattenLen) :
                    connsArray.append(neurons[i].GetInputConnsAsDataObject())
            return connsArray

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            activation = self._activation.GetAsDataObject() \
                         if self._activation else None
            bias       = { 'Value' : self._bias.Value } \
                         if self._bias else None
            conns      = self._recurGetConnsDataObject(self._neurons) \
                         if self.InputConnectionsCount > 0 else None
            return {
                'Type'            : type(self).__name__,
                'Dimensions'      : self._dimensions,
                'Shape'           : self._shape.GetAsDataObject(),
                'Activation'      : activation,
                'Bias'            : bias,
                'NeuronsCount'    : self._neuronsCount,
                'InputConnsCount' : self._inputConnCount,
                'Connections'     : conns
            }

        # ------------------------------------------------------

        @staticmethod
        def CreateFromDataObject(parentMicroNN, o) :
            try :
                layerType = o['Type']
                dims      = o['Dimensions']
                shape     = MicroNN.Shape.CreateFromDataObject(o['Shape'])
                if layerType == 'InputLayer' :
                    return MicroNN.InputLayer(parentMicroNN, dims, shape)
                else :
                    activation = MicroNN.Activation.CreateFromDataObject(o['Activation']) \
                                 if o['Activation'] else None
                    connStruct = MicroNN.DataObjectConnStruct(o['Connections']) \
                                 if o['Connections'] else None
                    biasValue  = o['Bias']['Value'] \
                                 if o['Bias'] else None
                    if layerType == 'OutputLayer' :
                        return MicroNN.OutputLayer( parentMicroNN = parentMicroNN,
                                                    dimensions    = dims,
                                                    shape         = shape,
                                                    activation    = activation,
                                                    connStruct    = connStruct,
                                                    biasValue     = biasValue )
                    elif layerType == 'Layer' :
                        return MicroNN.Layer( parentMicroNN = parentMicroNN,
                                              dimensions    = dims,
                                              shape         = shape,
                                              activation    = activation,
                                              connStruct    = connStruct,
                                              biasValue     = biasValue )
                    else :
                        raise Exception()
            except :
                raise MicroNN.LayerException('Data object is not valid.')

        # -[ Properties ]---------------------------------------

        @property
        def ParentMicroNN(self) :
            return self._parentMicroNN

        @property
        def Dimensions(self) :
            return self._dimensions

        @property
        def DimensionsCount(self) :
            return len(self._dimensions)

        @property
        def Shape(self) :
            return self._shape

        @property
        def Activation(self) :
            return self._activation

        @property
        def TopLayer(self) :
            return self._topLayer

        @property
        def Neurons(self) :
            return self._neurons

        @property
        def NeuronsCount(self) :
            return self._neuronsCount

        @property
        def InputConnectionsCount(self) :
            return self._inputConnCount

        @property
        def BottomLayer(self) :
            return self._bottomLayer

        @property
        def Bias(self) :
            return self._bias

    # -------------------------------------------------------------------------
    # --( Class : Layer )------------------------------------------------------
    # -------------------------------------------------------------------------

    class Layer(BaseLayer) :

        # -[ Constructor ]--------------------------------------

        def __init__( self,
                      parentMicroNN,
                      dimensions,
                      shape,
                      activation    = None,
                      initializer   = None,
                      connStruct    = None,
                      biasValue     = 1.0 ) :
            super().__init__( parentMicroNN = parentMicroNN,
                              dimensions    = dimensions,
                              shape         = shape,
                              activation    = activation,
                              initializer   = initializer,
                              biasValue     = biasValue )
            if self._topLayer is not None :
                if connStruct is None :
                    raise MicroNN.LayerException('"connStruct" must be defined for this layer type.')
                if not isinstance(connStruct, MicroNN.ConnStruct) :
                    raise MicroNN.LayerException('"connStruct" must be of ConnStruct type.')
            elif connStruct is not None :
                raise MicroNN.LayerException('"connStruct" must be "None" for this layer type.')
            self._inputConnCount = ( connStruct.ConnectLayer(self) if connStruct else 0 )

        # -[ Methods ]------------------------------------------

        def InitWeights(self) :
            if self._initializer :
                self._initializer.InitWeights(self)

        # ------------------------------------------------------

        def _recurComputeInput(self, neurons, dimIdx=0) :
            if dimIdx < self.DimensionsCount :
                for i in range(self._dimensions[dimIdx]) :
                    self._recurComputeInput(neurons[i], dimIdx+1)
            else :
                for i in range(self._shape.FlattenLen) :
                    neurons[i].ComputeInput()

        # ------------------------------------------------------

        def ComputeInput(self) :
            if isinstance(self, MicroNN.InputLayer) :
                raise MicroNN.LayerException('An input layer cannot computes it input.')
            self._recurComputeInput(self._neurons)
            if self._activation :
                self._activation.OnLayerInputComputed(self)

        # ------------------------------------------------------

        def _recurComputeOutput(self, neurons, dimIdx=0) :
            if dimIdx < self.DimensionsCount :
                for i in range(self._dimensions[dimIdx]) :
                    self._recurComputeOutput(neurons[i], dimIdx+1)
            else :
                for i in range(self._shape.FlattenLen) :
                    neurons[i].ComputeOutput()

        # ------------------------------------------------------

        def ComputeOutput(self) :
            if isinstance(self, MicroNN.InputLayer) :
                raise MicroNN.LayerException('An input layer cannot computes it output.')
            self._recurComputeOutput(self._neurons)

    # -------------------------------------------------------------------------
    # --( Class : InputLayer )-------------------------------------------------
    # -------------------------------------------------------------------------

    class InputLayer(Layer) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, parentMicroNN, dimensions, shape) :
            super().__init__(parentMicroNN, dimensions, shape)

        # -[ Methods ]------------------------------------------

        def _recurSetInputValues(self, neurons, values, dimIdx=0) :
            if dimIdx < self.DimensionsCount :
                if type(values) not in (list, tuple) or len(values) != self._dimensions[dimIdx] :
                    raise MicroNN.LayerException( 'Dimension %s of input values must be a list or a tuple of size %s.'
                                                  % (dimIdx+1, self._dimensions[dimIdx]) )
                for i in range(self._dimensions[dimIdx]) :
                    self._recurSetInputValues(neurons[i], values[i], dimIdx+1)
            else :
                scaled = (type(self._shape.ValueType) is not MicroNN.NeuronValueType)
                flattenValues = self._shape.Flatten(values)
                for i in range(self._shape.FlattenLen) :
                    if scaled :
                        flattenValues[i] = flattenValues[i] * 2 - 1
                    neurons[i].Output = flattenValues[i]

        # ------------------------------------------------------

        def SetInputValues(self, values) :
            self._recurSetInputValues(self._neurons, values)

    # -------------------------------------------------------------------------
    # --( Class : OutputLayer )------------------------------------------------
    # -------------------------------------------------------------------------

    class OutputLayer(Layer) :

        # -[ Methods ]------------------------------------------

        def _recurComputeTargetError(self, neurons, targetValues, dimIdx=0) :
            if dimIdx < self.DimensionsCount :
                if type(targetValues) not in (list, tuple) or len(targetValues) != self._dimensions[dimIdx] :
                    raise MicroNN.LayerException( 'Dimension %s of target values must be a list or a tuple of size %s.'
                                                  % (dimIdx+1, self._dimensions[dimIdx]) )
                for i in range(self._dimensions[dimIdx]) :
                    self._recurComputeTargetError(neurons[i], targetValues[i], dimIdx+1)
            else :
                scaled = (type(self._shape.ValueType) is not MicroNN.NeuronValueType)
                if scaled :
                    aMin, aMax = self._activation.GetRangeValues()
                flattenTargetValues = self._shape.Flatten(targetValues)
                for i in range(self._shape.FlattenLen) :
                    if scaled :
                        t = aMin + (flattenTargetValues[i] * (aMax-aMin))
                    else :
                        t = flattenTargetValues[i]
                    neurons[i].SetErrorFromTarget(t)

        # ------------------------------------------------------

        def ComputeTargetError(self, targetValues) :
            self._recurComputeTargetError(self._neurons, targetValues)

    # -------------------------------------------------------------------------
    # --( Class : Conv2DLayer )------------------------------------------------
    # -------------------------------------------------------------------------
    '''
    class Conv2DLayer(BaseLayer) :
 
        # -[ Constructor ]--------------------------------------
 
        def __init__( self,
                      parentMicroNN,
                      filtersCount,
                      overlappedShapesCount,
                      width,
                      height,
                      shape,
                      activation,
                      initializer,
                      biasValue = 1.0 ) :
            if not isinstance(filtersCount, int) or filtersCount <= 0 :
                raise MicroNN.LayerException('"filtersCount" must be of "int" type greater than zero.')
            if not isinstance(overlappedShapesCount, int) or overlappedShapesCount < 0 :
                raise MicroNN.LayerException('"overlappedShapesCount" must be of "int" type >= zero.')
            if type(width)  is not int or width  <= 0 or \
               type(height) is not int or height <= 0 :
                raise MicroNN.LayerException('"width" and "height" must be of "int" type greater than zero.')
            super().__init__( parentMicroNN = parentMicroNN,
                              dimensions    = [width, height, filtersCount],
                              shape         = shape,
                              activation    = activation,
                              initializer   = initializer,
                              biasValue     = biasValue )
            if isinstance(self._topLayer, MicroNN.Conv2DLayer) :
                self._inDepth = self._topLayer.Dimensions[2]
            elif self._topLayer.DimensionsCount == 2 :
                self._inDepth = None
            else :
                raise MicroNN.LayerException('2D convolution layer cannot be added after this top layer.')
            self._inWidth               = self._topLayer.Dimensions[0]
            self._inHeight              = self._topLayer.Dimensions[1]
            self._outWidth              = width
            self._outHeight             = height
            self._outDepth              = filtersCount
            self._out2DNeuronsCount     = self._outWidth  \
                                        * self._outHeight \
                                        * self._shape.FlattenLen
            self._overlappedShapesCount = overlappedShapesCount
            if self._outWidth > self._inWidth or self._outHeight > self._inHeight :
                raise MicroNN.LayerException('Layer 2D size (width x height) must be <= than top layer 2D size.')
            self._winCountX   = ceil(self._inWidth  / self._outWidth)
            self._winCountY   = ceil(self._inHeight / self._outHeight)
            self._winWidth    = self._winCountX + (2 * self._overlappedShapesCount)
            self._winHeight   = self._winCountY + (2 * self._overlappedShapesCount)
            self._kernels     = [ ]
            for i in range(filtersCount) :
                kernel                       = MicroNN()
                kernel.LearningRate          = parentMicroNN.LearningRate
                kernel.PlasticityStrengthing = parentMicroNN.PlasticityStrengthing
                kernel.AddInputLayer  ( dimensions  = [ self._winWidth,
                                                        self._winHeight,
                                                        (self._inDepth if self._inDepth else 1) ],
                                        shape       = self._topLayer.Shape )
                kernel.AddOutputLayer ( dimensions  = [1],
                                        shape       = shape,
                                        activation  = activation,
                                        initializer = initializer,
                                        connStruct  = MicroNN.FullyConnected,
                                        biasValue   = None )
                self._kernels.append(kernel)
 
        # -[ Methods ]------------------------------------------

        def InitWeights(self) :
            for kernel in self._kernels :
                kernel.InitWeights()

        # ------------------------------------------------------

        def ComputeInput(self) :
            for kernelIdx in range(len(self._kernels)) :
                kernel         = self._kernels[kernelIdx]
                kernelInNrn    = kernel.GetInputLayer().Neurons
                kernelOutLayer = kernel.GetOutputLayer()
                kernelOutNrn   = kernelOutLayer.Neurons[0]
                for outX in range(self._outWidth) :
                    winStartX = (outX * self._winCountX) - self._overlappedShapesCount
                    for outY in range(self._outHeight) :
                        winStartY = (outY * self._winCountY) - self._overlappedShapesCount
                        outputNrn = self._neurons[outX][outY][kernelIdx]
                        for winX in range(self._winWidth) :
                            inX = winStartX + winX
                            for winY in range(self._winHeight) :
                                inY           = winStartY + winY
                                kernelInNrnXY = kernelInNrn[winX][winY]
                                zeroPadding   = ( inX < 0 or inX >= self._inWidth or \
                                                  inY < 0 or inY >= self._inHeight )
                                if not zeroPadding :
                                    topLayerNrnXY = self._topLayer.Neurons[inX][inY]
                                if self._inDepth :
                                    for depth in range(self._inDepth) :
                                        for i in range(self._topLayer.Shape.FlattenLen) :
                                            if not zeroPadding :
                                                val = topLayerNrnXY[depth][i].Output
                                            else :
                                                val = 0.0
                                            kernelInNrnXY[depth][i].Output      = val
                                            kernelInNrnXY[depth][i].TotalOutput = 0.0
                                else :
                                    for i in range(self._topLayer.Shape.FlattenLen) :
                                        if not zeroPadding :
                                            val = topLayerNrnXY[i].Output
                                        else :
                                            val = 0.0
                                        kernelInNrnXY[0][i].Output      = val
                                        kernelInNrnXY[0][i].TotalOutput = 0.0
                        kernelOutLayer.ComputeInput()
                        for i in range(self._shape.FlattenLen) :
                            outputNrn[i].ComputeInput()
                            outputNrn[i].Input += kernelOutNrn[i].Input
                            kernelOutNrn[i].TotalSigErr = 0.0

        # ------------------------------------------------------

        def ComputeOutput(self) :
            for outX in range(self._outWidth) :
                for outY in range(self._outHeight) :
                    for depth in range(self._outDepth) :
                        for i in range(self._shape.FlattenLen) :
                            self._neurons[outX][outY][depth][i].ComputeOutput()

        # ------------------------------------------------------

        def BackPropagateError(self) :
            super().BackPropagateError()
            for kernelIdx in range(len(self._kernels)) :
                kernel         = self._kernels[kernelIdx]
                kernelInLayer  = kernel.GetInputLayer()
                kernelInNrn    = kernelInLayer.Neurons
                kernelOutLayer = kernel.GetOutputLayer()
                kernelOutNrn   = kernelOutLayer.Neurons[0]
                for outX in range(self._outWidth) :
                    winStartX = (outX * self._winCountX) - self._overlappedShapesCount
                    for outY in range(self._outHeight) :
                        winStartY = (outY * self._winCountY) - self._overlappedShapesCount
                        outputNrn = self._neurons[outX][outY][kernelIdx]
                        for i in range(self._shape.FlattenLen) :
                            kernelOutNrn[i].BackPropagateError()
                        for winX in range(self._winWidth) :
                            inX = winStartX + winX
                            for winY in range(self._winHeight) :
                                inY           = winStartY + winY
                                kernelInNrnXY = kernelInNrn[winX][winY]
                                zeroPadding   = ( inX < 0 or inX >= self._inWidth or \
                                                  inY < 0 or inY >= self._inHeight )
                                if not zeroPadding :
                                    topLayerNrnXY = self._topLayer.Neurons[inX][inY]
                                    if self._inDepth :
                                        for depth in range(self._inDepth) :
                                            for i in range(self._topLayer.Shape.FlattenLen) :
                                                topLayerNrnXY[depth][i].Error       += kernelInNrnXY[depth][i].Error
                                                kernelInNrnXY[depth][i].TotalOutput += topLayerNrnXY[depth][i].Output
                                    else :
                                        for i in range(self._topLayer.Shape.FlattenLen) :
                                            topLayerNrnXY[i].Error          += kernelInNrnXY[0][i].Error
                                            kernelInNrnXY[0][i].TotalOutput += topLayerNrnXY[i].Output
    '''
    # -------------------------------------------------------------------------
    # --( Class : ConnStructException )----------------------------------------
    # -------------------------------------------------------------------------

    class ConnStructException(Exception) :
        pass

    # -------------------------------------------------------------------------
    # --( Abstract class : ConnStruct )----------------------------------------
    # -------------------------------------------------------------------------

    class ConnStruct :

        # -[ Constructor ]--------------------------------------

        def __init__(self) :
            if type(self) is MicroNN.ConnStruct :
                raise MicroNN.ConnStructException('"ConnStruct" is an abstract class and cannot be instancied.')

        # -[ Methods ]------------------------------------------

        def ConnectLayer(self, layer) :
            raise MicroNN.ConnStructException('"ConnectLayer" method must be implemented.')

    # -------------------------------------------------------------------------
    # --( Class : FullyConnStruct )--------------------------------------------
    # -------------------------------------------------------------------------

    class FullyConnStruct(ConnStruct) :

        # -[ Methods ]------------------------------------------

        def _recurConnectTo(self, dstLayer, srcNeurons, dstNeurons, dimIdx=0) :
            connCount = 0
            if dimIdx < dstLayer.DimensionsCount :
                for i in range(dstLayer.Dimensions[dimIdx]) :
                    connCount += self._recurConnectTo(dstLayer, srcNeurons, dstNeurons[i], dimIdx+1)
            else :
                for src in srcNeurons :
                    for dst in dstNeurons :
                        MicroNN.Connection(src, dst)
                        connCount += 1
            return connCount

        # ------------------------------------------------------

        def _recurConnectFrom(self, dstLayer, srcNeurons, dimIdx=0) :
            connCount = 0
            if dimIdx < dstLayer.TopLayer.DimensionsCount :
                for i in range(dstLayer.TopLayer.Dimensions[dimIdx]) :
                    connCount += self._recurConnectFrom(dstLayer, srcNeurons[i], dimIdx+1)
            else :
                connCount += self._recurConnectTo(dstLayer, srcNeurons, dstLayer.Neurons)
            return connCount

        # ------------------------------------------------------

        def ConnectLayer(self, layer) :
            return self._recurConnectFrom(layer, layer.TopLayer.Neurons)

    # -------------------------------------------------------------------------
    # --( Class : LocallyConnStruct )------------------------------------------
    # -------------------------------------------------------------------------

    class LocallyConnStruct(ConnStruct) :

        RoleIncrease = 0x0A
        RoleDecrease = 0x0B

        # -[ Constructor ]--------------------------------------

        def __init__(self, role, overlappedShapesCount=0) :
            super().__init__()
            if role not in (self.RoleIncrease, self.RoleDecrease) :
                raise MicroNN.ConnStructException('LocallyConnStruct : "role" is not a correct value.')
            if not isinstance(overlappedShapesCount, int) or overlappedShapesCount < 0 :
                raise MicroNN.ConnStructException('LocallyConnStruct : "overlappedShapesCount" must be of "int" type >= zero.')
            self._role                  = role
            self._overlappedShapesCount = overlappedShapesCount

        # -[ Methods ]------------------------------------------

        def _recurConnect(self, highLayer, lowLayer, dimsInfo, highNeurons, lowNeurons, dimIdx=0) :
            connCount = 0
            if dimIdx < lowLayer.DimensionsCount :
                for lowIdx in range(lowLayer.Dimensions[dimIdx]) :
                    di       = dimsInfo[dimIdx]
                    winStart = (lowIdx * di.winCount) - self._overlappedShapesCount
                    for x in range(di.winSize) :
                        highIdx = winStart + x
                        if highIdx >= 0 and highIdx < highLayer.Dimensions[dimIdx] :
                            connCount += self._recurConnect( highLayer,
                                                             lowLayer,
                                                             dimsInfo,
                                                             highNeurons[highIdx],
                                                             lowNeurons[lowIdx],
                                                             dimIdx+1 )
            else :
                if self._role == self.RoleIncrease :
                    srcNeurons = lowNeurons
                    dstNeurons = highNeurons
                else :
                    srcNeurons = highNeurons
                    dstNeurons = lowNeurons
                for src in srcNeurons :
                    for dst in dstNeurons :
                        MicroNN.Connection(src, dst)
                        connCount += 1
            return connCount

        # ------------------------------------------------------

        def ConnectLayer(self, layer) :
            if layer.DimensionsCount != layer.TopLayer.DimensionsCount :
                raise MicroNN.ConnStructException('LocallyConnStruct : Both layers must have the same number of dimensions.')
            if self._role == self.RoleIncrease :
                lowLayer  = layer.TopLayer
                highLayer = layer
            else :
                lowLayer  = layer
                highLayer = layer.TopLayer
            class dimInfo :
                pass
            dimsInfo = [ ]
            for i in range(layer.DimensionsCount) :
                if lowLayer.Dimensions[i] > highLayer.Dimensions[i] :
                    raise MicroNN.ConnStructException('LocallyConnStruct : The role is not suitable to the layers dimension(s).')
                di          = dimInfo()
                di.winCount = ceil(highLayer.Dimensions[i] / lowLayer.Dimensions[i])
                di.winSize  = di.winCount + (2 * self._overlappedShapesCount)
                dimsInfo.append(di)
            return self._recurConnect(highLayer, lowLayer, dimsInfo, highLayer.Neurons, lowLayer.Neurons)

    # -------------------------------------------------------------------------
    # --( Class : DataObjectConnStruct )---------------------------------------
    # -------------------------------------------------------------------------

    class DataObjectConnStruct(ConnStruct) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, o) :
            super().__init__()
            if not isinstance(o, list) :
                raise MicroNN.ConnStructException('DataObjectConnStruct : Data object is not valid.')
            self._o = o

        # -[ Methods ]------------------------------------------

        def ConnectLayer(self, layer) :
            try :
                connCount = 0
                topLayer  = layer.TopLayer
                for dstIdx in range(len(self._o)) :
                    neuronDst = layer.GetNeuronByIndex(dstIdx)
                    for conn in self._o[dstIdx] :
                        srcIdx = conn[0]
                        weight = conn[1]
                        if srcIdx >= 0 :
                            neuronSrc  = topLayer.GetNeuronByIndex(srcIdx)
                            connCount += 1
                        else :
                            neuronSrc  = layer.Bias
                        MicroNN.Connection(neuronSrc, neuronDst, weight)
                return connCount
            except :
                raise MicroNN.ConnStructException('DataObjectConnStruct : Data object is not valid.')

    # -------------------------------------------------------------------------
    # --( Class : ActivationException )----------------------------------------
    # -------------------------------------------------------------------------

    class ActivationException(Exception) :
        pass

    # -------------------------------------------------------------------------
    # --( Abstract class : Activation )----------------------------------------
    # -------------------------------------------------------------------------

    class Activation :

        # -[ Constructor ]--------------------------------------

        def __init__(self) :
            if type(self) is MicroNN.Activation :
                raise MicroNN.ActivationException('"Activation" is an abstract class and cannot be instancied.')

        # -[ Methods ]------------------------------------------

        def OnLayerInputComputed(self, layer) :
            pass

        # ------------------------------------------------------

        def Get(self, neuron) :
            raise MicroNN.ActivationException('"Get" method must be implemented.')

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            raise MicroNN.ActivationException('"GetDerivative" method must be implemented.')

        # ------------------------------------------------------

        def GetRangeValues(self) :
            raise MicroNN.ActivationException('"GetRangeValues" method must be implemented.')

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            raise MicroNN.ActivationException('"GetAsDataObject" method must be implemented.')

        # ------------------------------------------------------

        @staticmethod
        def CreateFromDataObject(o) :
            try :
                name = o['Name']
                if   name == 'Identity' :
                    return MicroNN.IdentityActivation()
                elif name == 'Heaviside' :
                    return MicroNN.HeavisideActivation()
                elif name == 'Sigmoid' :
                    return MicroNN.SigmoidActivation()
                elif name == 'TanH' :
                    return MicroNN.TanHActivation()
                elif name == 'ReLU' :
                    return MicroNN.ReLUActivation()
                elif name == 'PReLU' :
                    return MicroNN.PReLUActivation(o['Alpha'])
                elif name == 'LeakyReLU' :
                    return MicroNN.LeakyReLUActivation()
                elif name == 'SoftPlus' :
                    return MicroNN.SoftPlusActivation()
                elif name == 'Sinusoid' :
                    return MicroNN.SinusoidActivation()
                elif name == 'Gaussian' :
                    return MicroNN.GaussianActivation()
                elif name == 'SoftMax' :
                    return MicroNN.SoftMaxActivation()
                else :
                    raise Exception()
            except :
                raise MicroNN.ActivationException('Data object is not valid.')

    # -------------------------------------------------------------------------
    # --( Class : IdentityActivation )-----------------------------------------
    # -------------------------------------------------------------------------

    class IdentityActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return neuron.Input

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            return 1.0

        # ------------------------------------------------------

        def GetRangeValues(self) :
            return (-inf, inf)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'Identity' }

    # -------------------------------------------------------------------------
    # --( Class : HeavisideActivation )----------------------------------------
    # -------------------------------------------------------------------------

    class HeavisideActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return 0.0 if neuron.Input < 0 else 1.0

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            return 1.0

        # ------------------------------------------------------

        def GetRangeValues(self) :
            return (0.0, 1.0)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'Heaviside' }

    # -------------------------------------------------------------------------
    # --( Class : SigmoidActivation )------------------------------------------
    # -------------------------------------------------------------------------

    class SigmoidActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return 1.0 / ( 1.0 + exp(-neuron.Input) )

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            f = neuron.Output
            return f * (1.0-f)

        # ------------------------------------------------------

        def GetRangeValues(self) :
            return (0.0, 1.0)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'Sigmoid' }

    # -------------------------------------------------------------------------
    # --( Class : TanHActivation )---------------------------------------------
    # -------------------------------------------------------------------------

    class TanHActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return ( 2.0 / (1.0 + exp(-2.0 * neuron.Input)) ) - 1.0

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            f = neuron.Output
            return 1.0 - f**2

        # ------------------------------------------------------

        def GetRangeValues(self) :
            return (-1.0, 1.0)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'TanH' }

    # -------------------------------------------------------------------------
    # --( Class : ReLUActivation )---------------------------------------------
    # -------------------------------------------------------------------------

    class ReLUActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return max(0.0, neuron.Input)

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            return 0.0 if neuron.Input < 0.0 else 1.0

        # ------------------------------------------------------

        def GetRangeValues(self) :
            return (0.0, inf)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'ReLU' }

    # -------------------------------------------------------------------------
    # --( Class : PReLUActivation )--------------------------------------------
    # -------------------------------------------------------------------------

    class PReLUActivation(Activation) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, alpha) :
            if type(alpha) not in (float, int) :
                raise MicroNN.ActivationException('"alpha" must be of "float" or "int" type.')
            super().__init__()
            self._alpha = float(alpha)

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return (self._alpha * neuron.Input) if neuron.Input < 0.0 else neuron.Input

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            return self._alpha if neuron.Input < 0.0 else 1.0

        # ------------------------------------------------------

        def GetRangeValues(self) :
            return (-inf, inf)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return {
                'Name'  : 'PReLU',
                'Alpha' : self._alpha
            }

    # -------------------------------------------------------------------------
    # --( Class : LeakyReLUActivation )----------------------------------------
    # -------------------------------------------------------------------------

    class LeakyReLUActivation(PReLUActivation) :

        # -[ Constructor ]--------------------------------------

        def __init__(self) :
            super().__init__(alpha=0.01)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'LeakyReLU' }

    # -------------------------------------------------------------------------
    # --( Class : SoftPlusActivation )-----------------------------------------
    # -------------------------------------------------------------------------

    class SoftPlusActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return log(1 + exp(neuron.Input))

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            return 1 / (1 + exp(-neuron.Input))

        # ------------------------------------------------------

        def GetRangeValues(self) :
            return (0.0, inf)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'SoftPlus' }

    # -------------------------------------------------------------------------
    # --( Class : SinusoidActivation )-----------------------------------------
    # -------------------------------------------------------------------------

    class SinusoidActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return sin(neuron.Input)

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            return cos(neuron.Input)

        # ------------------------------------------------------

        def GetRangeValues(self) :
            return (-1.0, 1.0)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'Sinusoid' }

    # -------------------------------------------------------------------------
    # --( Class : GaussianActivation )-----------------------------------------
    # -------------------------------------------------------------------------

    class GaussianActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return exp(-neuron.Input**2)

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            f = neuron.Output
            return -2 * neuron.Input * f

        # ------------------------------------------------------

        def GetRangeValues(self) :
            return (0.0, 1.0)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'Gaussian' }

    # -------------------------------------------------------------------------
    # --( Class : SoftMaxActivation )------------------------------------------
    # -------------------------------------------------------------------------

    class SoftMaxActivation(Activation) :

        # -[ Constructor ]--------------------------------------

        def __init__(self) :
            super().__init__()
            self._layerNrnList    = [ ]
            self._layerNrnOutExps = [ ]
            self._expsSum         = 0.0

        # -[ Methods ]------------------------------------------

        def OnLayerInputComputed(self, layer) :
            if not self._layerNrnList :
                self._layerNrnList    = layer.GetNeuronsList()
                self._layerNrnOutExps = [0] * len(self._layerNrnList)
            self._expsSum = 0.0
            for i in range(len(self._layerNrnList)) :
                x = exp(self._layerNrnList[i].Input)
                self._layerNrnOutExps[i]  = x
                self._expsSum            += x

        # ------------------------------------------------------

        def Get(self, neuron) :
            return self._layerNrnOutExps[neuron.Index] / self._expsSum

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            f = neuron.Output
            return f * (1.0-f)

        # ------------------------------------------------------

        def GetRangeValues(self) :
            return (0.0, 1.0)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'SoftMax' }

    # -------------------------------------------------------------------------
    # --( Class : InitializerException )---------------------------------------
    # -------------------------------------------------------------------------

    class InitializerException(Exception) :
        pass

    # -------------------------------------------------------------------------
    # --( Abstract class : Initializer )---------------------------------------
    # -------------------------------------------------------------------------

    class Initializer :

        HeUniform      = 0x0A
        HeNormal       = 0x0B
        XavierUniform  = 0x0C
        XavierNormal   = 0x0D

        # -[ Constructor ]--------------------------------------

        def __init__(self, initialization) :
            if type(self) is MicroNN.Initializer :
                raise MicroNN.InitializerException('"Initializer" is an abstract class and cannot be instancied.')
            if initialization not in ( self.HeUniform,
                                       self.HeNormal,
                                       self.XavierUniform,
                                       self.XavierNormal ) :
                raise MicroNN.InitializerException('"initialization" is not correct.')
            self._initialization = initialization

        # -[ Methods ]------------------------------------------

        @staticmethod
        def _uniformDistrib(count, limit) :
            delta = 2 * limit
            return [ (random.random() * delta) - limit
                     for i in range(count) ]

        # ------------------------------------------------------

        @staticmethod
        def _normalDistrib(count, mean, deviation) :
            values = MicroNN.Initializer._uniformDistrib(count, 1.0)
            a      = 1 / (deviation * sqrt(2*pi))
            b      = (2 * deviation) ** 2
            for i in range(len(values)) :
                values[i] = a * exp( -( (values[i]-mean) ** 2 / b ) )
            return values

        # ------------------------------------------------------

        def _applyDistribToWeights(self, layer, factor) :
            if not layer.TopLayer :
                raise MicroNN.InitializerException('No top layer is present to initialize weights.')
            n     = layer.TopLayer.NeuronsCount
            count = layer.InputConnectionsCount
            if layer.Bias :
                n     += 1
                count += layer.Bias.OutputConnectionsCount
            if self._initialization in (self.XavierUniform, self.XavierNormal) :
                n += layer.NeuronsCount
            if self._initialization in (self.HeUniform, self.XavierUniform) :
                limit   = factor * sqrt(6/n)
                distrib = MicroNN.Initializer._uniformDistrib(count, limit)
            else :
                deviation = factor * sqrt(2/n)
                distrib   = MicroNN.Initializer._normalDistrib(count, 0.0, deviation)
            i = 0
            for n in layer.GetNeuronsList() :
                for c in n.InputConnections :
                    c.Weight = distrib[i]
                    i += 1

        # ------------------------------------------------------

        def InitWeights(self, layer) :
            raise MicroNN.InitializerException('"InitWeights" method must be implemented.')

        # -[ Properties ]---------------------------------------

        @property
        def Initialization(self) :
            return self._initialization

    # -------------------------------------------------------------------------
    # --( Class : LogisticInitializer )----------------------------------------
    # -------------------------------------------------------------------------

    class LogisticInitializer(Initializer) :

        # -[ Methods ]------------------------------------------

        def InitWeights(self, layer) :
            self._applyDistribToWeights(layer, factor=1)

    # -------------------------------------------------------------------------
    # --( Class : TanHInitializer )--------------------------------------------
    # -------------------------------------------------------------------------

    class TanHInitializer(Initializer) :

        # -[ Methods ]------------------------------------------

        def InitWeights(self, layer) :
            self._applyDistribToWeights(layer, factor=4)

    # -------------------------------------------------------------------------
    # --( Class : ReLUInitializer )--------------------------------------------
    # -------------------------------------------------------------------------

    class ReLUInitializer(Initializer) :

        # -[ Methods ]------------------------------------------

        def InitWeights(self, layer) :
            self._applyDistribToWeights(layer, factor=sqrt(2))

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    # -[ Constructor ]--------------------------------------

    def __init__(self) :
        self._learningRate          = MicroNN.DEFAULT_LEARNING_RATE
        self._plasticityStrengthing = MicroNN.DEFAULT_PLASTICITY_STRENGTHING
        self._layers                = [ ]
        self._examples              = [ ]

    # -[ Methods ]------------------------------------------

    @staticmethod
    def Init1D(shapesCount) :
        if type(shapesCount) is not int or shapesCount <= 0 :
            raise MicroNNException('"shapesCount" must be of "int" type greater than zero.')
        return [shapesCount]

    # ------------------------------------------------------

    @staticmethod
    def Init2D(xShapesCount, yShapesCount) :
        if type(xShapesCount) is not int or xShapesCount <= 0 or \
           type(yShapesCount) is not int or yShapesCount <= 0 :
            raise MicroNNException('"xShapesCount" and "yShapesCount" must be of "int" type greater than zero.')
        return [xShapesCount, yShapesCount]

    # ------------------------------------------------------

    @staticmethod
    def Init3D(xShapesCount, yShapesCount, zShapesCount) :
        if type(xShapesCount) is not int or xShapesCount <= 0 or \
           type(yShapesCount) is not int or yShapesCount <= 0 or \
           type(zShapesCount) is not int or zShapesCount <= 0 :
            raise MicroNNException('"xShapesCount", "yShapesCount" and "zShapesCount" must be of "int" type greater than zero.')
        return [xShapesCount, yShapesCount, zShapesCount]

    # ------------------------------------------------------

    def AddLayer( self,
                  dimensions,
                  shape,
                  activation  = None,
                  initializer = None,
                  connStruct  = None,
                  biasValue   = 1.0 ) :
        return MicroNN.Layer( parentMicroNN = self,
                              dimensions    = dimensions,
                              shape         = shape,
                              activation    = activation,
                              initializer   = initializer,
                              connStruct    = connStruct,
                              biasValue     = biasValue )

    # ------------------------------------------------------

    def AddInputLayer(self, dimensions, shape) :
        return MicroNN.InputLayer( parentMicroNN = self,
                                   dimensions    = dimensions,
                                   shape         = shape )

    # ------------------------------------------------------

    def AddOutputLayer( self,
                        dimensions,
                        shape,
                        activation  = None,
                        initializer = None,
                        connStruct  = None,
                        biasValue   = 1.0 ) :
        return MicroNN.OutputLayer( parentMicroNN = self,
                                    dimensions    = dimensions,
                                    shape         = shape,
                                    activation    = activation,
                                    initializer   = initializer,
                                    connStruct    = connStruct,
                                    biasValue     = biasValue )

    # ------------------------------------------------------

    def AddQLearningOutputLayer(self, actionsCount) :
        if not isinstance(actionsCount, int) or actionsCount <= 1 :
            raise MicroNNException('"actionsCount" must be of "int" type greater than 1.')
        initializer = MicroNN.LogisticInitializer(uniform=True, xavier=False)
        return MicroNN.OutputLayer( parentMicroNN = self,
                                    dimensions    = MicroNN.Init1D(actionsCount),
                                    shape         = MicroNN.ValueShape(),
                                    activation    = MicroNN.SoftMaxActivation(),
                                    initializer   = initializer,
                                    connStruct    = MicroNN.FullyConnStruct() )

    # ------------------------------------------------------

    def GetInputLayer(self) :
        if len(self._layers) > 0 :
            inputLayer = self._layers[0]
            if isinstance(inputLayer, MicroNN.InputLayer) :
                return inputLayer
        return None

    # ------------------------------------------------------

    def GetOutputLayer(self) :
        if len(self._layers) > 0 :
            outputLayer = self._layers[len(self._layers)-1]
            if isinstance(outputLayer, MicroNN.OutputLayer) :
                return outputLayer
        return None

    # ------------------------------------------------------

    def InitWeights(self) :
        self._ensureNetworkIsComplete()
        for layer in self._layers :
            layer.InitWeights()

    # ------------------------------------------------------

    def Test(self, inputValues, targetValues) :
        if not inputValues or not targetValues :
            raise MicroNNException('"inputValues" and "targetValues" must be defined.')
        self._simulate(inputValues, targetValues)
        return self.SuccessPercent

    # ------------------------------------------------------

    def Predict(self, inputValues) :
        if not inputValues :
            raise MicroNNException('"inputValues" must be defined.')
        self._simulate(inputValues)
        return self.GetOutputLayer().GetOutputValues()

    # ------------------------------------------------------

    def QLearningLearnChosenAction( self,
                                    stateInputValues,
                                    pastStateInputValues,
                                    chosenActionIndex,
                                    rewardValue,
                                    terminalState         = True,
                                    discountFactor        = None ) :
        self._ensureQLearningNetworkIsValid()
        outputLayer = self.GetOutputLayer()
        if not isinstance(chosenActionIndex, int) or \
           chosenActionIndex < 0 or chosenActionIndex >= outputLayer.NeuronsCount :
            raise MicroNNException( '"chosenActionIndex" must be of "int" type >= 0 and < %s.' \
                                    % outputLayer.NeuronsCount )
        if type(rewardValue) not in (float, int) :
            raise MicroNNException('"rewardValue" must be of "float" or "int" type.')
        if not terminalState :
            if type(discountFactor) not in (float, int) or \
               discountFactor < 0 or discountFactor > 1 :
                raise MicroNNException('"discountFactor" must be of "float" or "int" type >= 0 and <= 1.')
            bestActVal = None
            for actVal in self.Predict(stateInputValues) :
                if bestActVal is None or actVal > bestActVal :
                    bestActVal = actVal
            rewardValue += float(discountFactor) * bestActVal
        targetValues = self.Predict(pastStateInputValues)
        targetValues[chosenActionIndex] = rewardValue
        self.Learn(pastStateInputValues, targetValues)

    # ------------------------------------------------------

    def QLearningPredictBestAction(self, stateInputValues) :
        self._ensureQLearningNetworkIsValid()
        bestActIdx = None
        bestActVal = None
        idx        = 0
        for actVal in self.Predict(stateInputValues) :
            if bestActVal is None or actVal > bestActVal :
                bestActVal = actVal
                bestActIdx = idx
            idx += 1
        return bestActIdx

    # ------------------------------------------------------

    def AddExample(self, inputValues, targetValues) :
        if not inputValues or not targetValues :
            raise MicroNNException('"inputValues" and "targetValues" must be defined.')
        self._examples.append( (inputValues, targetValues) )

    # ------------------------------------------------------

    def ClearExamples(self) :
        self._examples.clear()

    # ------------------------------------------------------

    def LearnExamples( self,
                       minibatchSize = None,
                       maxEpochs     = None,
                       maxSeconds    = None,
                       verbose       = True ) :
        self._ensureNetworkIsComplete()
        examplesCount = len(self._examples)
        if examplesCount == 0 :
            raise MicroNNException('No examples found.')
        if minibatchSize is None :
            minibatchSize = examplesCount
        elif not isinstance(minibatchSize, int) or minibatchSize <= 0 :
            raise MicroNNException('"minibatchSize" must be of "int" type greater than zero.')
        if maxEpochs is not None :
            if not isinstance(maxEpochs, int) or maxEpochs <= 0 :
                raise MicroNNException('"maxEpochs" must be of "int" type greater than zero.')
        if maxSeconds is not None :
            if not isinstance(maxSeconds, int) or maxSeconds <= 0 :
                raise MicroNNException('"maxSeconds" must be of "int" type greater than zero.')
        epochCount = 0
        endTime    = (time() + maxSeconds) if maxSeconds else None
        while ( endTime   is None or time()     < endTime   ) and \
              ( maxEpochs is None or epochCount < maxEpochs ) :
            random.shuffle(self._examples)
            for i in range(0, examplesCount, minibatchSize) :
                for ex in self._examples[i:i+minibatchSize] :
                    self._simulate(ex[0], ex[1])
                    self._backPropagateError()
                self._updateWeights(batchSize=minibatchSize)
            epochCount += 1
            successAvg = 0.0
            for ex in self._examples :
                successAvg += self.Test(ex[0], ex[1])
            successAvg /= examplesCount
            if verbose :
                print( "MicroNN [ EPOCH %s -> SUCCESS %s%% ]"
                       % ( epochCount, round(successAvg*1000)/1000 ) )
            if successAvg == 100 :
                break

    # ------------------------------------------------------

    def GetAsDataObject(self) :
        self._ensureNetworkIsComplete()
        try :
            return {
                'MicroNNVersion'        : MicroNN.VERSION,
                'LearningRate'          : self._learningRate,
                'PlasticityStrengthing' : self._plasticityStrengthing,
                'LayersCount'           : len(self._layers),
                'Layers'                : [ l.GetAsDataObject()
                                            for l in self._layers ]
            }
        except :
            raise MicroNNException('Error to get neural network as data object.')

    # ------------------------------------------------------

    @staticmethod
    def CreateFromDataObject(o) :
        if not isinstance(o, dict) or not 'MicroNNVersion' in o :
            raise MicroNNException('Data object is not valid.')
        oVer = o['MicroNNVersion']
        if oVer != MicroNN.VERSION :
            raise MicroNNException( 'MicroNN version of data object (%s) is not valid for this version (%s).'
                                    % (oVer, MicroNN.VERSION) )
        try :
            microNN                       = MicroNN()
            microNN.LearningRate          = o['LearningRate']
            microNN.PlasticityStrengthing = o['PlasticityStrengthing']
            for oLayer in o['Layers'] :
                MicroNN.BaseLayer.CreateFromDataObject(microNN, oLayer)
            return microNN
        except :
            raise MicroNNException('Data object is not valid.')

    # ------------------------------------------------------

    def SaveToJSONFile(self, filename) :
        o = self.GetAsDataObject()
        try :
            file = open(filename, 'wt')
            file.write(json.dumps(o))
            file.close()
        except :
            raise MicroNNException('Error to save JSON file "%s".' % filename)

    # ------------------------------------------------------

    @staticmethod
    def LoadFromJSONFile(filename) :
        try :
            with open(filename, 'r') as file :
                o = json.load(file)
        except :
            raise MicroNNException('Error to load JSON file "%s".' % filename)
        return MicroNN.CreateFromDataObject(o)

    # ------------------------------------------------------

    def _ensureNetworkIsComplete(self) :
        if not self.IsNetworkComplete :
            raise MicroNNException('Neural network must have an input and an output layers.')

    # ------------------------------------------------------

    def _ensureQLearningNetworkIsValid(self) :
        self._ensureNetworkIsComplete()
        outputLayer = self.GetOutputLayer()
        if outputLayer.DimensionsCount > 1 or \
           not isinstance(outputLayer.Shape, MicroNN.ValueShape) or \
           not isinstance(outputLayer.Shape.ValueType, MicroNN.NeuronValueType) :
            raise MicroNNException('Output layer of neural network is not valid for QLearning usage.')

    # ------------------------------------------------------

    def _propagate(self) :
        self._ensureNetworkIsComplete()
        for layer in self._layers :
            if not isinstance(layer, MicroNN.InputLayer) :
                layer.ComputeInput()
                layer.ComputeOutput()

    # ------------------------------------------------------

    def _backPropagateError(self) :
        self._ensureNetworkIsComplete()
        i = len(self._layers)-1
        while i > 0 :
            self._layers[i].BackPropagateError()
            i -= 1

    # ------------------------------------------------------

    def _updateWeights(self, batchSize) :
        self._ensureNetworkIsComplete()
        i = len(self._layers)-1
        while i > 0 :
            self._layers[i].UpdateConnectionsWeight(batchSize)
            i -= 1

    # ------------------------------------------------------

    def _simulate(self, inputValues, targetValues=None) :
        self._ensureNetworkIsComplete()
        self.GetInputLayer().SetInputValues(inputValues)
        self._propagate()
        if targetValues :
            self.GetOutputLayer().ComputeTargetError(targetValues)

    # -[ Properties ]---------------------------------------

    @property
    def LearningRate(self) :
        return self._learningRate
    @LearningRate.setter
    def LearningRate(self, value) :
        if type(value) not in (float, int) :
            raise MicroNNException('"value" must be of "float" or "int" type.')
        self._learningRate = float(value)

    @property
    def PlasticityStrengthing(self) :
        return self._plasticityStrengthing
    @PlasticityStrengthing.setter
    def PlasticityStrengthing(self, value) :
        if type(value) not in (float, int) :
            raise MicroNNException('"value" must be of "float" or "int" type.')
        self._plasticityStrengthing = float(value)

    @property
    def Layers(self) :
        return self._layers

    @property
    def LayersCount(self) :
        return len(self._layers)

    @property
    def NeuronsCount(self) :
        count = 0
        for layer in self._layers :
            count += layer.NeuronsCount
        return count

    @property
    def ConnectionsCount(self) :
        count = 0
        i     = 1
        while i < len(self._layers) :
            count += self._layers[i].InputConnectionsCount
            i     += 1
        return count

    @property
    def IsNetworkComplete(self) :
        return ( self.GetInputLayer()  is not None and \
                 self.GetOutputLayer() is not None )

    @property
    def MSE(self) :
        self._ensureNetworkIsComplete()
        return self.GetOutputLayer().GetMeanSquareError()

    @property
    def MAE(self) :
        self._ensureNetworkIsComplete()
        return self.GetOutputLayer().GetMeanAbsoluteError()

    @property
    def MSEPercent(self) :
        self._ensureNetworkIsComplete()
        return self.GetOutputLayer().GetMeanSquareErrorAsPercent()

    @property
    def MAEPercent(self) :
        self._ensureNetworkIsComplete()
        return self.GetOutputLayer().GetMeanAbsoluteErrorAsPercent()

    @property
    def SuccessPercent(self) :
        return 100 - self.MAEPercent

    @property
    def ExamplesCount(self) :
        return len(self._examples)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

MicroNN.Shape.Neuron         = MicroNN.ValueShape()
MicroNN.Shape.Bool           = MicroNN.ValueShape(MicroNN.BoolValueType())
MicroNN.Shape.Byte           = MicroNN.ValueShape(MicroNN.ByteValueType())
MicroNN.Shape.Percent        = MicroNN.ValueShape(MicroNN.PercentValueType())
MicroNN.Shape.Color          = MicroNN.ColorShape()

MicroNN.Activation.Identity  = MicroNN.IdentityActivation()
MicroNN.Activation.Heaviside = MicroNN.HeavisideActivation()
MicroNN.Activation.Sigmoid   = MicroNN.SigmoidActivation()
MicroNN.Activation.TanH      = MicroNN.TanHActivation()
MicroNN.Activation.ReLU      = MicroNN.ReLUActivation()
MicroNN.Activation.LeakyReLU = MicroNN.LeakyReLUActivation()
MicroNN.Activation.SoftPlus  = MicroNN.SoftPlusActivation()
MicroNN.Activation.Sinusoid  = MicroNN.SinusoidActivation()
MicroNN.Activation.Gaussian  = MicroNN.GaussianActivation()

MicroNN.FullyConnected       = MicroNN.FullyConnStruct()
