"""
The MIT License (MIT)
Copyright © 2019 Jean-Christophe Bos & HC² (www.hc2.fr)
"""


from   math import ceil, exp, log, sin, cos
from   time import time
import json

try :
    from random  import random
except :
    from machine import rng

# -------------------------------------------------------------------------
# --( Class : MicroNNException )-------------------------------------------
# -------------------------------------------------------------------------

class MicroNNException(Exception) :
    pass

# -------------------------------------------------------------------------
# --( Class : MicroNN )----------------------------------------------------
# -------------------------------------------------------------------------

class MicroNN :

    VERSION                             = '1.0.0'

    DEFAULT_ERROR_CORRECTION_WEIGHTING  = 0.30
    DEFAULT_CONN_PLASTICITY_STRENGTHING = 0.75

    SUCCESS_PERCENT_LEARNED             = 99.5

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
            return self._minValue + ( max(0.0, min(1.0, value)) * (self._maxValue - self._minValue) )

        # ------------------------------------------------------

        def ToAnalog(self, value) :
            if type(value) not in (float, int) :
                raise MicroNN.ValueTypeException('Value must be of "float" or "int" type.')
            if value < self._minValue or value > self._maxValue :
                raise MicroNN.ValueTypeException('Value must be >= %s and <= %s.' % (self._minValue, self._maxValue))
            return ( float(value - self._minValue) / (self._maxValue - self._minValue) )

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
            self._neuronSrc           = neuronSrc
            self._neuronDst           = neuronDst
            self._weight              = weight if weight else ( (MicroNN.RandomFloat()-0.5) * 0.7 )
            self._momentumDeltaWeight = 0.0

        # -[ Methods ]------------------------------------------

        def UpdateWeight(self, errorCorrectionWeighting, connPlasticityStrengthing) :
            deltaWeight                = errorCorrectionWeighting \
                                       * self._neuronSrc.ComputedOutput \
                                       * self._neuronDst.ComputedSignalError
            self._weight              += deltaWeight \
                                       + (connPlasticityStrengthing * self._momentumDeltaWeight)
            self._momentumDeltaWeight  = deltaWeight

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

        def __init__(self, parentLayer, index) :
            if not isinstance(parentLayer, MicroNN.Layer) :
                raise MicroNN.NeuronException('"parentLayer" must be of Layer type.')
            if not isinstance(index, int) :
                raise MicroNN.NeuronException('"index" must be of "int" type.')
            self._parentLayer         = parentLayer
            self._index               = index
            self._inputConnections    = [ ]
            self._outputConnections   = [ ]
            self._computedInput       = 0.0
            self._computedOutput      = 0.0
            self._computedDeltaError  = 0.0
            self._computedSignalError = 0.0

        # -[ Methods ]------------------------------------------

        def ComputeInput(self) :
            self._computedInput = 0.0
            for conn in self._inputConnections :
                self._computedInput += conn.NeuronSrc.ComputedOutput * conn.Weight

        # ------------------------------------------------------

        def ComputeOutput(self) :
            self._computedOutput = self._parentLayer.Activation.Get(self)

        # ------------------------------------------------------

        def ComputeError(self, targetValue=None) :
            if targetValue is not None :
                self._computedDeltaError = targetValue - self._computedOutput
            else :
                self._computedDeltaError = 0.0
                for conn in self._outputConnections :
                    self._computedDeltaError += conn.NeuronDst._computedSignalError * conn.Weight
            self._computedSignalError = self._computedDeltaError \
                                      * self._parentLayer.Activation.GetDerivative(self)

        # ------------------------------------------------------

        def UpdateOutputWeights(self, errorCorrectionWeighting, connPlasticityStrengthing) :
            for conn in self._outputConnections :
                conn.UpdateWeight(errorCorrectionWeighting, connPlasticityStrengthing)

        # ------------------------------------------------------

        def GetInputConnsAsDataObject(self) :
            o = [ ]
            for conn in self._inputConnections :
                o.append([conn.NeuronSrc.Index, conn.Weight])
            return o

        # -[ Properties ]---------------------------------------

        @property
        def ParentLayer(self) :
            return self._parentLayer

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
        def ComputedInput(self) :
            return self._computedInput

        @property
        def ComputedOutput(self) :
            return self._computedOutput
        @ComputedOutput.setter
        def ComputedOutput(self, value) :
            if type(value) not in (float, int) :
                raise MicroNN.NeuronException('"value" must be of "float" or "int" type.')
            self._computedOutput = float(value)

        @property
        def ComputedDeltaError(self) :
            return self._computedDeltaError

        @property
        def ComputedSignalError(self) :
            return self._computedSignalError

        @property
        def Index(self) :
            return self._index

    # -------------------------------------------------------------------------
    # --( Class : Bias )-------------------------------------------------------
    # -------------------------------------------------------------------------

    class Bias(Neuron) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, parentLayer, index, biasValue=1.0) :
            super().__init__(parentLayer, index)
            self.ComputedOutput = biasValue

        # -[ Methods ]------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'Identity' }

    # -------------------------------------------------------------------------
    # --( Class : LayerException )---------------------------------------------
    # -------------------------------------------------------------------------

    class LayerException(Exception) :
        pass

    # -------------------------------------------------------------------------
    # --( Class : Layer )------------------------------------------------------
    # -------------------------------------------------------------------------

    class Layer :

        # -[ Constructor ]--------------------------------------

        def __init__( self,
                      parentMicroNN,
                      dimensions,
                      shape,
                      activation    = None,
                      connStruct    = None,
                      biasValue     = 1.0 ) :
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
                    raise MicroNN.LayerException('"activation" must be "None" for this layer type.')
                if not isinstance(activation, MicroNN.Activation) :
                    raise MicroNN.LayerException('"activation" must be of Activation type.')
            elif not isinstance(self, MicroNN.InputLayer) :
                raise MicroNN.LayerException('"activation" must be defined for this layer type.')
            if parentMicroNN.Layers :
                topLayer = parentMicroNN.Layers[len(parentMicroNN.Layers)-1]
            else :
                topLayer = None
            if topLayer is not None :
                if isinstance(self, MicroNN.InputLayer) :
                    raise MicroNN.LayerException('No layer must be present to add this InputLayer type.')
                if isinstance(topLayer, MicroNN.OutputLayer) :
                    raise MicroNN.LayerException('No layer can be added after an OutputLayer type.')
                if connStruct is None :
                    raise MicroNN.LayerException('"connStruct" must be defined for this layer type.')
                if not isinstance(connStruct, MicroNN.ConnStruct) :
                    raise MicroNN.LayerException('"connStruct" must be of ConnStruct type.')
            elif not isinstance(self, MicroNN.InputLayer) :
                raise MicroNN.LayerException('Only an InputLayer type can be added as first layer.')
            elif connStruct is not None :
                raise MicroNN.LayerException('"connStruct" must be "None" for this layer type.')
            if not isinstance(biasValue, float) or biasValue < 0 or biasValue > 1 :
                raise MicroNN.LayerException('"biasValue" must be of "float" type >= 0 and <= 1.')
            self._parentMicroNN  = parentMicroNN
            self._dimensions     = dimensions
            self._shape          = shape
            self._activation     = activation
            self._topLayer       = topLayer
            if not isinstance(self, MicroNN.InputLayer) :
                self._bias = MicroNN.Bias( parentLayer = self,
                                           index       = -1,
                                           biasValue   = biasValue )
            else :
                self._bias = None
            self._neuronsCount   = 0
            self._subDimNrnCount = self._getsubDimNrnCount()
            self._neurons        = self._recurCreateNeurons()
            self._inputConnCount = ( connStruct.ConnectLayer(self) if connStruct else 0 ) \
                                 + ( self._neuronsCount if self._bias else 0 )
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
                    n = MicroNN.Neuron( parentLayer = self,
                                        index       = self._neuronsCount )
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

        # ------------------------------------------------------

        def _recurComputeErrorAndUpdateWeights(self, neurons, computeError, dimIdx=0) :
            if dimIdx < self.DimensionsCount :
                for i in range(self._dimensions[dimIdx]) :
                    self._recurComputeErrorAndUpdateWeights(neurons[i], computeError, dimIdx+1)
            else :
                for i in range(self._shape.FlattenLen) :
                    if computeError :
                        neurons[i].ComputeError()
                    neurons[i].UpdateOutputWeights( self._parentMicroNN.ErrorCorrectionWeighting,
                                                    self._parentMicroNN.ConnPlasticityStrengthing )


        # ------------------------------------------------------

        def ComputeErrorAndUpdateWeights(self) :
            if not isinstance(self, MicroNN.OutputLayer) :
                computeError = (not isinstance(self, MicroNN.InputLayer))
                self._recurComputeErrorAndUpdateWeights(self._neurons, computeError)
            if self._bias :
                self._bias.UpdateOutputWeights( self._parentMicroNN.ErrorCorrectionWeighting,
                                                self._parentMicroNN.ConnPlasticityStrengthing )

        # ------------------------------------------------------

        def _recurGetOutputValues(self, neurons, dimIdx=0) :
            if dimIdx < self.DimensionsCount :
                dim = [ ]
                for i in range(self._dimensions[dimIdx]) :
                    dim.append(self._recurGetOutputValues(neurons[i], dimIdx+1))
                return dim
            else :
                flattenValues = [ neurons[i].ComputedOutput for i in range(self._shape.FlattenLen) ]
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
                    x += neurons[i].ComputedDeltaError ** 2
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
                    x += abs(neurons[i].ComputedDeltaError)
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
            if isinstance(self, MicroNN.InputLayer) :
                activation = None
                bias       = None
                conns      = None
            else :
                activation = self._activation.GetAsDataObject()
                bias       = { 'Value' : self._bias.ComputedOutput }
                conns      = self._recurGetConnsDataObject(self._neurons)
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
                    activation = MicroNN.Activation.CreateFromDataObject(o['Activation'])
                    biasValue  = o['Bias']['Value']
                    connStruct = MicroNN.DataObjectConnStruct(o['Connections'])
                    if layerType == 'OutputLayer' :
                        return MicroNN.OutputLayer( parentMicroNN,
                                                    dims,
                                                    shape,
                                                    activation,
                                                    connStruct,
                                                    biasValue )
                    elif layerType == 'Layer' :
                        return MicroNN.Layer( parentMicroNN,
                                              dims,
                                              shape,
                                              activation,
                                              connStruct,
                                              biasValue )
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
        def Bias(self) :
            return self._bias

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
                flattenValues = self._shape.Flatten(values)
                for i in range(self._shape.FlattenLen) :
                    neurons[i].ComputedOutput = flattenValues[i]

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
                flattenTargetValues = self._shape.Flatten(targetValues)
                for i in range(self._shape.FlattenLen) :
                    neurons[i].ComputeError(flattenTargetValues[i])

        # ------------------------------------------------------

        def ComputeTargetError(self, targetValues) :
            self._recurComputeTargetError(self._neurons, targetValues)

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

        ROLE_INCREASE = 0x0A
        ROLE_DECREASE = 0x0B

        # -[ Constructor ]--------------------------------------

        def __init__(self, role, overlappedShapesCount=0) :
            super().__init__()
            if role != self.ROLE_INCREASE and role != self.ROLE_DECREASE :
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
                    winStart = int( (lowIdx+0.5)*di.ratio - di.ratio/2 ) - self._overlappedShapesCount
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
                if self._role == self.ROLE_INCREASE :
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
            if self._role == self.ROLE_INCREASE :
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
                di         = dimInfo()
                di.ratio   = highLayer.Dimensions[i] / lowLayer.Dimensions[i]
                di.winSize = ceil(di.ratio) + (2 * self._overlappedShapesCount)
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
            return neuron.ComputedInput

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            return 1.0

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'Identity' }

    # -------------------------------------------------------------------------
    # --( Class : HeavisideActivation )----------------------------------------
    # -------------------------------------------------------------------------

    class HeavisideActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return 0.0 if neuron.ComputedInput < 0 else 1.0

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            return 1.0

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'Heaviside' }

    # -------------------------------------------------------------------------
    # --( Class : SigmoidActivation )------------------------------------------
    # -------------------------------------------------------------------------

    class SigmoidActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return 1.0 / ( 1.0 + exp(-neuron.ComputedInput) )

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            f = neuron.ComputedOutput
            return f * (1.0-f)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'Sigmoid' }

    # -------------------------------------------------------------------------
    # --( Class : TanHActivation )---------------------------------------------
    # -------------------------------------------------------------------------

    class TanHActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return ( 2.0 / (1.0 + exp(-2.0 * neuron.ComputedInput)) ) - 1.0

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            f = neuron.ComputedOutput
            return 1.0 - (f ** 2)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'TanH' }

    # -------------------------------------------------------------------------
    # --( Class : ReLUActivation )---------------------------------------------
    # -------------------------------------------------------------------------

    class ReLUActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return max(0.0, neuron.ComputedInput)

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            return 0.001 if neuron.ComputedInput < 0 else 1.0

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
            return max(self._alpha * neuron.ComputedInput, neuron.ComputedInput)

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            return self._alpha if neuron.ComputedInput < 0 else 1.0

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
            return log(1 + exp(neuron.ComputedInput))

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            return 1 / (1 + exp(-neuron.ComputedInput))

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'SoftPlus' }

    # -------------------------------------------------------------------------
    # --( Class : SinusoidActivation )-----------------------------------------
    # -------------------------------------------------------------------------

    class SinusoidActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return sin(neuron.ComputedInput)

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            return cos(neuron.ComputedInput)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'Sinusoid' }

    # -------------------------------------------------------------------------
    # --( Class : GaussianActivation )-----------------------------------------
    # -------------------------------------------------------------------------

    class GaussianActivation(Activation) :

        # -[ Methods ]------------------------------------------

        def Get(self, neuron) :
            return exp(-neuron.ComputedInput ** 2)

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            f = neuron.ComputedOutput
            return -2 * neuron.ComputedInput * f

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
                x = exp(self._layerNrnList[i].ComputedInput)
                self._layerNrnOutExps[i]  = x
                self._expsSum            += x

        # ------------------------------------------------------

        def Get(self, neuron) :
            return self._layerNrnOutExps[neuron.Index] / self._expsSum

        # ------------------------------------------------------

        def GetDerivative(self, neuron) :
            f = neuron.ComputedOutput
            return f * (1.0-f)

        # ------------------------------------------------------

        def GetAsDataObject(self) :
            return { 'Name' : 'SoftMax' }

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    # -[ Constructor ]--------------------------------------

    def __init__(self) :
        self._errorCorrectionWeighting  = MicroNN.DEFAULT_ERROR_CORRECTION_WEIGHTING
        self._connPlasticityStrengthing = MicroNN.DEFAULT_CONN_PLASTICITY_STRENGTHING
        self._layers                    = [ ]
        self._examples                  = [ ]

    # -[ Methods ]------------------------------------------

    @staticmethod
    def RandomFloat() :
        if 'rng' in globals() :
            return rng() / (2 ** 24)
        return random()

    # ------------------------------------------------------

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
                  connStruct  = None,
                  biasValue   = 1.0 ) :
        return MicroNN.Layer( parentMicroNN = self,
                              dimensions    = dimensions,
                              shape         = shape,
                              activation    = activation,
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
                        connStruct  = None,
                        biasValue   = 1.0 ) :
        return MicroNN.OutputLayer( parentMicroNN = self,
                                    dimensions    = dimensions,
                                    shape         = shape,
                                    activation    = activation,
                                    connStruct    = connStruct,
                                    biasValue     = biasValue )

    # ------------------------------------------------------

    def AddQLearningOutputLayer(self, actionsCount) :
        if not isinstance(actionsCount, int) or actionsCount <= 1 :
            raise MicroNNException('"actionsCount" must be of "int" type greater than 1.')
        return MicroNN.OutputLayer( parentMicroNN = self,
                                    dimensions    = MicroNN.Init1D(actionsCount),
                                    shape         = MicroNN.ValueShape(),
                                    activation    = MicroNN.SoftMaxActivation(),
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

    def Learn(self, inputValues, targetValues) :
        if not inputValues or not targetValues :
            raise MicroNNException('"inputValues" and "targetValues" must be defined.')
        self._simulate(inputValues, targetValues, True)

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
                       maxSeconds      = 30,
                       maxCount        = None,
                       stopWhenLearned = True,
                       verbose         = True ) :
        self._ensureNetworkIsComplete()
        if not isinstance(maxSeconds, int) or maxSeconds <= 0 :
            raise MicroNNException('"maxSeconds" must be of "int" type greater than zero.')
        if maxCount is not None :
            if not isinstance(maxCount, int) or maxCount <= 0 :
                raise MicroNNException('"maxCount" must be of "int" type greater than zero.')
        examplesCount = len(self._examples)
        count         = 0
        if examplesCount > 0 :
            endTime = time() + maxSeconds
            while time() < endTime and \
                  ( maxCount is None or count < maxCount ) :
                ex = self._examples[count % examplesCount]
                self.Learn(ex[0], ex[1])
                count += 1
                if count % examplesCount == 0 and (stopWhenLearned or printMAEAverage) :
                    successAvg = 0.0
                    for ex in self._examples :
                        successAvg += self.Test(ex[0], ex[1])
                    successAvg /= examplesCount
                    if verbose :
                        print( "MicroNN [ STEP : %s / SUCCESS : %s%% ]"
                               % ( count, round(successAvg*1000)/1000 ) )
                    if stopWhenLearned and successAvg >= MicroNN.SUCCESS_PERCENT_LEARNED :
                        break
        return count

    # ------------------------------------------------------

    def GetAsDataObject(self) :
        self._ensureNetworkIsComplete()
        try :
            return {
                'MicroNNVersion'            : MicroNN.VERSION,
                'ErrorCorrectionWeighting'  : self._errorCorrectionWeighting,
                'ConnPlasticityStrengthing' : self._connPlasticityStrengthing,
                'LayersCount'               : len(self._layers),
                'Layers'                    : [ l.GetAsDataObject()
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
            microNN                           = MicroNN()
            microNN.ErrorCorrectionWeighting  = o['ErrorCorrectionWeighting']
            microNN.ConnPlasticityStrengthing = o['ConnPlasticityStrengthing']
            for oLayer in o['Layers'] :
                MicroNN.Layer.CreateFromDataObject(microNN, oLayer)
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

    def _propagateSignal(self) :
        self._ensureNetworkIsComplete()
        for layer in self._layers :
            if not isinstance(layer, MicroNN.InputLayer) :
                layer.ComputeInput()
                layer.ComputeOutput()

    # ------------------------------------------------------

    def _backPropagateError(self) :
        self._ensureNetworkIsComplete()
        i = len(self._layers)-1
        while i >= 0 :
            self._layers[i].ComputeErrorAndUpdateWeights()
            i -= 1

    # ------------------------------------------------------

    def _simulate(self, inputValues, targetValues=None, training=False) :
        self._ensureNetworkIsComplete()
        if training and not targetValues :
            raise MicroNN.MicroNNException('"targetValues" must be defined when training is activated.')
        self.GetInputLayer().SetInputValues(inputValues)
        self._propagateSignal()
        if targetValues :
            self.GetOutputLayer().ComputeTargetError(targetValues)
            if training :
                self._backPropagateError()

    # -[ Properties ]---------------------------------------

    @property
    def ErrorCorrectionWeighting(self) :
        return self._errorCorrectionWeighting
    @ErrorCorrectionWeighting.setter
    def ErrorCorrectionWeighting(self, value) :
        if type(value) not in (float, int) :
            raise MicroNNException('"value" must be of "float" or "int" type.')
        self._errorCorrectionWeighting = float(value)

    @property
    def ConnPlasticityStrengthing(self) :
        return self._connPlasticityStrengthing
    @ConnPlasticityStrengthing.setter
    def ConnPlasticityStrengthing(self, value) :
        if type(value) not in (float, int) :
            raise MicroNNException('"value" must be of "float" or "int" type.')
        self._connPlasticityStrengthing = float(value)

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

MicroNN.Shape.Neuron   = MicroNN.ValueShape()
MicroNN.Shape.Bool     = MicroNN.ValueShape(MicroNN.BoolValueType())
MicroNN.Shape.Byte     = MicroNN.ValueShape(MicroNN.ByteValueType())
MicroNN.Shape.Percent  = MicroNN.ValueShape(MicroNN.PercentValueType())
MicroNN.Shape.Color    = MicroNN.ColorShape()
MicroNN.FullyConnected = MicroNN.FullyConnStruct()
