"""
The MIT License (MIT)
Copyright © 2018 Jean-Christophe Bos & HC² (www.hc2.fr)
"""


from math import sqrt

try :
    from random import random
except :
    from machine import rng

class MicroNN :

    # -------------------------------------------------------------------------
    # --( Class : Shape )------------------------------------------------------
    # -------------------------------------------------------------------------

    class Shape :

        # -[ Methods ]------------------------------------------

        def Zero(self) :
            return 0

        def FlattenSize(self) :
            return 1

        def FlattenZero(self) :
            return [[0]]

        def Flatten(self, value) :
            if not value :
                raise Exception('Shape : Bad value.')
            return [[value]]

        def Unflatten(self, vect) :
            if not vect or len(vect) != 1 :
                raise Exception('Shape : Bad vector.')
            return vect[0][0]

    # -------------------------------------------------------------------------
    # --( Class : VectorShape )------------------------------------------------
    # -------------------------------------------------------------------------

    class VectorShape(Shape) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, size) :
            self._size = size

        # -[ Methods ]------------------------------------------

        def Zero(self) :
            return [0] * self._size

        def FlattenSize(self) :
            return self._size

        def FlattenZero(self) :
            return [[0]] * self._size

        def Flatten(self, vect) :
            if not vect or len(vect) != self._size :
                raise Exception('VectorShape : Bad size.')
            return [[vect[x]] for x in range(self._size)]

        def Unflatten(self, vect) :
            if not vect or len(vect) != self._size :
                raise Exception('VectorShape : Bad size.')
            return [vect[x][0] for x in range(self._size)]

    # -------------------------------------------------------------------------
    # --( Class : Matrix2DShape )----------------------------------------------
    # -------------------------------------------------------------------------

    class Matrix2DShape(Shape) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, sizeX, sizeY) :
            self._sizeX = sizeX
            self._sizeY = sizeY

        # -[ Methods ]------------------------------------------

        def Zero(self) :
            return [ [ 0
                       for _ in range(self._sizeY) ]
                     for _ in range(self._sizeX) ]

        def FlattenSize(self) :
            return self._sizeX * self._sizeY

        def FlattenZero(self) :
            return [[0]] * self.FlattenSize()

        def Flatten(self, matrix2D) :
            return [ [matrix2D[x][y]]
                     for x in range(self._sizeX)
                     for y in range(self._sizeY) ]

        def Unflatten(self, vect) :
            return [ [ vect[ x * self._sizeY + y ][0]
                       for y in range(self._sizeY) ]
                     for x in range(self._sizeX) ]

    # -------------------------------------------------------------------------
    # --( Class : Matrix3DShape )----------------------------------------------
    # -------------------------------------------------------------------------

    class Matrix3DShape(Shape) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, sizeX, sizeY, sizeZ) :
            self._sizeX = sizeX
            self._sizeY = sizeY
            self._sizeZ = sizeZ

        # -[ Methods ]------------------------------------------

        def Zero(self) :
            return [ [ [ 0
                         for _ in range(self._sizeZ) ]
                       for _ in range(self._sizeY) ]
                     for _ in range(self._sizeX) ]

        def FlattenSize(self) :
            return self._sizeX * self._sizeY * self._sizeZ

        def FlattenZero(self) :
            return [[0]] * self.FlattenSize()

        def Flatten(self, matrix3D) :
            return [ [matrix3D[x][y][z]]
                     for x in range(self._sizeX)
                     for y in range(self._sizeY)
                     for z in range(self._sizeZ) ]

        def Unflatten(self, vect) :
            return [ [ [ vect[ x * self._sizeY * self._sizeZ + y * self._sizeZ + z ][0]
                         for z in range(self._sizeZ) ]
                       for y in range(self._sizeY) ]
                     for x in range(self._sizeX) ]

    # -------------------------------------------------------------------------
    # --( Class : Tensor )-----------------------------------------------------
    # -------------------------------------------------------------------------

    class Tensor :

        # -[ Constructor ]--------------------------------------

        def __init__(self, shape) :
            if not isinstance(shape, MicroNN.Shape) :
                raise Exception('Tensor : "shape" must be an instanciated Shape.')
            self._shape = shape
            self._data  = shape.FlattenZero()

        # -[ Methods ]------------------------------------------

        def GetShape(self) :
            return self._shape

        def SetShapeData(self, shapeData) :
            self._data = self._shape.Flatten(shapeData)

        def GetShapeData(self) :
            return self._shape.Unflatten(self._data)

        def GetShapeCount(self) :
            return 1

        def SetFlattenDataByIndex(self, index, flattenData) :
            if index < 0 or index >= self.GetShapeCount() :
                raise Exception('Tensor : "index" out of bound.')
            self._data[index] = flattenData

        def GetFlattenDataByIndex(self, index) :
            if index < 0 or index >= self.GetShapeCount() :
                raise Exception('Tensor : "index" out of bound.')
            return self._data[index]

        def SetFlattenData(self, flattenData) :
            if not flattenData or len(flattenData) != self._shape.FlattenSize() :
                raise Exception('Tensor : "flattenData" is incorrect.')
            self._data = flattenData

        def GetFlattenData(self) :
            return self._data

        def GetData(self) :
            return self._data

        def SetData(self, data) :
            self._data = data

        def CloneZero(self) :
            return MicroNN.Tensor(self._shape)

    # -------------------------------------------------------------------------
    # --( Class : VectorTensor )-----------------------------------------------
    # -------------------------------------------------------------------------

    class VectorTensor(Tensor) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, size, shape) :
            super().__init__(shape)
            self._data = [shape.FlattenZero()] * size
            self._size = size

        # -[ Methods ]------------------------------------------

        def SetShapeData(self, index, shapeData) :
            self._data[index] = self._shape.Flatten(shapeData)

        def GetShapeData(self, index) :
            return self._shape.Unflatten(self._data[index])

        def GetShapeCount(self) :
            return self._size

        def SetFlattenData(self, index, flattenData) :
            if not flattenData or len(flattenData) != self._shape.FlattenSize() :
                raise Exception('VectorTensor : "flattenData" is incorrect.')
            self._data[index] = flattenData

        def GetFlattenData(self, index) :
            return self._data[index]

        def CloneZero(self) :
            return MicroNN.VectorTensor(self._size, self._shape)

        # -[ Properties ]---------------------------------------

        @property
        def Size(self) :
            return self._size

    # -------------------------------------------------------------------------
    # --( Class : Matrix2DTensor )---------------------------------------------
    # -------------------------------------------------------------------------

    class Matrix2DTensor(Tensor) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, sizeX, sizeY, shape) :
            super().__init__(shape)
            self._data  = [shape.FlattenZero()] * (sizeX * sizeY)
            self._sizeX = sizeX
            self._sizeY = sizeY

        # -[ Methods ]------------------------------------------

        def _transToIndex(self, x, y) :
            return x * self._sizeY + y

        def SetShapeData(self, x, y, shapeData) :
            self._data[self._transToIndex(x, y)] = self._shape.Flatten(shapeData)

        def GetShapeData(self, x, y) :
            return self._shape.Unflatten(self._data[self._transToIndex(x, y)])

        def GetShapeCount(self) :
            return self._sizeX * self._sizeY

        def SetFlattenData(self, x, y, flattenData) :
            if not flattenData or len(flattenData) != self._shape.FlattenSize() :
                raise Exception('Matrix2DTensor : "flattenData" is incorrect.')
            self._data[self._transToIndex(x, y)] = flattenData

        def GetFlattenData(self, x, y) :
            return self._data[self._transToIndex(x, y)]

        def CloneZero(self) :
            return MicroNN.Matrix2DTensor(self._sizeX, self._sizeY, self._shape)

        # -[ Properties ]---------------------------------------

        @property
        def SizeX(self) :
            return self._sizeX

        @property
        def SizeY(self) :
            return self._sizeY

    # -------------------------------------------------------------------------
    # --( Class : Matrix3DTensor )---------------------------------------------
    # -------------------------------------------------------------------------

    class Matrix3DTensor(Tensor) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, sizeX, sizeY, sizeZ, shape) :
            super().__init__(shape)
            self._data  = [shape.FlattenZero()] * (sizeX * sizeY * sizeZ)
            self._sizeX = sizeX
            self._sizeY = sizeY
            self._sizeZ = sizeZ

        # -[ Methods ]------------------------------------------

        def _transToIndex(self, x, y, z) :
            return x * self._sizeY * self._sizeZ + y * self._sizeZ + z

        def SetShapeData(self, x, y, z, shapeData) :
            self._data[self._transToIndex(x, y, z)] = self._shape.Flatten(shapeData)

        def GetShapeData(self, x, y, z) :
            return self._shape.Unflatten(self._data[self._transToIndex(x, y, z)])

        def GetShapeCount(self) :
            return self._sizeX * self._sizeY * self._sizeZ

        def SetFlattenData(self, x, y, z, flattenData) :
            if not flattenData or len(flattenData) != self._shape.FlattenSize() :
                raise Exception('Matrix3DTensor : "flattenData" is incorrect.')
            self._data[self._transToIndex(x, y, z)] = flattenData

        def GetFlattenData(self, x, y, z) :
            return self._data[self._transToIndex(x, y, z)]

        def CloneZero(self) :
            return MicroNN.Matrix3DTensor(self._sizeX, self._sizeY, self._sizeZ, self._shape)

        # -[ Properties ]---------------------------------------

        @property
        def SizeX(self) :
            return self._sizeX

        @property
        def SizeY(self) :
            return self._sizeY

        @property
        def SizeZ(self) :
            return self._sizeZ

    # -------------------------------------------------------------------------
    # --( Class : BaseLayer )--------------------------------------------------
    # -------------------------------------------------------------------------

    class BaseLayer :

        # -[ Constructor ]--------------------------------------

        def __init__(self, lowerLayer=None) :
            if lowerLayer and not isinstance(lowerLayer, MicroNN.Layer) :
                raise Exception('BaseLayer : "lowerLayer" must be an instanciated Layer.')
            self._lowerLayer       = lowerLayer
            self._outputTensorOut  = None
            self._deltaErrorTensor = None

        # -[ Properties ]---------------------------------------

        @property
        def OutputTensor(self) :
            return self._outputTensorOut

        @property
        def DeltaErrorTensor(self) :
            return self._deltaErrorTensor

    # -------------------------------------------------------------------------
    # --( Class : InputLayer )-------------------------------------------------
    # -------------------------------------------------------------------------

    class InputLayer(BaseLayer) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, valuesTensor) :
            super().__init__()
            if not isinstance(valuesTensor, MicroNN.Tensor) :
                raise Exception('InputLayer : "valuesTensor" must be an instanciated Tensor.')
            self._outputTensorOut  = valuesTensor
            self._deltaErrorTensor = valuesTensor.CloneZero()

    # -------------------------------------------------------------------------
    # --( Class : Layer )------------------------------------------------------
    # -------------------------------------------------------------------------

    class Layer(BaseLayer) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, lowerLayer, outputTensor) :
            super().__init__(lowerLayer)
            if not isinstance(outputTensor, MicroNN.Tensor) :
                raise Exception('Layer : "outputTensor" must be an instanciated Tensor.')
            self._inputTensor             = lowerLayer.OutputTensor
            self._outputTensorIn          = outputTensor.CloneZero()
            self._outputTensorBias        = outputTensor.CloneZero()
            self._outputTensorOut         = outputTensor
            self._deltaErrorTensor        = outputTensor.CloneZero()
            self._signalErrorTensor       = outputTensor.CloneZero()
            self._inputShape              = self._inputTensor.GetShape()
            self._outputShape             = self._outputTensorIn.GetShape()
            self._windowShapeCount        = 0
            self._windowSideShapeCount    = 0
            self._setConnections(0)

        # -[ Methods ]------------------------------------------

        def _calcConnectionsCount(self, windowShapeCount) :
            return self._outputTensorIn.GetShapeCount() * \
                   self._outputShape.FlattenSize() * \
                   windowShapeCount * \
                   self._inputShape.FlattenSize()

        def _setConnections(self, connCount) :
            self._connCount      = connCount
            self._connSrcDataPtr = [[None]] * connCount
            self._connDstDataPtr = [[None]] * connCount
            self._connWeight     = [None]   * connCount
            self._connMemWeight  = [0.0]    * connCount

        def _randomWeight(self) :
            maxFloat = 2 ** 24
            for i in range(self._connCount) :
                self._connWeight[i] = ( rng()/maxFloat - 0.5 ) * 0.7

        def UpdateWeight(self, eta, alpha) :
            """
            for i in range(self._connCount) :
                deltaWeight                = eta \
                                           * self._neuronSrc.ComputedOutput \
                                           * self._neuronDst.ComputedSignalError
                self._weight              += deltaWeight + (alpha * self._momentumDeltaWeight)
                self._momentumDeltaWeight  = deltaWeight
            """

    # -------------------------------------------------------------------------
    # --( Class : Layer1D )----------------------------------------------------
    # -------------------------------------------------------------------------

    class Layer1D(Layer) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, lowerLayer, outputTensor, windowShapeCount) :
            super().__init__(lowerLayer, outputTensor)
            if not isinstance(lowerLayer.OutputTensor, MicroNN.VectorTensor) :
                raise Exception('Layer1D : "lowerLayer" must have a VectorTensor output.')
            if not isinstance(outputTensor, MicroNN.VectorTensor) :
                raise Exception('Layer1D : "outputTensor" must be a VectorTensor.')
            self._layersRatio          = self._inputTensor.Size / outputTensor.Size
            self._windowShapeCount     = windowShapeCount
            self._windowSideShapeCount = windowShapeCount
            if self._windowSideShapeCount <= 0 or \
               self._windowSideShapeCount > self._inputTensor.GetShapeCount() :
                raise Exception('Layer1D : "windowShapeCount" is out of bound.')
            self._setConnections(self._calcConnectionsCount(windowShapeCount))
            self._setConnectionsDataPtr(self)

        # -[ Methods ]------------------------------------------

        def _getWindowShapePos(self, outputShapePos) :
            pos = round( (outputShapePos + 0.5) * self._layersRatio - self._windowSideShapeCount/2 )
            pos = max(0, pos)
            pos = min(pos, self._inputTensor.Size-self._windowSideShapeCount)
            return pos

        def _setConnectionsDataPtr(self) :
            connIdx = 0
            for outputShapePos in range(self._outputTensorIn.Size) :
                outputFlattenData = self._outputTensorIn.GetFlattenData(outputShapePos)
                winPos            = self._getWindowShapePos(outputShapePos)
                for x in range(self._windowSideShapeCount) :
                    inputShapePos    = winPos + x
                    inputFlattenData = self._inputTensor.GetFlattenData(inputShapePos)
                    for inputDataIdx in range(self._inputShape.FlattenSize()) :
                        for outputDataIdx in range(self._outputShape.FlattenSize()) :
                            self._connSrcDataPtr[connIdx] = inputFlattenData[inputDataIdx]
                            self._connDstDataPtr[connIdx] = outputFlattenData[outputDataIdx]
                            connIdx += 1

    # -------------------------------------------------------------------------
    # --( Class : Layer2D )----------------------------------------------------
    # -------------------------------------------------------------------------

    class Layer2D(Layer) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, lowerLayer, outputTensor, windowShapeCount) :
            super().__init__(lowerLayer, outputTensor)
            if not isinstance(lowerLayer.OutputTensor, MicroNN.Matrix2DTensor) :
                raise Exception('Layer2D : "lowerLayer" must have a Matrix2DTensor output.')
            if not isinstance(outputTensor, MicroNN.Matrix2DTensor) :
                raise Exception('Layer2D : "outputTensor" must be a Matrix2DTensor.')
            self._layersRatioX         = self._inputTensor.SizeX / outputTensor.SizeX
            self._layersRatioY         = self._inputTensor.SizeY / outputTensor.SizeY
            self._windowShapeCount     = windowShapeCount
            self._windowSideShapeCount = round(sqrt(windowShapeCount))
            if self._windowSideShapeCount ** 2 != windowShapeCount :
                raise Exception('Layer2D : "windowShapeCount" must be a square value.')
            if self._windowSideShapeCount <= 0 or \
               self._windowSideShapeCount > self._inputTensor.SizeX or \
               self._windowSideShapeCount > self._inputTensor.SizeY :
                raise Exception('Layer2D : "windowShapeCount" is out of bound.')
            self._setConnections(self._calcConnectionsCount(windowShapeCount))
            self._setConnectionsDataPtr(self)

        # -[ Methods ]------------------------------------------

        def _getWindowShapePos(self, outputShapePosX, outputShapePosY) :
            half = self._windowSideShapeCount / 2
            posX = round( (outputShapePosX + 0.5) * self._layersRatioX - half )
            posY = round( (outputShapePosY + 0.5) * self._layersRatioY - half )
            posX = max(0, posX)
            posY = max(0, posY)
            posX = min(posX, self._inputTensor.SizeX-self._windowSideShapeCount)
            posY = min(posY, self._inputTensor.SizeY-self._windowSideShapeCount)
            return (posX, posY)

        def _setConnectionsDataPtr(self) :
            connIdx = 0
            for outputShapePosX in range(self._outputTensorIn.SizeX) :
                for outputShapePosY in range(self._outputTensorIn.SizeY) :
                    outputFlattenData = self._outputTensorIn.GetFlattenData(outputShapePosX, outputShapePosY)
                    winPos            = self._getWindowShapePos(outputShapePosX, outputShapePosY)
                    for x in range(self._windowSideShapeCount) :
                        for y in range(self._windowSideShapeCount) :
                            inputShapePosX   = winPos[0] + x
                            inputShapePosY   = winPos[1] + y
                            inputFlattenData = self._inputTensor.GetFlattenData(inputShapePosX, inputShapePosY)
                            for inputDataIdx in range(self._inputShape.FlattenSize()) :
                                for outputDataIdx in range(self._outputShape.FlattenSize()) :
                                    self._connSrcDataPtr[connIdx] = inputFlattenData[inputDataIdx]
                                    self._connDstDataPtr[connIdx] = outputFlattenData[outputDataIdx]
                                    connIdx += 1

    # -------------------------------------------------------------------------
    # --( Class : Layer3D )----------------------------------------------------
    # -------------------------------------------------------------------------

    
    class Layer3D(Layer) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, lowerLayer, outputTensor, windowShapeCount) :
            super().__init__(lowerLayer, outputTensor)
            if not isinstance(lowerLayer.OutputTensor, MicroNN.Matrix3DTensor) :
                raise Exception('Layer3D : "lowerLayer" must have a Matrix3DTensor output.')
            if not isinstance(outputTensor, MicroNN.Matrix3DTensor) :
                raise Exception('Layer3D : "outputTensor" must be a Matrix3DTensor.')
            self._layersRatioX         = self._inputTensor.SizeX / outputTensor.SizeX
            self._layersRatioY         = self._inputTensor.SizeY / outputTensor.SizeY
            self._layersRatioZ         = self._inputTensor.SizeZ / outputTensor.SizeZ
            self._windowShapeCount     = windowShapeCount
            self._windowSideShapeCount = round(windowShapeCount ** (1.0/3))
            if self._windowSideShapeCount ** 3 != windowShapeCount :
                raise Exception('Layer3D : "windowShapeCount" must be a cube value.')
            if self._windowSideShapeCount <= 0 or \
               self._windowSideShapeCount > self._inputTensor.SizeX or \
               self._windowSideShapeCount > self._inputTensor.SizeY or \
               self._windowSideShapeCount > self._inputTensor.SizeZ :
                raise Exception('Layer3D : "windowShapeCount" is out of bound.')
            self._setConnections(self._calcConnectionsCount(windowShapeCount))
            self._setConnectionsDataPtr(self)

        # -[ Methods ]------------------------------------------

        def _getWindowShapePos(self, outputShapePosX, outputShapePosY, outputShapePosZ) :
            half = self._windowSideShapeCount / 2
            posX = round( (outputShapePosX + 0.5) * self._layersRatioX - half )
            posY = round( (outputShapePosY + 0.5) * self._layersRatioY - half )
            posZ = round( (outputShapePosZ + 0.5) * self._layersRatioZ - half )
            posX = max(0, posX)
            posY = max(0, posY)
            posZ = max(0, posZ)
            posX = min(posX, self._inputTensor.SizeX-self._windowSideShapeCount)
            posY = min(posY, self._inputTensor.SizeY-self._windowSideShapeCount)
            posZ = min(posZ, self._inputTensor.SizeZ-self._windowSideShapeCount)
            return (posX, posY, posZ)

        def _setConnectionsDataPtr(self) :
            connIdx = 0
            for outputShapePosX in range(self._outputTensorIn.SizeX) :
                for outputShapePosY in range(self._outputTensorIn.SizeY) :
                    for outputShapePosZ in range(self._outputTensorIn.SizeZ) :
                        outputFlattenData = self._outputTensorIn.GetFlattenData(outputShapePosX, outputShapePosY, outputShapePosZ)
                        winPos            = self._getWindowShapePos(outputShapePosX, outputShapePosY, outputShapePosZ)
                        for x in range(self._windowSideShapeCount) :
                            for y in range(self._windowSideShapeCount) :
                                for z in range(self._windowSideShapeCount) :
                                    inputShapePosX   = winPos[0] + x
                                    inputShapePosY   = winPos[1] + y
                                    inputShapePosZ   = winPos[2] + z
                                    inputFlattenData = self._inputTensor.GetFlattenData(inputShapePosX, inputShapePosY, inputShapePosZ)
                                    for inputDataIdx in range(self._inputShape.FlattenSize()) :
                                        for outputDataIdx in range(self._outputShape.FlattenSize()) :
                                            self._connSrcDataPtr[connIdx] = inputFlattenData[inputDataIdx]
                                            self._connDstDataPtr[connIdx] = outputFlattenData[outputDataIdx]
                                            connIdx += 1

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    # -[ Constructor ]--------------------------------------

    def __init__(self) :
        pass

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
