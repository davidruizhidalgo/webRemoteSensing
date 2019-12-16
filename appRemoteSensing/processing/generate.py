#Toma los datos del formulario y ejecuta las configuraciones establecidas
# pylint: disable=E1136  # pylint/issues/3139
from .cargarHsi import CargarHsi
from .PCA import princiapalComponentAnalysis
from .MorphologicalProfiles import morphologicalProfiles
from .trainNets import trainNetworks
import numpy as np

class Generate:
    def __init__(self,object):
        self.object = object

    def __str__(self):
        pass

    def cargarImagen(self):
        data = CargarHsi(self.object.name)
        imagen = data.imagen
        groundTruth = data.groundTruth
        np.save('groundTruth', groundTruth) #Genera archivo con los datos del Ground Truth
        return imagen, groundTruth
    
    def pca_Analysis(self, imagen):
        pca = princiapalComponentAnalysis()
        imagenPCA = pca.pca_calculate(imagen, componentes=7)
        return imagenPCA  

    def execution(self):
        imagen, groundTruth = self.cargarImagen()
        #Etapa de reducción dimensional
        if self.object.dimension == 'NON':
            pass
        if self.object.dimension == 'PCA':
            imagen = self.pca_Analysis(imagen)
        if self.object.dimension == 'EAP':
            imagen = self.pca_Analysis(imagen)
            mp = morphologicalProfiles()
            imagen = mp.EAP(imagen, num_thresholds=6)   
        if self.object.dimension == 'EEP':
            imagen = self.pca_Analysis(imagen)
            mp = morphologicalProfiles()
            imagen =  mp.EEP(imagen, num_levels=4)      
        
        #Etapa de Extracción de caracteristicas
        feNet = trainNetworks()
        feModel = feNet.trainCNN2d(self.object.name, self.object.dimension+'_'+self.object.features, imagen, groundTruth)
        #print(self.object.features)

        #Etapa de Selección de información
        #print(self.object.classifier)

        #GENERAR IMAGEN MAPA DE CLASIFICACIÓN y COMPARACIÓN
        
        return feModel