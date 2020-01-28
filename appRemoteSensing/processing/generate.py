#Toma los datos del formulario y ejecuta las configuraciones establecidas
# pylint: disable=E1136  # pylint/issues/3139
from .cargarHsi import CargarHsi
from .PCA import princiapalComponentAnalysis
from .MorphologicalProfiles import morphologicalProfiles
from .trainNets import trainNetworks
from .testNets import testNetworks
import numpy as np
import os

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

    def execution(self, train= False):
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
        
        #GENERACIÓN O CARGA DE MODELO: FEATURE EXTRACTION + CLASSIFIER
        try:
            if train: #ENTRENAMIENTO RED NEURONAL
                dnnNet = trainNetworks()
                if self.object.features == 'CNN':
                    imagenSalida, imgCompare, true_labels_test, pred_labels_test, class_names, path = dnnNet.trainCNN2d(self.object.name, self.object.dimension+'_'+self.object.features, self.object.classifier, imagen, groundTruth)
                if self.object.features == 'INC':
                    imagenSalida, imgCompare, true_labels_test, pred_labels_test, class_names, path = dnnNet.trainInception(self.object.name, self.object.dimension+'_'+self.object.features, self.object.classifier, imagen, groundTruth)
                if self.object.features == 'SCA':
                    imagenSalida, imgCompare, true_labels_test, pred_labels_test, class_names, path = dnnNet.trainScae(self.object.name, self.object.dimension+'_'+self.object.features, self.object.classifier, imagen, groundTruth)
                if self.object.features == 'BCA':  
                    imagenSalida, imgCompare, true_labels_test, pred_labels_test, class_names, path = dnnNet.trainBcae(self.object.name, self.object.dimension+'_'+self.object.features, self.object.classifier, imagen, groundTruth)
            else:     #CARGA RED NEURONAL
                dnnModel = testNetworks()
                imagenSalida, imgCompare ,true_labels_test, pred_labels_test, class_names, path = dnnModel.testDNN(self.object.name, self.object.dimension+'_'+self.object.features, self.object.classifier, imagen, groundTruth)
            #CARGA LOS DATOS OA, AA y K para enviar al Front End
            filetxt = 'logger_'+self.object.name+'_'+self.object.classifier+'.txt'      
            table  = np.loadtxt(os.path.join(path,filetxt))
        except:
            table  = np.zeros((3,10))
            true_labels_test = np.zeros(10)
            pred_labels_test = np.zeros(10)
            class_names  = ['0','1','2','3','4','5','6','7','8','9']
            imagenSalida = np.zeros((groundTruth.shape[0], groundTruth.shape[1]))
            imgCompare = np.zeros((groundTruth.shape[0], groundTruth.shape[1]))

        return groundTruth, imagenSalida, imgCompare, table, true_labels_test, pred_labels_test, class_names 