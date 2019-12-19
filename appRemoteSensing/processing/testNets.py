#Entrenamiento de diferentes arquitecturas de DNN: CNN2D 
# pylint: disable=E1136  # pylint/issues/3139
import warnings
warnings.filterwarnings('ignore')
from .cargarHsi import CargarHsi
from .prepararDatos import PrepararDatos
from .PCA import princiapalComponentAnalysis
from .MorphologicalProfiles import morphologicalProfiles
from .dataLogger import DataLogger
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout
from keras.models import Model
from keras import regularizers
from keras import backend as K 
from keras.models import load_model
from sklearn.metrics import cohen_kappa_score
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import os 

class testNetworks:
    def __init__(self):
        pass

    def __str__(self):
        pass
    
    def euclidean_distance_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    def reshapeFeaturesSVM(self, features_test):
        features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
        features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2])
        return features_test

    def reshapeFeaturesRIE(self, features_test):
        features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
        features_test = np.transpose(features_test, (0, 2, 1))
        return features_test

    def accuracy(self, y_true, y_pred):
        if y_true.ndim>1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim>1:
            y_pred = np.argmax(y_pred, axis=1)

        OA = accuracy_score(y_true, y_pred)
        return OA

    def testDNN(self, dataSet, test, clss, imagenFE, groundTruth):
        ################################################################################
        #####################PREPARAR DATOS PARA VALIDACIÓN#############################
        if test[-3:] == 'CNN' or test[-3:] == 'INC':
            ventana = 9
        else:
            ventana = 8           
        preparar = PrepararDatos(imagenFE, groundTruth, False)
        datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)        
        ################################################################################
        ##########################CREAR PATH AL DATA LOGGER#############################
        logger = DataLogger(fileName = dataSet+'_'+clss, folder = test, save = False) 
        ################################################################################
        ####################CARGAR RED DE EXTRACCIÓN DE CARACTERISTICAS#################
        if clss != 'LRC':
            if test[-3:] == 'CNN' or test[-3:] == 'INC':
                feNet = load_model(os.path.join(logger.path,test+'.h5'))
            else:
                feNet = load_model(os.path.join(logger.path,test+'.h5'), custom_objects={'euclidean_distance_loss': self.euclidean_distance_loss})  
        ################################################################################
        ################Generar caracteristicicas con los datos de entrada##############
            features_test = feNet.predict(datosPrueba)
            if clss == 'SVM':
                features_test = self.reshapeFeaturesSVM(features_test)
            if clss == 'RIE':
                features_test = self.reshapeFeaturesRIE(features_test)
        #######################Cargar el clasificador seleccionado######################
            clssNet = joblib.load(os.path.join(logger.path,test+'_'+clss+'.pkl'))
            datosSalida = clssNet.predict(features_test)
        #######################Ejecutar clasificador seleccionado#######################
        else:
        #######################Cargar el clasificador seleccionado######################
            clssNet = load_model(os.path.join(logger.path,test+'_'+clss+'.h5'))
        #######################Ejecutar clasificador seleccionado#######################
            datosSalida = clssNet.predict(datosPrueba)

        ################################################################################
        #########################GENERAR MAPA FINAL DE CLASIFICACIÓN####################
        imagenSalida = preparar.predictionToImage(datosSalida)
        ##############################IMAGEN DE COMPARACION#############################
        imgCompare = np.absolute(groundTruth-imagenSalida)
        imgCompare = (imgCompare > 0) * 1 
        ################################################################################
        np.save('ClassificationMap', imagenSalida) #Genera archivo con los datos de Salida
        np.save('imageCompare', imgCompare) #Genera archivo con los datos de Salida
        #TERMINAR PROCESOS DE KERAS
        K.clear_session()
        path = logger.path
        logger.close()
        #RETORNAR ETIQUETAS DE PRUEBA Y PREDICCIÓN
        class_names = []
        for i in range(1,groundTruth.max()+1):
            class_names.append('Class '+str(i))
        if etiquetasPrueba.ndim>1:
            etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
        if datosSalida.ndim>1:
            datosSalida = datosSalida.argmax(axis=1)

        return etiquetasPrueba, datosSalida, class_names, path