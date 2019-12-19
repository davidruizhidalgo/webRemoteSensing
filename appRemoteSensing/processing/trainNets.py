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

class trainNetworks:
    def __init__(self):
        pass

    def __str__(self):
        pass
    
    def reshapeFeaturesSVM(self, features_test):
        features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
        features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2])
        return features_test

    def reshapeFeaturesRIE(self, features_test):
        features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
        features_test = np.transpose(features_test, (0, 2, 1))
        return features_test

    def svm_classifier(self,features_tr, etiquetasEntrenamiento, kernel='linear'):
        #Reshape features (n_samples, m_features)
        features_tr = features_tr.reshape(features_tr.shape[0],features_tr.shape[1]*features_tr.shape[2],features_tr.shape[3])
        features_tr = features_tr.reshape(features_tr.shape[0],features_tr.shape[1]*features_tr.shape[2])
        #Reshape labels from categorical 
        etiquetasEntrenamiento = np.argmax(etiquetasEntrenamiento, axis=1)
        #SVM Classifier one-against-one
        classifier = svm.SVC(gamma='scale', decision_function_shape='ovo', kernel=kernel, verbose=False)
        classifier.fit(features_tr,etiquetasEntrenamiento)
        return classifier

    def riemann_classifier(self, features_tr, etiquetasEntrenamiento, method='tan'):
        #Reshape features (n_samples, m_filters, p_features)
        features_tr = features_tr.reshape(features_tr.shape[0],features_tr.shape[1]*features_tr.shape[2],features_tr.shape[3])
        features_tr = np.transpose(features_tr, (0, 2, 1))
        #Reshape labels from categorical 
        etiquetasEntrenamiento = np.argmax(etiquetasEntrenamiento, axis=1)

        #Riemannian Classifier using Minimum Distance to Riemannian Mean
        covest = Covariances(estimator='lwf')
        if method == 'mdm':
            mdm = MDM()
            classifier = make_pipeline(covest,mdm)
        if method == 'tan':
            ts = TangentSpace()
            lda = LinearDiscriminantAnalysis()
            classifier = make_pipeline(covest,ts,lda)
        
        classifier.fit(features_tr, etiquetasEntrenamiento)
        return classifier

    def accuracy(self, y_true, y_pred):
        if y_true.ndim>1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim>1:
            y_pred = np.argmax(y_pred, axis=1)

        OA = accuracy_score(y_true, y_pred)
        return OA

    def trainCNN2d(self, dataSet, test, imagenFE, groundTruth):
        epochs = 35
        #CREAR FICHERO DATA LOGGER 
        logger = DataLogger(fileName = dataSet+'_LRC', folder = test, save = True)      
        #PREPARAR DATOS PARA ENTRENAMIENTO
        ventana = 9  #VENTANA 2D de PROCESAMIENTO
        preparar = PrepararDatos(imagenFE, groundTruth, False)
        datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
        datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)
        #DEFINICION RED CONVOLUCIONAL
        input_layer = Input(shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3])) 
        x = Conv2D(48, (5, 5), kernel_regularizer=regularizers.l2(0.001),activation='relu')(input_layer)
        x = Conv2D(96, (3, 3), kernel_regularizer=regularizers.l2(0.001),activation='relu')(x)
        x = Conv2D(96, (3, 3), kernel_regularizer=regularizers.l2(0.001),activation='relu', padding='same')(x)
        ################################## CLASIFICADOR LOGISTIC REGRESION ############################################
        #CAPA FULLY CONNECTED
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)
        output_layer = Dense(groundTruth.max()+1, activation='softmax')(x)
        lrc_net = Model(input_layer,output_layer) 
        #ENTRENAMIENTO DE LA RED CONVOLUCIONAL
        lrc_net.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        print('LOGGISTIC REGRESION CLASSIFIER #####################################################################')
        history = lrc_net.fit(datosEntrenamiento,etiquetasEntrenamiento,epochs=epochs,batch_size=128,validation_data=(datosValidacion, etiquetasValidacion))
        #LOGGER DATOS DE ENTRENAMIENTO
        #logger.savedataTrain(history)
        #GUARDAR MODELO DE RED CONVOLUCIONAL
        cnn_net = Model(inputs = lrc_net.input, outputs = lrc_net.layers[-7].output)
        cnn_net.save(os.path.join(logger.path,test+'.h5'))
        #GUARDAR MODELO DE RED CONVOLUCIONAL + LRC
        lrc_net.save(os.path.join(logger.path,test+'_LRC.h5'))
        #EVALUAR MODELO
        #GENERACION OA - Overall Accuracy 
        test_loss, OA = lrc_net.evaluate(datosPrueba, etiquetasPrueba)
        print('LOGISTIC OA:'+str(OA))
        #GENERAR MAPA FINAL DE CLASIFICACIÓN
        #GENERACION AA - Average Accuracy 
        AA = np.zeros(groundTruth.max()+1)
        for j in range(1,groundTruth.max()+1):                                     #QUITAR 1 para incluir datos del fondo
            datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,j)   #MUESTRAS DE UNA CLASE                 
            test_loss, AA[j] = lrc_net.evaluate(datosClase, etiquetasClase)        #EVALUAR MODELO PARA LA CLASE
        #GENERACION Kappa Coefficient
        datosSalida = lrc_net.predict(datosPrueba)
        datosSalida = datosSalida.argmax(axis=1)
        etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
        kappa = cohen_kappa_score(etiquetasPrueba, datosSalida)
        #LOGGER DATOS DE VALIDACIÓN
        logger.savedataPerformance(OA, AA, kappa)
        logger.close()
        
        ################################################################################
        ################Generar caracteristicicas con los datos de entrada##############
        features_tr = cnn_net.predict(datosEntrenamiento)
        features_test = cnn_net.predict(datosPrueba)
        ################################################################################
        ################################################################################

        ################################## CLASIFICADOR RIEMANN ############################################
        datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)
        #CREAR FICHERO DATA LOGGER 
        logger = DataLogger(fileName = dataSet+'_RIE', folder = test, save = True) 
        #Crear y entrenar clasificador 
        print('RIEMANNIAN CLASSIFIER #####################################################################')
        rie_net = self.riemann_classifier(features_tr, etiquetasEntrenamiento, method='tan')
        #Guarda el clasificador entrenado
        joblib.dump(rie_net, os.path.join(logger.path,test+'_RIE.pkl')) 
        #EVALUAR MODELO
        #GENERACION OA - Overall Accuracy 
        features_testRIE = self.reshapeFeaturesRIE(features_test)
        datosSalida = rie_net.predict(features_testRIE)
        OA = self.accuracy(etiquetasPrueba, datosSalida)
        print('RIEMANNIAN OA:'+str(OA))
        #GENERACION AA - Average Accuracy 
        AA = np.zeros(groundTruth.max()+1)
        for j in range(1,groundTruth.max()+1):                          
            datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,j)       #MUESTRAS DE UNA CLASE                 
            features_class = cnn_net.predict(datosClase)
            features_class = self.reshapeFeaturesRIE(features_class)                   # SOLO PARA RIEMMANIAN CLASSIFIER
            claseSalida = rie_net.predict(features_class)
            AA[j] = self.accuracy(etiquetasClase, claseSalida)                         #EVALUAR MODELO PARA LA CLASE
        #GENERACION Kappa Coefficient
        etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
        kappa = cohen_kappa_score(etiquetasPrueba, datosSalida)
        #LOGGER DATOS DE VALIDACIÓN
        logger.savedataPerformance(OA, AA, kappa)
        logger.close()
        ####################################################################################################

        ################################## CLASIFICADOR SVM #############################################################
        datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)
        #CREAR FICHERO DATA LOGGER 
        logger = DataLogger(fileName = dataSet+'_SVM', folder = test, save = True) 
        #Crear y entrenar clasificador 
        print('SVM CLASSIFIER ##########################################################################')
        svm_net = self.svm_classifier(features_tr, etiquetasEntrenamiento, kernel='linear')
        #Guarda el clasificador entrenado
        joblib.dump(svm_net, os.path.join(logger.path,test+'_SVM.pkl')) 
        #EVALUAR MODELO
        #GENERACION OA - Overall Accuracy 
        features_testSVM = self.reshapeFeaturesSVM(features_test)
        datosSalida = svm_net.predict(features_testSVM)
        OA = self.accuracy(etiquetasPrueba, datosSalida)
        print('SVM OA:'+str(OA))
        #GENERACION AA - Average Accuracy 
        AA = np.zeros(groundTruth.max()+1)
        for j in range(1,groundTruth.max()+1):                          
            datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,j)       #MUESTRAS DE UNA CLASE                 
            features_class = cnn_net.predict(datosClase)
            features_class = self.reshapeFeaturesSVM(features_class)                   # SOLO PARA RIEMMANIAN CLASSIFIER
            claseSalida = svm_net.predict(features_class)
            AA[j] = self.accuracy(etiquetasClase, claseSalida)                         #EVALUAR MODELO PARA LA CLASE
        #GENERACION Kappa Coefficient
        etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
        kappa = cohen_kappa_score(etiquetasPrueba, datosSalida)
        #LOGGER DATOS DE VALIDACIÓN
        logger.savedataPerformance(OA, AA, kappa)
        path = logger.path
        logger.close()
        #################################################################################################################
        #TERMINAR PROCESOS DE KERAS
        K.clear_session()
        #RETORNAR ETIQUETAS DE PRUEBA Y PREDICCIÓN
        class_names = []
        for i in range(1,groundTruth.max()+1):
            class_names.append('Class '+str(i))
        if etiquetasPrueba.ndim>1:
            etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
        if datosSalida.ndim>1:
            datosSalida = datosSalida.argmax(axis=1)

        return etiquetasPrueba, datosSalida, class_names, path
        
    def trainInception(self, dataSet, test, imagenFE, groundTruth):
        pass

    def trainScae(self, dataSet, test, imagenFE, groundTruth):
        pass

    def trainBcae(self, dataSet, test, imagenFE, groundTruth):
        pass

    ## SAVE and LOAD SKLEARN
    ## now you can save it to a file
    #joblib.dump(clf, 'filename.pkl') 
    ## and later you can load it
    #clf = joblib.load('filename.pkl')