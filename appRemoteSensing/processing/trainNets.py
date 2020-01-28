#Entrenamiento de diferentes arquitecturas de DNN: CNN2D 
# pylint: disable=E1136  # pylint/issues/3139
import warnings
warnings.filterwarnings('ignore')
from .cargarHsi import CargarHsi
from .prepararDatos import PrepararDatos
from .PCA import princiapalComponentAnalysis
from .MorphologicalProfiles import morphologicalProfiles
from .dataLogger import DataLogger
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, concatenate
from keras.layers import UpSampling2D, Dropout, Conv2DTranspose, MaxPooling2D, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import regularizers
from keras import backend as K 
from keras import layers
from keras.optimizers import SGD
from keras import initializers
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
    
    def euclidean_distance_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    def refinement_layer(self, lateral_tensor, vertical_tensor, num_filters, l2_loss):
        conv1 = Conv2D(num_filters, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(lateral_tensor)
        bchn1 = BatchNormalization()(conv1)
        conv2 = Conv2D(num_filters, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(bchn1)
        conv3 = Conv2D(num_filters, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(vertical_tensor)
        added = Add()([conv2, conv3])
        up = UpSampling2D(size=(2, 2), interpolation='nearest')(added)
        refinement = BatchNormalization()(up)
        return refinement

    def cae(self, N , input_tensor, input_layer,nb_bands, l2_loss):
        encoded1_bn = BatchNormalization()(input_layer)
        encoded1 = Conv2D(N, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded1_bn)
        encoded1_dg = MaxPooling2D((3,3), strides=2, padding='same')(encoded1)
        
        encoded2_bn = BatchNormalization()(encoded1_dg)
        encoded2 = Conv2D(2*N, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded2_bn)
        encoded2_dg = MaxPooling2D((3,3), strides=2, padding='same')(encoded2)
        
        encoded3_bn = BatchNormalization()(encoded2_dg)
        encoded3 = Conv2D(2*N, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded3_bn)
        encoded3_dg = MaxPooling2D((3,3), strides=2, padding='same')(encoded3)
        
        encoded4_bn = BatchNormalization()(encoded3_dg)
        encoded4 = Conv2D(4*N, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded4_bn)
        
        refinement3 = self.refinement_layer(encoded3_dg, encoded4, 2*N, l2_loss)
        refinement2 = self.refinement_layer(encoded2_dg, refinement3, 2*N, l2_loss)
        refinement1 = self.refinement_layer(encoded1_dg, refinement2, N, l2_loss)
        
        output_tensor = Conv2D(nb_bands, (1, 1), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(refinement1)
        autoencoder = Model(input_tensor, output_tensor)
        return autoencoder
    
    def refinement_bae(self, lateral_tensor, vertical_tensor, kernel_size, num_filters, l2_loss):
        conv1 = Conv2D(num_filters, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(lateral_tensor)
        bchn1 = BatchNormalization()(conv1)
        conv2 = Conv2D(num_filters, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(bchn1)
        conv3 = Conv2D(num_filters, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(vertical_tensor)
        added = Add()([conv2, conv3])
        up = UpSampling2D(size=(2, 2), interpolation='nearest')(added)
        refinement = BatchNormalization()(up)
        return refinement

    def bae(self, N , input_tensor, nb_bands, kernel_size, l2_loss):
        encoded1_bn = BatchNormalization()(input_tensor)
        encoded1 = Conv2D(N, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded1_bn)
        encoded1_dg = MaxPooling2D((2,2), strides=2, padding='same')(encoded1)
        
        encoded2_bn = BatchNormalization()(encoded1_dg)
        encoded2 = Conv2D(2*N, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded2_bn)
        encoded2_dg = MaxPooling2D((2,2), strides=2, padding='same')(encoded2)
        
        encoded3_bn = BatchNormalization()(encoded2_dg)
        encoded3 = Conv2D(2*N, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded3_bn)
        encoded3_dg = MaxPooling2D((2,2), strides=2, padding='same')(encoded3)
        
        encoded4_bn = BatchNormalization()(encoded3_dg)
        encoded4 = Conv2D(4*N, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded4_bn)
        
        refinement3 = self.refinement_bae(encoded3_dg, encoded4, kernel_size, 2*N, l2_loss)
        refinement2 = self.refinement_bae(encoded2_dg, refinement3, kernel_size, 2*N, l2_loss)
        refinement1 = self.refinement_bae(encoded1_dg, refinement2, kernel_size, N, l2_loss)
        
        output_tensor = Conv2D(nb_bands, (1, 1), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(refinement1)
        autoencoder = Model(input_tensor, output_tensor)
        return autoencoder

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

    def trainCNN2d(self, dataSet, test, clss, imagenFE, groundTruth):
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
        if clss == 'LRC':
            datosSalidaSelecc = datosSalida
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
        if clss == 'RIE':
            datosSalidaSelecc = datosSalida
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
        if clss == 'SVM':
            datosSalidaSelecc = datosSalida
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
        if datosSalidaSelecc.ndim>1:
            datosSalidaSelecc = datosSalidaSelecc.argmax(axis=1)

        #########################GENERAR MAPA FINAL DE CLASIFICACIÓN####################
        imagenSalida = preparar.predictionToImage(datosSalidaSelecc)
        ##############################IMAGEN DE COMPARACION#############################
        imgCompare = np.absolute(groundTruth-imagenSalida)
        imgCompare = (imgCompare > 0) * 1 
        np.save('ClassificationMap', imagenSalida) #Genera archivo con los datos de Salida
        np.save('imageCompare', imgCompare) #Genera archivo con los datos de Salida
        return imagenSalida, imgCompare, etiquetasPrueba, datosSalidaSelecc, class_names, path
        
    def trainInception(self, dataSet, test, clss, imagenFE, groundTruth):
        epochs = 35
        #CREAR FICHERO DATA LOGGER 
        logger = DataLogger(fileName = dataSet+'_LRC', folder = test, save = True)      
        #PREPARAR DATOS PARA ENTRENAMIENTO
        ventana = 9  #VENTANA 2D de PROCESAMIENTO
        preparar = PrepararDatos(imagenFE, groundTruth, False)
        datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
        datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

        #DEFINICION RED CONVOLUCIONAL INCEPTION 
        input_tensor = Input(shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3]))
        # Cada rama tiene el mismo estado de padding='same', lo cual es necesario para mantener todas las salidas de las ramas 
        # en el mismo tamaño. Esto posibilita la ejecución de la instrucción concatenate.
        # Rama A
        branch_a = layers.Conv2D(64, (1,1), activation='relu', padding='same')(input_tensor)
        # Rama B
        branch_b = layers.Conv2D(64, (1,1), activation='relu', padding='same')(input_tensor)
        branch_b = layers.Conv2D(64, (3,3), activation='relu', padding='same')(branch_b)
        # Rama C
        branch_c = layers.AveragePooling2D((3,3), strides=(1,1), padding='same')(input_tensor)
        branch_c = layers.Conv2D(64, (3,3), activation='relu', padding='same')(branch_c)
        # Rama D
        branch_d = layers.Conv2D(64, (1,1), activation='relu', padding='same')(input_tensor)
        branch_d = layers.Conv2D(64, (3,3), activation='relu', padding='same')(branch_d)
        branch_d = layers.Conv2D(64, (3,3), activation='relu', padding='same')(branch_d)
        # Se concatenan todas las rama  para tener un solo modelo en output
        output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
        ################################## CLASIFICADOR LOGISTIC REGRESION ############################################
        # Se añade como capa final de salida un clasificador tipo Multinomial logistic regression
        output = Flatten()(output)
        output    = Dense(groundTruth.max()+1, activation='softmax')(output)
        # Se define el modelo total de la red 
        lrc_net = Model(inputs = input_tensor, outputs = output)
        #ENTRENAMIENTO DE LA RED CONVOLUCIONAL
        lrate = 0.01
        decay = lrate/epochs
        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        lrc_net.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print('LOGGISTIC REGRESION CLASSIFIER #####################################################################')
        history = lrc_net.fit(datosEntrenamiento,etiquetasEntrenamiento,epochs=epochs,batch_size=256,validation_data=(datosValidacion, etiquetasValidacion))
        #LOGGER DATOS DE ENTRENAMIENTO
        #logger.savedataTrain(history)
        #GUARDAR MODELO DE RED CONVOLUCIONAL
        cnn_net = Model(inputs = lrc_net.input, outputs = lrc_net.layers[-4].output)
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
        if clss == 'LRC':
            datosSalidaSelecc = datosSalida
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
        if clss == 'RIE':
            datosSalidaSelecc = datosSalida
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
        if clss == 'SVM':
            datosSalidaSelecc = datosSalida  
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
        if datosSalidaSelecc.ndim>1:
            datosSalidaSelecc = datosSalidaSelecc.argmax(axis=1)

        #########################GENERAR MAPA FINAL DE CLASIFICACIÓN####################
        imagenSalida = preparar.predictionToImage(datosSalidaSelecc)
        ##############################IMAGEN DE COMPARACION#############################
        imgCompare = np.absolute(groundTruth-imagenSalida)
        imgCompare = (imgCompare > 0) * 1 
        np.save('ClassificationMap', imagenSalida) #Genera archivo con los datos de Salida
        np.save('imageCompare', imgCompare) #Genera archivo con los datos de Salida
        return imagenSalida, imgCompare, etiquetasPrueba, datosSalidaSelecc, class_names, path

    def trainScae(self, dataSet, test, clss, imagenFE, groundTruth):
        epochs = 50
        #CREAR FICHERO DATA LOGGER 
        logger = DataLogger(fileName = dataSet+'_LRC', folder = test, save = True)      
        #PREPARAR DATOS PARA ENTRENAMIENTO
        ventana = 8  #VENTANA 2D de PROCESAMIENTO
        preparar = PrepararDatos(imagenFE, groundTruth, False)
        datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
        datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

        #DEFINICION STACKED CONVOLUTIONAL AUTOENCODER
        input_img = Input(shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3])) #tensor de entrada
        nd_scae = [64, 32, 16] #dimension de cada uno de los autoencoders
        #convolutional autoencoders
        len_i = 0
        encoder = input_img
        for j in range(len(nd_scae)):
            autoencoder = self.cae(nd_scae[j] , input_img, encoder, datosEntrenamiento.shape[3], 0.01)
            #congelar capas ya entrenadas
            if j != 0:
                for layer in autoencoder.layers[1:len_i +1]:
                    layer.trainable = False
            print(autoencoder.summary())
            autoencoder.compile(optimizer='rmsprop', loss=self.euclidean_distance_loss, metrics=['accuracy']) #   loss='binary_crossentropy'
            autoencoder.fit(datosEntrenamiento, datosEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(datosValidacion, datosValidacion))
            #Quitar capa de salida del autoencoder j
            autoencoder.layers.pop()
            len_i = len(autoencoder.layers)
            encoder = autoencoder.layers[-1].output
        #######################MODELO STACKED AUTOENCODER##################################################################
        cnn_net = Model(inputs = autoencoder.input, outputs = autoencoder.layers[-1].output)
        cnn_net.save(os.path.join(logger.path,test+'.h5'))

        ################################################################################
        ################Generar caracteristicicas con los datos de entrada##############
        features_tr = cnn_net.predict(datosEntrenamiento)
        features_val = cnn_net.predict(datosValidacion)
        features_test = cnn_net.predict(datosPrueba)
        ################################################################################
        ################################################################################


        ################################## CLASIFICADOR LOGISTIC REGRESION ############################################
        input_features = Input(shape=(features_tr.shape[1],features_tr.shape[2],features_tr.shape[3]))
        fullconected = Flatten()(input_features)
        fullconected = Dense(128, activation = 'relu')(fullconected) 
        fullconected = Dense(groundTruth.max()+1, activation = 'softmax')(fullconected) 
        lrc_net = Model(inputs = input_features, outputs = fullconected) #GENERA EL MODELO FINAL

        lrc_net.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   
        history = lrc_net.fit(features_tr, etiquetasEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(features_val, etiquetasValidacion))

        #GUARDAR MODELO CALSIFICADOR LRC
        lrc_net.save(os.path.join(logger.path,test+'_LRC.h5'))
        #EVALUAR MODELO
        #GENERACION OA - Overall Accuracy 
        test_loss, OA = lrc_net.evaluate(features_test, etiquetasPrueba)
        print('LOGISTIC OA:'+str(OA))
        #GENERAR MAPA FINAL DE CLASIFICACIÓN
        #GENERACION AA - Average Accuracy 
        AA = np.zeros(groundTruth.max()+1)
        for j in range(1,groundTruth.max()+1):                                     #QUITAR 1 para incluir datos del fondo
            datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,j)       #MUESTRAS DE UNA CLASE    
            features_class = cnn_net.predict(datosClase)             
            test_loss, AA[j] = lrc_net.evaluate(features_class, etiquetasClase)        #EVALUAR MODELO PARA LA CLASE
        #GENERACION Kappa Coefficient
        datosSalida = lrc_net.predict(features_test)
        datosSalida = datosSalida.argmax(axis=1)
        if clss == 'LRC':
            datosSalidaSelecc = datosSalida
        etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
        kappa = cohen_kappa_score(etiquetasPrueba, datosSalida)
        #LOGGER DATOS DE VALIDACIÓN
        logger.savedataPerformance(OA, AA, kappa)
        logger.close()
        
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
        if clss == 'RIE':
            datosSalidaSelecc = datosSalida
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
        if clss == 'SVM':
            datosSalidaSelecc = datosSalida  
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
        if datosSalidaSelecc.ndim>1:
            datosSalidaSelecc = datosSalidaSelecc.argmax(axis=1)

        #########################GENERAR MAPA FINAL DE CLASIFICACIÓN####################
        imagenSalida = preparar.predictionToImage(datosSalidaSelecc)
        ##############################IMAGEN DE COMPARACION#############################
        imgCompare = np.absolute(groundTruth-imagenSalida)
        imgCompare = (imgCompare > 0) * 1 
        np.save('ClassificationMap', imagenSalida) #Genera archivo con los datos de Salida
        np.save('imageCompare', imgCompare) #Genera archivo con los datos de Salida
        return imagenSalida, imgCompare, etiquetasPrueba, datosSalidaSelecc, class_names, path

    def trainBcae(self, dataSet, test, clss, imagenFE, groundTruth):
        epochs = 50
        #CREAR FICHERO DATA LOGGER 
        logger = DataLogger(fileName = dataSet+'_LRC', folder = test, save = True)      
        #PREPARAR DATOS PARA ENTRENAMIENTO
        ventana = 8  #VENTANA 2D de PROCESAMIENTO
        preparar = PrepararDatos(imagenFE, groundTruth, False)
        datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
        datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

        #DEFINICION BRANCHED CONVOLUTIONAL AUTOENCODER
        input_img = Input(shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3])) #tensor de entrada
        nd_bcae = [1, 3, 5, 7] #kernel de cada uno de los autoencoders
        #convolutional autoencoders
        branches = []
        for j in range(len(nd_bcae)): 
            autoencoder = self.bae(16, input_img, datosEntrenamiento.shape[3], nd_bcae[j], 0.01)
            print(autoencoder.summary())
            autoencoder.compile(optimizer='rmsprop', loss=self.euclidean_distance_loss, metrics=['accuracy']) #   loss='binary_crossentropy'
            autoencoder.fit(datosEntrenamiento, datosEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(datosValidacion, datosValidacion))
            #Quitar capa de salida del autoencoder j
            autoencoder.layers.pop()
            branches.append( autoencoder.layers[-1].output ) 
        output = concatenate(branches, axis=-1)
        
        ################################### BRANCHES OF STACKED AUTOENCODERS ##########################################################################################
        cnn_net = Model(inputs = input_img, outputs = output)
        cnn_net.save(os.path.join(logger.path,test+'.h5'))

        ################################################################################
        ################Generar caracteristicicas con los datos de entrada##############
        features_tr = cnn_net.predict(datosEntrenamiento)
        features_val = cnn_net.predict(datosValidacion)
        features_test = cnn_net.predict(datosPrueba)
        ################################################################################
        ################################################################################

        ################################## CLASIFICADOR LOGISTIC REGRESION ############################################
        input_features = Input(shape=(features_tr.shape[1],features_tr.shape[2],features_tr.shape[3]))
        fullconected = Flatten()(input_features)
        fullconected = Dense(128, activation = 'relu')(fullconected) 
        fullconected = Dense(groundTruth.max()+1, activation = 'softmax')(fullconected) 
        lrc_net = Model(inputs = input_features, outputs = fullconected) #GENERA EL MODELO FINAL

        print(lrc_net.summary())
        lrc_net.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   
        history = lrc_net.fit(features_tr, etiquetasEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(features_val, etiquetasValidacion))

        #GUARDAR MODELO CALSIFICADOR LRC
        lrc_net.save(os.path.join(logger.path,test+'_LRC.h5'))
        #EVALUAR MODELO
        #GENERACION OA - Overall Accuracy 
        test_loss, OA = lrc_net.evaluate(features_test, etiquetasPrueba)
        print('LOGISTIC OA:'+str(OA))
        #GENERAR MAPA FINAL DE CLASIFICACIÓN
        #GENERACION AA - Average Accuracy 
        AA = np.zeros(groundTruth.max()+1)
        for j in range(1,groundTruth.max()+1):                                     #QUITAR 1 para incluir datos del fondo
            datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,j)       #MUESTRAS DE UNA CLASE    
            features_class = cnn_net.predict(datosClase)             
            test_loss, AA[j] = lrc_net.evaluate(features_class, etiquetasClase)        #EVALUAR MODELO PARA LA CLASE
        #GENERACION Kappa Coefficient
        datosSalida = lrc_net.predict(features_test)
        datosSalida = datosSalida.argmax(axis=1)
        if clss == 'LRC':
            datosSalidaSelecc = datosSalida
        etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
        kappa = cohen_kappa_score(etiquetasPrueba, datosSalida)
        #LOGGER DATOS DE VALIDACIÓN
        logger.savedataPerformance(OA, AA, kappa)
        logger.close()
        
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
        if clss == 'RIE':
            datosSalidaSelecc = datosSalida
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
        if clss == 'SVM':
            datosSalidaSelecc = datosSalida  
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
        if datosSalidaSelecc.ndim>1:
            datosSalidaSelecc = datosSalidaSelecc.argmax(axis=1)

        #########################GENERAR MAPA FINAL DE CLASIFICACIÓN####################
        imagenSalida = preparar.predictionToImage(datosSalidaSelecc)
        ##############################IMAGEN DE COMPARACION#############################
        imgCompare = np.absolute(groundTruth-imagenSalida)
        imgCompare = (imgCompare > 0) * 1 
        np.save('ClassificationMap', imagenSalida) #Genera archivo con los datos de Salida
        np.save('imageCompare', imgCompare) #Genera archivo con los datos de Salida
        return imagenSalida, imgCompare, etiquetasPrueba, datosSalidaSelecc, class_names, path

    ## SAVE and LOAD SKLEARN
    ## now you can save it to a file
    #joblib.dump(clf, 'filename.pkl') 
    ## and later you can load it
    #clf = joblib.load('filename.pkl')