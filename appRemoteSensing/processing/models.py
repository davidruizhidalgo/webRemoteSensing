from django.db import models

# Create your models here.
class DataSet(models.Model):
    DIMENSION_CHOISES=(
        ('NON','Ninguno'),
        ('PCA','PCA'),
        ('EAP','EAP'),
        ('EEP','EEP'),
    )
    FEATURES_CHOISES=(
        ('CNN','Red Convolucional'),
        ('INC','Red Inception'),
        ('SCA','Autoencoder Convolucional Apilado'),
        ('BCA','Autoencoder Convolucional Ramificado'),
    )
    CLASSIFIER_CHOISES=(
        ('LRC','Logistic Regression'),
        ('SVM','Maquina de Sporte Vectorial'),
        ('RIE','Riemannian Classifier'),
    )
    
    name = models.CharField(max_length=200, verbose_name="Nombre", unique = True)
    description = models.TextField(verbose_name="Descripción")
    image = models.ImageField(verbose_name="Imagen", upload_to="dataSets") 
    
    dimension = models.CharField(max_length=10, verbose_name="Reducción Dimensional", choices= DIMENSION_CHOISES,    default='')
    features = models.CharField(max_length=10, verbose_name="Extracción de Caracteristicas", choices= FEATURES_CHOISES, default='')
    classifier = models.CharField(max_length=10, verbose_name="Selección de Información", choices= CLASSIFIER_CHOISES, default='')
    
    created = models.DateTimeField(auto_now_add=True,verbose_name="Fecha de Creación")
    updated = models.DateTimeField(auto_now=True,verbose_name="Fecha de Modificación")

    class Meta:
        verbose_name = 'Imagen HSI'
        verbose_name_plural =  'Imagenes HSI'
        ordering = ["created"]

    def __str__(self):
        return self.name