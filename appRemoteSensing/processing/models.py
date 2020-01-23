from django.db import models

# Create your models here.
class DataSet(models.Model):
    DIMENSION_CHOISES=(
        ('NON','None'),
        ('PCA','Principal Component Analysis'),
        ('EAP','Attibute Profiles'),
        ('EEP','Extintion Profiles'),
    )
    FEATURES_CHOISES=(
        ('CNN','Convolutional Neural Network'),
        ('INC','Inception Network'),
        ('SCA','Stacked Autoencoder CNN'),
        ('BCA','Branched Autoencoder CNN'),
    )
    CLASSIFIER_CHOISES=(
        ('LRC','Logistic Regression'),
        ('SVM','Support Vector Machine'),
        ('RIE','Riemannian Classifier'),
    )
    
    name = models.CharField(max_length=200, verbose_name="Nombre", unique = True)
    description = models.TextField(verbose_name="Descripción", default='')
    description_US = models.TextField(verbose_name="Descripción_US", default='')
    image = models.ImageField(verbose_name="Imagen", upload_to="dataSets") 
    
    dimension = models.CharField(max_length=10, verbose_name="Metodo de Reducción Dimensional", choices= DIMENSION_CHOISES,    default='')
    features = models.CharField(max_length=10, verbose_name="Metodo de Extracción de Caracteristicas", choices= FEATURES_CHOISES, default='')
    classifier = models.CharField(max_length=10, verbose_name="Metodo de Selección de Información", choices= CLASSIFIER_CHOISES, default='')
    
    created = models.DateTimeField(auto_now_add=True,verbose_name="Fecha de Creación")
    updated = models.DateTimeField(auto_now=True,verbose_name="Fecha de Modificación")

    class Meta:
        verbose_name = 'Imagen HSI'
        verbose_name_plural =  'Imagenes HSI'
        ordering = ["created"]

    def __str__(self):
        return self.name