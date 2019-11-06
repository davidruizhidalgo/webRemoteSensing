from django.db import models

# Create your models here.
class DataSet(models.Model):
    name = models.CharField(max_length=200,verbose_name="Nombre")
    description = models.TextField(verbose_name="Descripción")
    image = models.ImageField(verbose_name="Imagen") #,upload_to="projects"
    created = models.DateTimeField(auto_now_add=True,verbose_name="Fecha de Creación")
    updated = models.DateTimeField(auto_now=True,verbose_name="Fecha de Modificación")
