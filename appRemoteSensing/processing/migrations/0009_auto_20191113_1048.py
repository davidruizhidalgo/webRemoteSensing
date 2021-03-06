# Generated by Django 2.2.7 on 2019-11-13 15:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('processing', '0008_auto_20191109_1001'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dataset',
            name='dimension',
            field=models.CharField(choices=[('NON', 'Ninguno'), ('PCA', 'Componentes Principales'), ('EAP', 'Attibute Profiles'), ('EEP', 'Extintion Profiles')], default='', max_length=10, verbose_name='Metodo de Reducción Dimensional'),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='features',
            field=models.CharField(choices=[('CNN', 'Red Convolucional'), ('INC', 'Red Inception'), ('SCA', 'Stacked Autoencoder CNN'), ('BCA', 'Branched Autoencoder CNN')], default='', max_length=10, verbose_name='Metodo de Extracción de Caracteristicas'),
        ),
    ]
