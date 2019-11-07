# Generated by Django 2.2.7 on 2019-11-07 14:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('processing', '0003_auto_20191106_1615'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='dataset',
            options={'ordering': ['created'], 'verbose_name': 'Imagen HSI', 'verbose_name_plural': 'Imagenes HSI'},
        ),
        migrations.AddField(
            model_name='dataset',
            name='classifier',
            field=models.CharField(choices=[('LRC', 'Logistic Regression'), ('SVM', 'Maquina de Sporte Vectorial'), ('RIE', 'Riemmanian Classifier')], default='', max_length=100, verbose_name='Extracción Información'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='dimension',
            field=models.CharField(choices=[('NON', 'Ninguno'), ('PCA', 'PCA'), ('EAP', 'EAP'), ('EEP', 'EEP')], default='', max_length=100, verbose_name='Reducción Dimensional'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='features',
            field=models.CharField(choices=[('CNN', 'Red Convolucional'), ('INC', 'Red Inception'), ('SCA', 'Autoencoder Convolucional Apilado'), ('BCA', 'Autoencoder Convolucional Ramificado')], default='', max_length=100, verbose_name='Extracción Caracteristicas'),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='name',
            field=models.CharField(max_length=200, unique=True, verbose_name='Nombre'),
        ),
    ]
