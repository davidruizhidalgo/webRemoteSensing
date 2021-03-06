from django.views.generic.list import ListView
from django.views.generic.detail import DetailView
from django.views.generic.edit import UpdateView
from django.views.generic.base import TemplateView
from django.urls import reverse, reverse_lazy
from django.template.defaultfilters import slugify
from django.shortcuts import render

from .models import DataSet
from .forms import ProcessingForm

from .generate import Generate

import io
import matplotlib.pyplot as plt
import numpy as np

from django.http import HttpResponse
from django.shortcuts import render
from matplotlib.backends.backend_agg import FigureCanvasAgg
from random import sample

from threading import RLock
verrou = RLock()

import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from scipy.stats import binom

# Function for plotting confusion matrix using Plotly
def plotlyConfusionMatrix(true_labels, pred_labels, class_names): 
  cm = confusion_matrix(true_labels, pred_labels).astype(float)
  sum_hor = cm.sum(axis=1)
  
  accuracy = 0
  for i in range(len(sum_hor)):
    accuracy = accuracy + cm[i,i]
  accuracy = 100.0*accuracy/sum_hor.sum()
  
  for i in range(len(sum_hor)):
    cm[i,:] = (100.0/sum_hor[i])*cm[i,:]
  cm = np.flip(cm,axis=0)
  
  alpha = 0.05
  n = len(true_labels)
  c = len(class_names)
  chancelevel = (100.0/n)*binom.ppf(1.0-alpha, n, 1.0/c)
  
  y=[name+' ('+str(val)+')' for name,val in zip(class_names[::-1],sum_hor)]
  data=go.Heatmap(
                  z=cm.tolist(),
                  x=class_names,
                  y=y,
                  colorscale='Rainbow'
       )
	   #- Chance level = '+str(chancelevel)+' %'
  layout = go.Layout(
                  title='Confusion Matrix (z in %) - Overall Accuracy = '+
                        str(accuracy)+' % ',

                  xaxis=dict(
                          title='Predicted Labels'
                  ),
                  yaxis=dict(
                          title='True Labels',
                  )
          )
  fig = go.Figure(data=data, layout=layout)
  return fig

#  VISTAS DE LA APP
def plot_groundtruth(request):
	with verrou:
		# Recuperamos los datos para representar en el gráfico
		img = np.load('groundTruth.npy')
		# Creamos una figura y le dibujamos el gráfico
		f = plt.figure()
		plt.imshow(img)
		plt.title('Ground Truth')

		# Como enviaremos la imagen en bytes la guardaremos en un buffer
		buf = io.BytesIO()
		canvas = FigureCanvasAgg(f)
		canvas.print_png(buf)

		# Creamos la respuesta enviando los bytes en tipo imagen png
		response = HttpResponse(buf.getvalue(), content_type='image/png')
		# Limpiamos la figura para liberar memoria
		f.clear()

		# Añadimos la cabecera de longitud de fichero para más estabilidad
		response['Content-Length'] = str(len(response.content))
		# Devolvemos la response
		return response

def plot_classification(request):
	with verrou:
		# Recuperamos los datos para representar en el gráfico
		img = np.load('ClassificationMap.npy')
		# Creamos una figura y le dibujamos el gráfico
		f = plt.figure()
		plt.imshow(img)
		plt.title('Classification Map')

		# Como enviaremos la imagen en bytes la guardaremos en un buffer
		buf = io.BytesIO()
		canvas = FigureCanvasAgg(f)
		canvas.print_png(buf)

		# Creamos la respuesta enviando los bytes en tipo imagen png
		response = HttpResponse(buf.getvalue(), content_type='image/png')
		# Limpiamos la figura para liberar memoria
		f.clear()
		# Añadimos la cabecera de longitud de fichero para más estabilidad
		response['Content-Length'] = str(len(response.content))
		# Devolvemos la response
		return response

def plot_imgCompare(request):
	with verrou:
		# Recuperamos los datos para representar en el gráfico
		img = np.load('imageCompare.npy')
		# Creamos una figura y le dibujamos el gráfico
		f = plt.figure()
		plt.imshow(img, cmap='Greys',  interpolation='nearest')
		plt.title('Image Comparation')

		# Como enviaremos la imagen en bytes la guardaremos en un buffer
		buf = io.BytesIO()
		canvas = FigureCanvasAgg(f)
		canvas.print_png(buf)

		# Creamos la respuesta enviando los bytes en tipo imagen png
		response = HttpResponse(buf.getvalue(), content_type='image/png')
		# Limpiamos la figura para liberar memoria
		f.clear()
		# Añadimos la cabecera de longitud de fichero para más estabilidad
		response['Content-Length'] = str(len(response.content))
		# Devolvemos la response
		return response


############################################################################################################################
################################# VISTAS EN ESPAÑOL#########################################################################
############################################################################################################################
class DatasetListView(ListView):
	model = DataSet
	def get_context_data(self, **kwargs):
		context = super().get_context_data(**kwargs)
		parametro = self.kwargs.get('parametro', None) 
		context['Data'] = parametro
		return context

class DatasetDetailView(DetailView):
	model = DataSet
	#REALIZAR PROCESAMIENTO
	#RETORNAR VARIABLES DE CONTEXTO A LA VISTA
	def get_context_data(self, **kwargs):
		context = super().get_context_data(**kwargs)
		object = super().get_object()
		parametro = self.kwargs.get('parametro', None)

		if parametro == 'Ejecutar Modelo':
			training = False
		else:
			training = True
		#procesos
		process = Generate(object)
		imagenGT, imagenSalida, imgCompare, data, true_labels_test, pred_labels_test, class_names = process.execution(train= training)	
		OA_data = data[0]*100
		AA_data = data[1,0:]*100
		kappa_data = data[2]
		metadata = [len(class_names), AA_data[1:].mean()]
		#Enviar datos a variables de contexto. OA, AA, Kappa, training logger
		fig = plotlyConfusionMatrix(true_labels=true_labels_test, pred_labels=pred_labels_test, class_names=class_names)
		confusionData = plot(fig, output_type='div', include_plotlyjs=False)
		
		fig1 = px.imshow(imagenGT, color_continuous_scale='Viridis')
		fig1.update_layout(title={'text': "Salida Deseada",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, coloraxis_showscale=False)
		groundTruth = plot(fig1, output_type='div', include_plotlyjs=False)

		fig2 = px.imshow(imagenSalida, color_continuous_scale='Viridis')
		fig2.update_layout(title={'text': "Imagen de Salida",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, coloraxis_showscale=False)
		imagenOutput = plot(fig2, output_type='div', include_plotlyjs=False)

		fig3 = px.imshow(imgCompare, color_continuous_scale='Viridis')
		fig3.update_layout(title={'text': "Imagen Comparativa",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},coloraxis_showscale=False)
		imgComp = plot(fig3, output_type='div', include_plotlyjs=False)
		
		#Determiar si el modelo nececita ser entrenado
		#Determiar si el modelo nececita ser entrenado
		warningMSM = ''
		if np.count_nonzero(imagenSalida) == 0:	
			warningMSM = '- El modelo necesita ser entrenado'

		context['OA_data'] = [ '%.2f' % elem for elem in OA_data]
		context['AA_data'] = [ '%.2f' % elem for elem in AA_data]
		context['kappa_data'] = [ '%.2f' % elem for elem in kappa_data]
		context['class_names'] = class_names
		context['confusionData'] = confusionData
		context['groundTruth'] = groundTruth
		context['imagenOutput'] = imagenOutput
		context['imgComp'] = imgComp
		context['metadata'] = [ '%.2f' % elem for elem in metadata]
		context['Parametro'] = parametro
		context['warningMSM'] = warningMSM
		return context

class DatasetUpdate(UpdateView):
	model = DataSet
	form_class = ProcessingForm
	template_name_suffix = '_update_form'

	def get_success_url(self, **kwargs):
		context = super().get_context_data(**kwargs)
		parametro = self.kwargs.get('parametro', None)
		return reverse_lazy('updateFE', args=[parametro, self.object.id]) + '?ok'

class DatasetUpdateFE(UpdateView):
	model = DataSet
	form_class = ProcessingForm
	template_name_suffix = '_update_formFE'

	def get_success_url(self, **kwargs):
		context = super().get_context_data(**kwargs)
		parametro = self.kwargs.get('parametro', None)
		return reverse_lazy('updateCL', args=[parametro, self.object.id]) + '?ok'

class DatasetUpdateCL(UpdateView):
	model = DataSet
	form_class = ProcessingForm
	template_name_suffix = '_update_formCL'

	def get_success_url(self, **kwargs):
		context = super().get_context_data(**kwargs)
		parametro = self.kwargs.get('parametro', None)
		return reverse_lazy('img_processing', args=[parametro, self.object.id, slugify(self.object.name)]) + '?ok'

class ConstructionPageView(TemplateView):
    template_name = "processing/coming_soon.html"


############################################################################################################################
################################# VIEWS IN ENGLISH #########################################################################
############################################################################################################################
class DatasetListViewUS(ListView):
	template_name_suffix = '_list_us'
	model = DataSet
	def get_context_data(self, **kwargs):
		context = super().get_context_data(**kwargs)
		parametro = self.kwargs.get('parametro', None) 
		context['Data'] = parametro
		return context

class DatasetDetailViewUS(DetailView):
	template_name_suffix = '_detail_us'
	model = DataSet
	#REALIZAR PROCESAMIENTO
	#RETORNAR VARIABLES DE CONTEXTO A LA VISTA
	def get_context_data(self, **kwargs):
		context = super().get_context_data(**kwargs)
		object = super().get_object()
		parametro = self.kwargs.get('parametro', None)

		if parametro == 'Execute Model':
			training = False
		else:
			training = True
		#procesos
		process = Generate(object)
		imagenGT, imagenSalida, imgCompare, data, true_labels_test, pred_labels_test, class_names = process.execution(train= training)	
		OA_data = data[0]*100
		AA_data = data[1,0:]*100
		kappa_data = data[2]
		metadata = [len(class_names), AA_data[1:].mean()]
		#Enviar datos a variables de contexto. OA, AA, Kappa, training logger
		fig = plotlyConfusionMatrix(true_labels=true_labels_test, pred_labels=pred_labels_test, class_names=class_names)
		confusionData = plot(fig, output_type='div', include_plotlyjs=False)
		
		fig1 = px.imshow(imagenGT, color_continuous_scale='Viridis')
		fig1.update_layout(title={'text': "Ground Truth",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},coloraxis_showscale=False)
		groundTruth = plot(fig1, output_type='div', include_plotlyjs=False)

		fig2 = px.imshow(imagenSalida, color_continuous_scale='Viridis')
		fig2.update_layout(title={'text': "Image Output",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},coloraxis_showscale=False)
		imagenOutput = plot(fig2, output_type='div', include_plotlyjs=False)

		fig3 = px.imshow(imgCompare, color_continuous_scale='Viridis')
		fig3.update_layout(title={'text': "Image Comparation",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},coloraxis_showscale=False)
		imgComp = plot(fig3, output_type='div', include_plotlyjs=False)

		warningMSM = ''
		if np.count_nonzero(imagenSalida) == 0:	
			warningMSM = ' - The model needs to be training'
		
		context['OA_data'] = [ '%.2f' % elem for elem in OA_data]
		context['AA_data'] = [ '%.2f' % elem for elem in AA_data]
		context['kappa_data'] = [ '%.2f' % elem for elem in kappa_data]
		context['class_names'] = class_names
		context['confusionData'] = confusionData
		context['groundTruth'] = groundTruth
		context['imagenOutput'] = imagenOutput
		context['imgComp'] = imgComp
		context['metadata'] = [ '%.2f' % elem for elem in metadata]
		context['Parametro'] = parametro
		context['warningMSM'] = warningMSM
		return context

class DatasetUpdateUS(UpdateView):
	model = DataSet
	form_class = ProcessingForm
	template_name_suffix = '_update_form_us'

	def get_success_url(self, **kwargs):
		context = super().get_context_data(**kwargs)
		parametro = self.kwargs.get('parametro', None)
		return reverse_lazy('updateFE_us', args=[parametro, self.object.id]) + '?ok'

class DatasetUpdateFEUS(UpdateView):
	model = DataSet
	form_class = ProcessingForm
	template_name_suffix = '_update_formFE_us'

	def get_success_url(self, **kwargs):
		context = super().get_context_data(**kwargs)
		parametro = self.kwargs.get('parametro', None)
		return reverse_lazy('updateCL_us', args=[parametro, self.object.id]) + '?ok'

class DatasetUpdateCLUS(UpdateView):
	model = DataSet
	form_class = ProcessingForm
	template_name_suffix = '_update_formCL_us'

	def get_success_url(self, **kwargs):
		context = super().get_context_data(**kwargs)
		parametro = self.kwargs.get('parametro', None)
		return reverse_lazy('img_processing_us', args=[parametro, self.object.id, slugify(self.object.name)]) + '?ok'

class ConstructionPageViewUS(TemplateView):
    template_name = "processing/coming_soon_us.html"