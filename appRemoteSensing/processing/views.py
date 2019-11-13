from django.views.generic.list import ListView
from django.views.generic.detail import DetailView
from django.views.generic.edit import UpdateView
from django.urls import reverse, reverse_lazy
from django.template.defaultfilters import slugify

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

def plot_gt(request):
	# Recuperamos los datos para representar en el gráfico
	img_gt = np.load('groundTruth.npy')
	# Creamos una figura y le dibujamos el gráfico
	f = plt.figure()
	plt.imshow(img_gt)
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

# Create your views here.
class DatasetListView(ListView):
    model = DataSet
	
class DatasetDetailView(DetailView):
	model = DataSet
	#REALIZAR PROCESAMIENTO
	#RETORNAR VARIABLES DE CONTEXTO A LA VISTA
	def get_context_data(self, **kwargs):
		context = super().get_context_data(**kwargs)
		object = super().get_object()
		#procesos
		process = Generate(object)
		data = process.execution()
				
		#Enviar datos a variables de contexto. OA, AA, Kappa, training logger
		context['data'] = data
		return context

class DatasetUpdate(UpdateView):
	model = DataSet
	form_class = ProcessingForm
	template_name_suffix = '_update_form'

	def get_success_url(self):
		return reverse_lazy('updateFE', args=[self.object.id]) + '?ok'

class DatasetUpdateFE(UpdateView):
	model = DataSet
	form_class = ProcessingForm
	template_name_suffix = '_update_formFE'

	def get_success_url(self):
		return reverse_lazy('updateCL', args=[self.object.id]) + '?ok'

class DatasetUpdateCL(UpdateView):
	model = DataSet
	form_class = ProcessingForm
	template_name_suffix = '_update_formCL'

	def get_success_url(self):
		return reverse_lazy('img_processing', args=[self.object.id, slugify(self.object.name)]) + '?ok'
