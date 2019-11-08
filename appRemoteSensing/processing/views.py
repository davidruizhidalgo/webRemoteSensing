from django.views.generic.list import ListView
from django.views.generic.detail import DetailView
from django.views.generic.edit import UpdateView
from django.urls import reverse, reverse_lazy
from django.template.defaultfilters import slugify

from .models import DataSet
from .forms import ProcessingForm

# Create your views here.
class DatasetListView(ListView):
    model = DataSet
	
class DatasetDetailView(DetailView):
    model = DataSet
	#REALIZAR PROCESAMIENTO #

	#RETORNAR VARIABLES DE CONTEXTO A LA VISTA
	#def get_context_data(self, **kwargs):
    #    context = super().get_context_data(**kwargs)
	#	#procesos
	#	return context

class DatasetUpdate(UpdateView):
	model = DataSet
	form_class = ProcessingForm
	template_name_suffix = '_update_form'

	def get_success_url(self):
		return reverse_lazy('img_processing', args=[self.object.id, slugify(self.object.name)]) + '?ok'
