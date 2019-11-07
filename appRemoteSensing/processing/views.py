from django.shortcuts import render
from .models import DataSet

# Create your views here.
def newProcess(request):
	datasets = DataSet.objects.all()
	return render(request,"processing/newProcess.html",{'datasets':datasets})