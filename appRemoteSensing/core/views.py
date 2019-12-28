from django.views.generic.base import TemplateView
from django.shortcuts import render

# Create your views here.
class IndexPageView(TemplateView):
    template_name = "core/index.html"


