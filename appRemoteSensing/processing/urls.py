from django.urls import path
from . import views as processing_views

urlpatterns = [
    path('processing/',processing_views.newProcess,name="processing"),
]