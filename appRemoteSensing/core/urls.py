from django.urls import path
from . import views as core_views

urlpatterns = [
	path('',core_views.home,name="home"),
	path('about/',core_views.about,name="about"),
	path('contact/',core_views.contact,name="contact"),
]