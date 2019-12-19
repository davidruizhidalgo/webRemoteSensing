from django.urls import path
from .views import DatasetListView, DatasetDetailView, DatasetUpdate, DatasetUpdateFE, DatasetUpdateCL
from .views import  plot_groundtruth, plot_classification, plot_imgCompare

urlpatterns = [
    path('', DatasetListView.as_view(), name="processing"),
    path('<int:pk>/<slug:slug>/', DatasetDetailView.as_view(), name="img_processing"),
    path('update/<int:pk>/', DatasetUpdate.as_view(), name='update'),
    path('updateFE/<int:pk>/', DatasetUpdateFE.as_view(), name='updateFE'),
    path('updateCls/<int:pk>/', DatasetUpdateCL.as_view(), name='updateCL'),
    path('plot_groundtruth/', plot_groundtruth, name='plot_groundtruth'),
    path('plot_classification/', plot_classification, name='plot_classification'),
    path('plot_imgCompare/', plot_imgCompare, name='plot_imgCompare'),

]