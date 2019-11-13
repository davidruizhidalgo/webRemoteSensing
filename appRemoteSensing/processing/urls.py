from django.urls import path
from .views import DatasetListView, DatasetDetailView, DatasetUpdate, DatasetUpdateFE, DatasetUpdateCL
from .views import  plot_gt

urlpatterns = [
    path('', DatasetListView.as_view(), name="processing"),
    path('<int:pk>/<slug:slug>/', DatasetDetailView.as_view(), name="img_processing"),
    path('update/<int:pk>/', DatasetUpdate.as_view(), name='update'),
    path('updateFE/<int:pk>/', DatasetUpdateFE.as_view(), name='updateFE'),
    path('updateCls/<int:pk>/', DatasetUpdateCL.as_view(), name='updateCL'),
    path('plot_gt/', plot_gt, name='plot_gt'),

]