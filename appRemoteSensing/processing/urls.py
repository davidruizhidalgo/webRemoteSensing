from django.urls import path
from .views import DatasetListView, DatasetDetailView, DatasetUpdate, DatasetUpdateFE, DatasetUpdateCL
from .views import  plot_groundtruth, plot_classification, plot_imgCompare
from .views import ConstructionPageView, ConstructionPageViewUS
from .views import DatasetListViewUS, DatasetDetailViewUS,DatasetUpdateUS, DatasetUpdateFEUS, DatasetUpdateCLUS

urlpatterns = [
    path('<str:parametro>', DatasetListView.as_view(), name="processing"),
    path('us/<str:parametro>', DatasetListViewUS.as_view(), name="processing_us"),
 
    path('upload/', ConstructionPageView.as_view(), name="upload"),
    path('us/upload/', ConstructionPageViewUS.as_view(), name="upload_us"),
    
    path('<str:parametro>/<int:pk>/<slug:slug>/', DatasetDetailView.as_view(), name="img_processing"),
    path('us/<str:parametro>/<int:pk>/<slug:slug>/', DatasetDetailViewUS.as_view(), name="img_processing_us"),
    
    path('<str:parametro>/update/<int:pk>/', DatasetUpdate.as_view(), name='update'),
    path('us/<str:parametro>/update/<int:pk>/', DatasetUpdateUS.as_view(), name='update_us'),
    
    path('<str:parametro>/updateFE/<int:pk>/', DatasetUpdateFE.as_view(), name='updateFE'),
    path('us/<str:parametro>/updateFE/<int:pk>/', DatasetUpdateFEUS.as_view(), name='updateFE_us'),

    path('<str:parametro>/updateCls/<int:pk>/', DatasetUpdateCL.as_view(), name='updateCL'),
    path('us/<str:parametro>/updateCls/<int:pk>/', DatasetUpdateCLUS.as_view(), name='updateCL_us'),
    
    path('plot_groundtruth/', plot_groundtruth, name='plot_groundtruth'),
    path('plot_classification/', plot_classification, name='plot_classification'),
    path('plot_imgCompare/', plot_imgCompare, name='plot_imgCompare'),

]
