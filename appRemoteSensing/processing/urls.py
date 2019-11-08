from django.urls import path
from .views import DatasetListView, DatasetDetailView, DatasetUpdate

urlpatterns = [
    path('', DatasetListView.as_view(), name="processing"),
    path('<int:pk>/<slug:slug>/', DatasetDetailView.as_view(), name="img_processing"),
    path('update/<int:pk>/', DatasetUpdate.as_view(), name='update'),
]