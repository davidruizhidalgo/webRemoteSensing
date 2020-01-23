from django.urls import path
from .views import IndexPageView,  IndexPageViewUS

urlpatterns = [
	path('', IndexPageView.as_view(), name="index"),
	path('us/', IndexPageViewUS.as_view(), name="index_us"),
]
