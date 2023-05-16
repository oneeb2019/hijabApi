from django.urls import path
from API import views

urlpatterns = [
    path('Image_processing', views.image_api, name='image_api'),
]
