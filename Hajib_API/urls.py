from django.urls import path
from API import views

urlpatterns = [
    path('Image_processing', views.image_api, name='image_api'),
    path('v2/Image_processing', views.image_api_v2, name='image_api'),
]
