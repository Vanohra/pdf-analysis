from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('upload/success/', views.upload_success, name='upload_success'),
]
