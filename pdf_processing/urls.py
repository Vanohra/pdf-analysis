from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('upload_success/<int:pdf_id>/', views.upload_success, name='upload_success'),
     path('ask/', views.ask_question, name='ask_question'),  # New endpoint for Q&A
]
