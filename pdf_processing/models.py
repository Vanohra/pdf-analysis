from django.db import models

# Create your models here.

class UploadedPDF(models.Model):
    title = models.CharField(max_length=255)  # PDF title (name)
    pdf_file = models.FileField(upload_to='pdfs/')  # Uploads PDFs into 'pdfs/' folder
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Auto timestamp when uploaded
    extracted_text = models.TextField(blank=True, null=True)  # Store extracted text (optional for now)

    def __str__(self):
        return self.title
