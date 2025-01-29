from django.shortcuts import render, redirect
from .forms import PDFUploadForm
from .models import UploadedPDF

def upload_pdf(request):
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('upload_success')  # Redirect after successful upload
    else:
        form = PDFUploadForm()

    return render(request, 'pdf_processing/upload.html', {'form': form})

def upload_success(request):
    return render(request, 'pdf_processing/upload_success.html')
