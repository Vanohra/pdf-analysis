from django.shortcuts import render, redirect
from .forms import PDFUploadForm
from .models import UploadedPDF
from django.http import Http404  # Handle missing PDF errors
import openai
import os

# ✅ Use OpenAI's new API client format
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def upload_pdf(request):
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_pdf = form.save()  # Save the PDF file to the database

            try:
                extracted_text = uploaded_pdf.extract_text()  # Extract and clean text from the PDF

                # Only generate embeddings if text extraction was successful
                if extracted_text.strip():  
                    uploaded_pdf.generate_embeddings()  # Generate embeddings
                
            except Exception as e:
                print(f"Error during text extraction or embedding: {e}")  # Log errors but don't crash

            return redirect('upload_success', pdf_id=uploaded_pdf.id)  # Redirect to success page

    else:
        form = PDFUploadForm()

    return render(request, 'pdf_processing/upload.html', {'form': form})

def upload_success(request, pdf_id):
    try:
        uploaded_pdf = UploadedPDF.objects.get(id=pdf_id)  # Retrieve the uploaded PDF
    except UploadedPDF.DoesNotExist:
        raise Http404("PDF not found")  # Handle invalid PDF IDs

    return render(request, 'pdf_processing/upload_success.html', {'text': uploaded_pdf.extracted_text})


def ask_question(request):
    if request.method == "POST":
        question = request.POST.get("question")  # Get user input
        pdfs = UploadedPDF.objects.all()  # Get all PDFs

        best_answer = "No relevant answer found."

        # Search for relevant content in the PDFs
        for pdf in pdfs:
            answer = pdf.search_relevant_text(question)
            if answer != "No relevant information found.":
                best_answer = answer
                break  # Stop after finding the first relevant document

        return render(request, "pdf_processing/answer.html", {"question": question, "answer": best_answer})

    return render(request, "pdf_processing/ask.html")
