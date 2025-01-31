from openai import OpenAI  # Correct import for OpenAI's latest version
import re
import fitz  # PyMuPDF for text extraction
import pytesseract  # Tesseract OCR for images
from pdf2image import convert_from_path  # Convert PDF pages to images
import camelot  # Extract tables from PDFs
from django.db import models
import json  # To store embeddings as JSON
import os  # For environment variables
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UploadedPDF(models.Model):
    title = models.CharField(max_length=255)  # PDF title (name)
    pdf_file = models.FileField(upload_to='pdfs/')  # Uploads PDFs into 'pdfs/' folder
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Auto timestamp when uploaded
    extracted_text = models.TextField(blank=True, null=True)  # Store extracted text
    embeddings = models.JSONField(blank=True, null=True)  # Store embeddings as JSON

    def __str__(self):
        return self.title

    def clean_text(self, text):
        """Cleans extracted text while preserving meaningful line breaks."""
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
        text = re.sub(r'[ \t]+', ' ', text)  # Remove extra spaces but preserve structure
        text = re.sub(r'[^a-zA-Z0-9.,!?()\[\]{}\-\n]', ' ', text)  # Remove unwanted special characters
        text = re.sub(r'(?<=\w)-\n(?=\w)', '', text)  # Fix hyphenated words
        text = re.sub(r'(\d+)\.(\s+)', r'\n\1. ', text)  # Fix numbered lists
        text = re.sub(r'(\*|\•|\-|\•)\s*', r'\n- ', text)  # Fix bullet points
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Ensure proper paragraph spacing
        return text.strip()

    def extract_text(self):
        """Extracts text from the PDF, including images (OCR) and tables."""
        pdf_path = self.pdf_file.path  # Get full path of uploaded PDF
        extracted_content = ""

        # Extract normal text
        with fitz.open(pdf_path) as doc:
            for page in doc:
                extracted_content += page.get_text("text") + "\n"  # Add extra newlines to separate paragraphs

        # Extract text from images (OCR)
        images = convert_from_path(pdf_path)  # Convert PDF pages to images
        for image in images:
            extracted_content += "\n[Extracted from Image]\n"
            extracted_content += pytesseract.image_to_string(image) + "\n"

        # Extract tables (if present)
        try:
            tables = camelot.read_pdf(pdf_path, pages="all")
            for i, table in enumerate(tables):
                extracted_content += f"\n[Extracted Table {i+1}]\n"
                extracted_content += table.df.to_string() + "\n"
        except:
            extracted_content += "\n[No Tables Found]\n"

        # Clean extracted text before saving
        cleaned_text = self.clean_text(extracted_content)

        # Save cleaned text into the database
        self.extracted_text = cleaned_text
        self.save()

        return cleaned_text  # Return cleaned extracted text
    
    def chunk_text(self, text, chunk_size=200):
        """Splits text into smaller chunks for better retrieval."""
        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
    
    def generate_embeddings(self):
        if not self.extracted_text:
            print("No extracted text, skipping embeddings...")
            return None  # No text to embed

        # ✅ Correct way to set the OpenAI API key
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        chunks = self.chunk_text(self.extracted_text)  # Break text into small parts
        embeddings_list = []

        try:
            for chunk in chunks:
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=chunk
                )
                embeddings_list.append({
                    "text": chunk,
                    "embedding": response.data[0].embedding
                })

            # Store as JSON
            self.embeddings = json.dumps(embeddings_list)
            self.save()
            print("✅ Embeddings Generated Successfully!")

        except Exception as e:
            print("❌ Error generating embeddings:", e)



    def search_relevant_text(self, query):
        """Finds the most relevant text chunks for a given query using stored embeddings."""
        
        if not self.embeddings:
            return "No relevant information found."  # No embeddings stored

        try:
            # Load stored embeddings
            pdf_embeddings = json.loads(self.embeddings)

            if not isinstance(pdf_embeddings, list):  # Ensure it's a list
                return "Error: Invalid embedding format."

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Generate query embedding
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_embedding = np.array(response.data[0].embedding).reshape(1, -1)

            best_chunks = []
            similarities = []

            # Loop through stored embeddings
            for embedding_dict in pdf_embeddings:
                chunk_text = embedding_dict["text"]
                chunk_embedding = np.array(embedding_dict["embedding"]).reshape(1, -1)

                # Compute similarity
                similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]

                if similarity > 0.5:  # Lowered threshold to get more context
                    best_chunks.append(chunk_text)
                    similarities.append(similarity)

            # Sort best matches
            top_chunks = [text for _, text in sorted(zip(similarities, best_chunks), reverse=True)[:3]]

            if not top_chunks:
                return "No relevant information found."

            return " ".join(top_chunks)  # Return multiple chunks for context

        except Exception as e:
            print(f"❌ Error in search_relevant_text: {e}")
            return "Error processing query."
    