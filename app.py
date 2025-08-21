import os
import json
import tempfile
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
import PyPDF2
import numpy as np
from fastembed import TextEmbedding
import faiss
from typing import List, Dict, Tuple
import gc
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import io
import logging

app = Flask(__name__)
CORS(app)

# Allow large PDF uploads (up to 100 MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_SUMMARY_LENGTH = 4000
BATCH_SIZE = 8

# Globals
pdf_text = ""
chunks: List[str] = []
faiss_index = None
chunk_embeddings = None
_embedding_model: TextEmbedding | None = None

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def get_embedding_model() -> TextEmbedding:
	global _embedding_model
	if _embedding_model is None:
		cache_dir = os.environ.get("FASTEMBED_CACHE_PATH", "models")
		os.makedirs(cache_dir, exist_ok=True)
		logging.info("Loading embedding model BAAI/bge-small-en-v1.5 into cache_dir=%s", cache_dir)
		_embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=512, cache_dir=cache_dir)
		logging.info("Embedding model ready")
	return _embedding_model


def extract_text_from_pdf(pdf_file) -> str:
	try:
		pdf_reader = PyPDF2.PdfReader(pdf_file)
		logging.info("PDF loaded with %d pages", len(pdf_reader.pages))
		text = ""
		for i in range(0, len(pdf_reader.pages), BATCH_SIZE):
			for page in pdf_reader.pages[i:i + BATCH_SIZE]:
				pt = page.extract_text()
				if pt:
					text += pt + "\n"
			gc.collect()
		logging.info("Extracted %d characters of text", len(text))
		return text.strip()
	except Exception as e:
		logging.exception("Error extracting PDF text: %s", e)
		return ""


def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
	if len(text) <= chunk_size:
		return [text]
	res: List[str] = []
	start = 0
	while start < len(text):
		end = start + chunk_size
		if end < len(text):
			for i in range(end, max(start + chunk_size - 100, start), -1):
				if text[i] in '.!?':
					end = i + 1
					break
		chunk = text[start:end].strip()
		if chunk:
			res.append(chunk)
		start = end - overlap
		if start >= len(text):
			break
	logging.info("Created %d chunks (chunk_size=%d, overlap=%d)", len(res), chunk_size, overlap)
	return res


def generate_embeddings(texts: List[str]) -> np.ndarray:
	try:
		model = get_embedding_model()
		n = len(texts)
		logging.info("Embedding %d chunks with batch_size=%d", n, BATCH_SIZE)
		start_total = time.time()
		vecs: List[np.ndarray] = []
		batch_num = 0
		for i in range(0, n, BATCH_SIZE):
			batch = texts[i:i + BATCH_SIZE]
			batch_num += 1
			b_start = time.time()
			emb = list(model.embed(batch))
			vecs.append(np.array(emb, dtype=np.float32))
			b_end = time.time()
			logging.info("Embedded batch %d/%d (%d items) in %.2fs", batch_num, (n + BATCH_SIZE - 1)//BATCH_SIZE, len(batch), b_end - b_start)
			gc.collect()
		arr = np.vstack(vecs) if vecs else np.zeros((0, 384), dtype=np.float32)
		logging.info("Embedding complete in %.2fs; shape=%s", time.time() - start_total, arr.shape if arr.size else (0,))
		return arr
	except Exception as e:
		logging.exception("Error generating embeddings: %s", e)
		return np.zeros((0, 384), dtype=np.float32)


def create_faiss_index(embeddings: np.ndarray):
	try:
		if embeddings.size == 0:
			return None
		faiss.normalize_L2(embeddings)
		idx = faiss.IndexFlatIP(embeddings.shape[1])
		idx.add(embeddings.astype('float32'))
		logging.info("FAISS index built (dim=%d, n=%d)", embeddings.shape[1], embeddings.shape[0])
		return idx
	except Exception as e:
		logging.exception("Error creating FAISS index: %s", e)
		return None


def search_similar_chunks(query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
	global faiss_index, chunks
	if faiss_index is None:
		return []
	try:
		q_vec = np.array(list(get_embedding_model().embed([query])), dtype=np.float32)
		faiss.normalize_L2(q_vec)
		scores, indices = faiss_index.search(q_vec, top_k)
		res: List[Tuple[int, float, str]] = []
		for score, idx in zip(scores[0], indices[0]):
			if 0 <= idx < len(chunks):
				res.append((int(idx), float(score), chunks[idx]))
		return res
	except Exception as e:
		print(f"Error searching chunks: {e}")
		return []


def calculate_sentence_importance(sentence: str) -> float:
	if not sentence.strip():
		return 0.0
	length_score = min(len(sentence.split()) / 20.0, 1.0)
	words = sentence.lower().split()
	unique_words = len(set(words))
	keyword_score = min(unique_words / len(words), 1.0) if words else 0
	indicators = ['key', 'important', 'significant', 'major', 'primary', 'essential', 'critical', 'crucial']
	indicator_score = sum(1 for w in indicators if w in sentence.lower()) / len(indicators)
	return (length_score * 0.3 + keyword_score * 0.3 + indicator_score * 0.4)


def generate_summary_mapreduce() -> str:
	global chunks
	if not chunks:
		return "No document loaded for summarization."
	try:
		logging.info("Summarization started for %d chunks", len(chunks))
		chunk_summaries: List[str] = []
		for i, chunk in enumerate(chunks):
			sents = chunk.split('. ')
			if len(sents) > 2:
				scored: List[Tuple[str, float]] = []
				for j, s in enumerate(sents):
					if len(s.split()) > 3:
						score = calculate_sentence_importance(s)
						if j == 0 or j == len(sents) - 1:
							score *= 1.2
						scored.append((s, score))
				scored.sort(key=lambda x: x[1], reverse=True)
				part = '. '.join([s for s, _ in scored[:3]]) + '.'
			else:
				part = chunk
			chunk_summaries.append(part)
			if (i + 1) % 20 == 0:
				gc.collect()
		combined = ' '.join(chunk_summaries)
		logging.info("Map phase complete; combined length=%d", len(combined))
		sents = combined.split('. ')
		if len(sents) > 30:
			scored = [(s, calculate_sentence_importance(s)) for s in sents if len(s.split()) > 5]
			scored.sort(key=lambda x: x[1], reverse=True)
			final = '. '.join([s for s, _ in scored[:30]]) + '.'
		else:
			final = combined
		if len(final) > MAX_SUMMARY_LENGTH:
			final = final[:MAX_SUMMARY_LENGTH] + '...'
		logging.info("Reduce phase complete; summary length=%d", len(final))
		return final
	except Exception as e:
		logging.exception("Error in enhanced MapReduce summarization: %s", e)
		return f"Error generating summary: {str(e)}"


def create_summary_pdf(summary: str) -> bytes:
	try:
		buffer = io.BytesIO()
		doc = SimpleDocTemplate(buffer, pagesize=letter)
		story = []
		styles = getSampleStyleSheet()
		title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=30, alignment=TA_CENTER)
		body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontSize=11, spaceAfter=12, alignment=TA_JUSTIFY, leading=16)
		story.append(Paragraph("Document Summary", title_style))
		story.append(Spacer(1, 20))
		for para in summary.split('. '):
			p = para.strip()
			if not p:
				continue
			if not p.endswith('.'):
				p += '.'
			story.append(Paragraph(p, body_style))
			story.append(Spacer(1, 8))
		doc.build(story)
		pdf_content = buffer.getvalue()
		buffer.close()
		return pdf_content
	except Exception as e:
		print(f"Error creating PDF: {e}")
		return None


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
	global pdf_text, chunks, faiss_index, chunk_embeddings
	try:
		if 'file' not in request.files:
			return jsonify({'error': 'No file provided'}), 400
		file = request.files['file']
		if file.filename == '':
			return jsonify({'error': 'No file selected'}), 400
		if not file.filename.lower().endswith('.pdf'):
			return jsonify({'error': 'File must be a PDF'}), 400
		pdf_text = extract_text_from_pdf(file)
		if not pdf_text:
			return jsonify({'error': 'Could not extract text from PDF'}), 400
		chunks = create_chunks(pdf_text)
		chunk_embeddings = generate_embeddings(chunks)
		if chunk_embeddings.size == 0:
			return jsonify({'error': 'Failed to generate embeddings'}), 400
		faiss_index = create_faiss_index(chunk_embeddings)
		if faiss_index is None:
			return jsonify({'error': 'Failed to create search index'}), 400
		gc.collect()
		return jsonify({'message': 'PDF processed successfully','chunks': len(chunks),'text_length': len(pdf_text)})
	except Exception as e:
		print(f"Error processing PDF: {e}")
		return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500


@app.route('/chat', methods=['POST'])
def chat():
	try:
		data = request.get_json()
		query = data.get('message', '').strip()
		if not query:
			return jsonify({'error': 'No message provided'}), 400
		if not chunks or faiss_index is None:
			return jsonify({'error': 'No document loaded. Please upload a PDF first.'}), 400
		similar_chunks = search_similar_chunks(query, top_k=3)
		if not similar_chunks:
			return jsonify({'response': "I couldn't find relevant information in the document for your question."})
		context = "\n\n".join([chunk for _, _, chunk in similar_chunks])
		response = f"Based on the document, here's what I found:\n\n{context[:1000]}..."
		if len(context) > 1000:
			response += "\n\n(Response truncated for readability)"
		return jsonify({'response': response})
	except Exception as e:
		print(f"Error in chat: {e}")
		return jsonify({'error': f'Error processing chat: {str(e)}'}), 500


@app.route('/summarize', methods=['POST'])
def summarize():
	try:
		if not chunks:
			return jsonify({'error': 'No document loaded. Please upload a PDF first.'}), 400
		start_time = time.time()
		summary = generate_summary_mapreduce()
		end_time = time.time()
		return jsonify({'summary': summary,'processing_time': f"{end_time - start_time:.2f} seconds",'summary_length': len(summary),'target_length': MAX_SUMMARY_LENGTH})
	except Exception as e:
		print(f"Error in summarization: {e}")
		return jsonify({'error': f'Error generating summary: {str(e)}'}), 500


@app.route('/download-summary', methods=['POST'])
def download_summary():
	try:
		if not chunks:
			return jsonify({'error': 'No document loaded. Please upload a PDF first.'}), 400
		summary = generate_summary_mapreduce()
		pdf_content = create_summary_pdf(summary)
		if pdf_content is None:
			return jsonify({'error': 'Failed to create PDF'}), 500
		buffer = io.BytesIO(pdf_content)
		buffer.seek(0)
		return send_file(buffer, as_attachment=True, download_name="document_summary.pdf", mimetype="application/pdf")
	except Exception as e:
		print(f"Error in PDF download: {e}")
		return jsonify({'error': f'Error creating PDF: {str(e)}'}), 500


@app.route('/status')
def status():
	return jsonify({'document_loaded': len(chunks) > 0,'chunks_count': len(chunks),'faiss_index_ready': faiss_index is not None,'text_length': len(pdf_text) if pdf_text else 0})


@app.route('/health')
def health():
	return jsonify({'status': 'healthy', 'message': 'Business Optima PDF Analysis API is running'})


if __name__ == '__main__':
	print("Starting Business Optima PDF Analysis Application (Local Summarization)...")
	# Use 7860 by default (Hugging Face Spaces convention). Override via $PORT locally if needed.
	port = int(os.environ.get('PORT', 7860))
	app.run(host='0.0.0.0', port=port)
