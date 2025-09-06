"""
main pipeline file

outline:
	0. utils functions to get files

		get_benchmark_sets
		default_extract_cnt
		read_all_documents_in_folder
	
	1. parse documents into chunks (my strategies & hyper-parameters)
		list[strs,tables] -> list[chunk]

		ChunkerWrapper

	2. index content of documents into list of indexes with metadata, embeddings
		list[chunk] -> list[index]

	4. merge relevant chunks & generate answer
		query: str + list[chunks] -> answer: str

	5. measure quality
"""


"""
0. utils function to work with benchmarks
"""

from pathlib import Path
import typing as tp

def get_benchmark_sets(
	# path to benchmark
	pt: Path
) -> tp.Dict[str, Path]:
	"""
	Returns a dictionary mapping benchmark set names to their corresponding pts
	"""
	sets: tp.Dict[str, Path] = {}
	
	# Iterate through all items in the benchmark directory
	for item in pt.iterdir():
		if item.is_dir():
			# Use the directory name as the key and the full path as the value
			sets[item.name] = item
	
	return sets


from pathlib import Path
import pdfplumber
from tqdm import tqdm

def default_extract_cnt(
	f_pt: Path,
	# is needed to show process bar of reading process of f_pt
	v: bool=False,
) -> str:
	"""
	Reads a .txt or .pdf file and returns its content as a string.
	For PDFs, extracts text (including tables) using pdfplumber.
	"""
	if not f_pt.exists():
		raise FileNotFoundError(f"File not found: {f_pt}")

	if f_pt.suffix.lower() == ".txt":
		with open(f_pt, "r", encoding="utf-8") as f:
			return f.read()

	elif f_pt.suffix.lower() == ".pdf":
		cnt_str: str = ""
		# Using pdfplumber for tables
		with pdfplumber.open(f_pt) as pdf_f:
			for page in pdf_f.pages if not v else tqdm(pdf_f.pages, desc=f"reading {f_pt.name}"):
				cnt_str += page.extract_text() + "\n"
		return cnt_str.strip()

	else:
		raise ValueError("Unsupported file format. Only .txt and .pdf are supported.")
	

def read_all_documents_in_folder(
	# path to folder
	pt: Path,
	# extract function that extracts contents of files inside pt-folder
	# first argument - path to file, other arguments - extract_f_kwargs
	extract_f: tp.Callable[[Path, tp.Any], str],
	extract_f_kwargs: dict[str, tp.Any],
	# verbose
	v: bool=False,
) -> tp.Iterator[tuple[str, Path]]:
	"""
	Reads all .txt and .pdf files in a folder and returns their contents as strings.
	Shows current filename in progress bar.
	"""
	if not pt.exists() or not pt.is_dir():
		raise FileNotFoundError(f"Folder not found: {pt}")

	# Find all .txt and .pdf files in the folder
	fs = list(pt.glob("*.txt")) + list(pt.glob("*.pdf"))
	with tqdm(fs, desc="Processing files", disable=not v) as pbar:
		for f in pbar:
			try:
				pbar.set_description(f"Processing {f.name}")
				yield extract_f(f, **extract_f_kwargs), f
				pbar.set_description("Processing files")
				pbar.update(1)
			except Exception:
				continue



"""
1. chunker part of file

suggests wrapper for chunker for pipeline and metachunk type
"""

from dataclasses import dataclass
from langchain.schema import Document
from pydantic import BaseModel

# list of chunks with metadata
# chunk text & chunk metadata
MetaChunkTp = tuple[str, dict[str, tp.Any]]
# metachunk should have these neccessary fields to be used in pipeline:
# TODO:

class Chunk(BaseModel):
	text: str=""
	idx: str=""
	document_id: str=""
	score: float=0
	has_table: bool=False

	def to_qdrant_payload(self) -> dict[str, tp.Any]:
		return {
			"idx": self.idx,
			"document_id": self.document_id,
			"text": self.s,
		}
	
	def to_metadata(self) -> dict[str, tp.Any]:
		return {
			'idx': self.idx,
			'document_id': self.document_id,
			'has_table': self.has_table,
		}

	def with_vector(self, vector: list[float]) -> "ChunkWithVector":
		if isinstance(vector, np.ndarray):
			vector = vector.tolist()
		
		if isinstance(vector[0], np.ndarray):
			vector = [v.tolist() for v in vector]
		
		return ChunkWithVector(
			text=self.text,
			idx=self.idx,
			document_id=self.document_id,
			vector=vector,
		)

class ChunkerWrapper:
	def __init__(
		self,
		# chunker class to call it inside pipeline
		# chunker should return dict[str, str]
		chunker: tp.Any,
		# kwargs of pipeline
		chunker_kwargs: dict[str,tp.Any],
	):
		self.chunker = chunker(**chunker_kwargs)

	def process(
		self,
		text: list[Document],
		# function that launch chunker class with first arugment text
		# and chunker_kwargs - other arugments
		launch_f: tp.Callable[[str, tp.Any], list[Chunk]],
		chunker_kwargs: dict[str, tp.Any],
	) -> list[Chunk]:
		"""Split a document into chunks with metadata"""
		meta_chunks_for_documents: list[Chunk] = []
		for t in text:
			meta_chunks_for_documents.append(
				launch_f(self.chunker, t, chunker_kwargs)
			)
		
		processed_chunks: list[Chunk] = []
		for document_chunks in meta_chunks_for_documents:
			for i, chunk in enumerate(document_chunks):
				processed_chunks.append(
					Chunk(
						text=chunk.text,
						idx=chunk.idx,
						document_id=chunk.document_id,
						score=chunk.score,
						has_table=chunk.has_table,
					)
				)
		
		return processed_chunks





"""
2. Indexing part:

	1. get vectors of chunks -> create list[ChunkWithVector]
	2. create indeces of that chunks (we have already metachunk - texts with metadata)
		and we convert it into dicts with texts, vectors, metadata and store to qdrant
	3. rerank chunks with reranker class
"""

import openai
import numpy as np

class ChunkWithVector(Chunk):
	vector: tp.Union[list[list[float]], list[float]]

	def payload(self) -> dict:
		return self.model_dump(exclude={"vector"})

class EmbeddingClient:
	def __init__(
		self,
		# configure api-provider via openai
		base_url: str,
		api_key: str,
		model: str="embedding-model",
	):
		self.model = model
		self.client = openai.OpenAI(
			api_key=api_key,
			base_url=base_url,
		)

	def embed(
		self,
		# text to vectorize
		text: str
	) -> list[float]:
		resp = self.client.embeddings.create(
			model=self.model,
			input=text,
			encoding_format="float",
		)
		return resp.data[0].embedding


	def embed_batch(
		self,
		# texts to vectorize
		texts: list[str]
	) -> list[list[float]]:
		resp = self.client.embeddings.create(
			model=self.model,
			input=texts,
			encoding_format="float",
		)
		return [embedding.embedding for embedding in resp.data]

import faiss

class FAISSCollection:
	"""
	FAISS-based vector collection that mimics Qdrant's basic functionality
	"""
	
	def __init__(
		self,
		index_path: tp.Optional[str]=None,
		vector_size: int=4096,
		create_if_missing: bool=True,
		# "cosine", "l2", or "ip"
		metric_type: str="cosine"
	):
		self.vector_size = vector_size
		self.index_path = index_path
		self.metric_type = metric_type.lower()

		self.index = None
		# FAISS IDs -> chunk objects
		self.id_to_chunk = {}
		if index_path:
			try:
				self.index = faiss.read_index(index_path)
				assert self.index.d == vector_size, "Vector size mismatch"
			except:
				if create_if_missing:
						self._create_index()
				else:
						raise ValueError(f"Index not found at {index_path}")
		else:
			self._create_index()
	
	def _create_index(self):
		"""Create a new FAISS index"""
		if self.metric_type == "cosine":
			self.index = faiss.IndexFlatIP(self.vector_size)
		elif self.metric_type == "l2":
			self.index = faiss.IndexFlatL2(self.vector_size)
		elif self.metric_type == "ip":
			self.index = faiss.IndexFlatIP(self.vector_size)
		else:
			raise ValueError(f"Unsupported metric type: {self.metric_type}")
	
	def save(self, path: str):
		"""Save the FAISS index to disk"""
		faiss.write_index(self.index, path)
	
	def upload_chunks(self, chunks: list[Chunk]):
		"""
		Add chunks to the FAISS index
		"""
		vectors = np.stack([chunk.vector for chunk in chunks]).astype('float32')
		
		# Generate IDs for new vectors
		start_id = len(self.id_to_chunk)
		end_id = start_id + len(chunks)
		ids = np.arange(start_id, end_id)
		
		# Add to index
		self.index.add(vectors)
		
		# Store chunk metadata
		for idx, chunk in zip(ids, chunks):
			self.id_to_chunk[idx] = chunk
	
	def search_chunks(
		self,
		vector: np.ndarray,
		limit: int=10
	) -> list[Chunk]:
		"""
		Search for similar chunks
		"""
		if len(vector.shape) == 1:
			vector = vector.reshape(1, -1)
		vector = vector.astype('float32')
		distances, indices = self.index.search(vector, k=limit)
		
		results = []
		for i in range(len(indices[0])):
			idx = indices[0][i]
			chunk = self.id_to_chunk.get(idx)
			# print(f"c:{chunk}")
			if chunk:
				result_chunk = ChunkWithVector(
					text=chunk.text,
					idx=chunk.idx,
					document_id=chunk.document_id,
					vector=chunk.vector,
					score=float(distances[0][i])
				)
				results.append(result_chunk)
		
		return results
	
	def retrieve_chunks(
		self,
		document_id: str,
		chunk_idxs: list[str]=None,
	) -> list[Chunk]:
		"""
		Retrieve chunks by document ID and optional chunk indices
		"""
		results = []
		for chunk in self.id_to_chunk.values():
			if chunk.document_id == document_id:
				if chunk_idxs is None or chunk.idx in chunk_idxs:
					results.append(chunk)
		return results
	
	def __len__(self):
		return len(self.id_to_chunk)

"""
3. generation of answer
"""


from typing import List
import openai


SYSTEM_PROMPT = (
	"Ты — экспертная система Compressa RAG, "
	"предоставляющая точные и релевантные ответы на вопросы, "
	"используя только предоставленную контекстную информацию. "
	"Отвечай на русском языке. Ответ не должен содержать иероглифы."
)

DEFAULT_SUMMARY_PROMPT = (
	"# Контекстная информация:\n\n{context}\n\n"
	"---\n"
	"# Инструкции:\n\n"
	"1. Дай полный и краткий ответ на вопрос, используя только информацию из контекста.\n"
	"2. Укажи номер источника в квадратных скобках после использовании фактов из него, например: [1].\n"
	"3. Каждый ответ должен содержать хотя бы одну ссылку на источник.\n"
	"4. Ссылайся на источник только если информация взята из него.\n"
	"5. Если ответа на вопрос нет в источниках, ответь: \"{reject_answ}\".\n"
	"6. Не используй знания вне предоставленного контекста.\n\n"
	"7. Будь очень внимательна к условиям, которым твой ответ не должен противоречить. "
	"Пример: товар для доставки не должен попадать в определенную категорию.\n\n"
	"# Пример:\n\n"
	"**Контекстная информация:**\n"
	"Источник [1]: 'Документ о безопасности'\n"
	"Всегда носите защитные очки при работе с оборудованием.\n\n"
	"Источник [2]: 'Руководство пользователя'\n"
	"Перед началом работы проверьте оборудование на наличие повреждений.\n\n"
	"**Вопрос:** Нужно ли носить защитные очки при работе с оборудованием?\n\n"
	"**Ответ:** Да, при работе с оборудованием необходимо носить защитные очки [1].\n\n"
	"---\n"
	"# Вопрос:\n\n{question}\n"
)

def generate_response_from_chunks(
	client: tp.Any,
	model_name: str,
	q: str,
	chunks: List[ChunkWithVector], 
	# mapping chunk.document_id -> document-name
	docs: dict[str, str],
	max_tokens: int = 2048,
) -> str:
	"""
	Concatenates chunks' text and generates a response using the specified model.
	
	Args:
			chunks: List of ChunkWithVector objects containing text to process
			model: The model identifier to use
			system_prompt: The system prompt to guide the model's behavior
			max_tokens: Maximum number of tokens to generate
			
	Returns:
			The model's generated response as a string
	"""
	concatenated_text = "\n\n".join(
		[f"Источник: {docs[chunk.document_id]} " + chunk.text for chunk in chunks]
	)
	final_q: str = DEFAULT_SUMMARY_PROMPT.format(
		context=concatenated_text,
		reject_answ='Среди представленных документов нет ответа на вопрос',
		question=q
	)
	# print(f"f q:{final_q}")
	try:
		response = client.chat.completions.create(
			model=model_name,
			messages=[
					# {"role": "system", "content": q},
					{"role": "user", "content": final_q}
			],
			max_tokens=max_tokens
		)
		return response.choices[0].message.content
	except Exception as e:
		print(f"Error generating response: {e}")
		raise



"""
4. Evaluation of the model
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

class ScoreResponse(BaseModel):
	relevance: float = Field(ge=0, le=100)
	completeness: float = Field(ge=0, le=100)
	accuracy: float = Field(ge=0, le=100)
	comment: str


def parse_score_response(response_text: str) -> Optional[ScoreResponse]:
	try:
		res = ScoreResponse.model_validate_json(response_text)
		return res
	except ValidationError as e:
		print(response_text)
		return None


def get_score(
	client,
	model_nm: str,
	q: str,
	true_a: str,
	a: str
) -> Optional[ScoreResponse]:
	"""
	q - question from the csv, a - answer of the model, true_a - correct answer
	"""
	messages = [
		{
			"role": "system",
			"content": (
					"Strictly evaluate the answer against the ground truth. You need to provide most informative answer to compare different RAG systems"
					"Provide scores for relevance, completeness, accuracy (0 to 100), and a brief comment explaining deductions. "
					"Respond ONLY in JSON format: {\"relevance\": number, \"completeness\": number, \"accuracy\": number, \"comment\": string}. Don't need to wrap it with ```"
			)
		},
		{
			"role": "user",
			"content": f"Question: {q}\n\n RAG Answer to Score: {a}\n\nGround truth (to compare with): {true_a}"
		}
	]
	try:
		response = client.chat.completions.create(
			model=model_nm,
			messages=messages,
			extra_body={"guided_json": ScoreResponse.model_json_schema()},
			response_format={
					'type': 'json_object'
			},
		)
		response_text = response.choices[0].message.content
		# print(response_text)
		return parse_score_response(response_text)
	except Exception as e:
		print(e)
		return None
	

def score_column(client, model_nm: str, df: pd.DataFrame, col_nm: str="answer"):
	if col_nm not in df.columns:
		raise ValueError(f"Column {col_nm} not found in DataFrame")

	scores = []
	def process_row(row):
			result = get_score(client, model_nm, row['q'], row['true_a'], row[col_nm])
			if result:
					return result
			else:
					return ScoreResponse(relevance=0, completeness=0, accuracy=0, comment="ERROR")

	with ThreadPoolExecutor(max_workers=100) as executor:
		future_to_row = {executor.submit(process_row, row): idx for idx, row in df.iterrows()}
		for future in as_completed(future_to_row):
			idx = future_to_row[future]
			score = future.result()
			scores.append((idx, score))

		scores.sort()
		df[f'{col_nm}_relevance'] = [s.relevance for _, s in scores]
		df[f'{col_nm}_completeness'] = [s.completeness for _, s in scores]
		df[f'{col_nm}_accuracy'] = [s.accuracy for _, s in scores]
		df[f'{col_nm}_comment'] = [s.comment for _, s in scores]

	return df

    
def print_average_scores(df, column_to_score="answer"):
	column_to_score = column_to_score.replace(" ", "_")

	relevance_col = f"{column_to_score}_relevance"
	completeness_col = f"{column_to_score}_completeness"
	accuracy_col = f"{column_to_score}_accuracy"
	
	avg_relevance = df[relevance_col].mean()
	avg_completeness = df[completeness_col].mean()
	avg_accuracy = df[accuracy_col].mean()
	
	print(f"Average scores for '{column_to_score}':")
	print(f"  Relevance:    {avg_relevance:.2f}")
	print(f"  Completeness: {avg_completeness:.2f}")
	print(f"  Accuracy:     {avg_accuracy:.2f}")
	print(f"  Overall:      {(avg_relevance + avg_completeness + avg_accuracy) / 3:.2f}")
	print("--------------------------------")


def score_column_and_save(
	client,
	model_nm: str, 
	df,
	column_to_score="a",
):
	if column_to_score not in df.columns:
		print(f"Column {column_to_score} not found in DataFrame")
		return
	# columns = df.columns.tolist()
	df = score_column(
		client,
		model_nm,
		df,
		col_nm=column_to_score
	)
	return df


"""
Launch parallel
	- answering to the questions
	- scoring
"""

from concurrent.futures import (
	ThreadPoolExecutor,
	as_completed
)
import pandas as pd
import typing as tp
from tqdm import tqdm
import json
import numpy as np

def process_experiment(
	client,
	model_name: str,
	faiss_col,
	experiment_path: Path,
	cols: dict[str,str],
	emb_m: openai.OpenAI,
	n_requests: tp.Optional[int] = None,
	v: bool = False
):
	"""
	взять все вопросы из файла бенчмарка
	и сделать запрос к LLM, чтобы она ответила
	"""
	df = pd.read_csv(experiment_path / 'bench.csv')
	if not n_requests is None:
		df = df[:n_requests]
	idx2answers = {}
	def fetch_answer(idx):
		q = df.loc[idx, cols['q']]
		right_answer = df.loc[idx, cols['a']]
		try:
			q_emb = np.array(emb_m.embed_batch([q]))
			relevant_chunks = faiss_col.search_chunks(q_emb, limit=5)
			resp = generate_response_from_chunks(
				client=client,
				model_name=model_name,
				q=q,
				chunks=relevant_chunks,
				docs={},
			)
		except Exception as e:
			resp = ""
		if v:
			print(f"ВОПРОС: {q}\nОТВЕТ: {resp}\nПРАВИЛЬНЫЙ ОТВЕТ: {right_answer}\n")
			print("-" * 100)
		return idx, resp

	with ThreadPoolExecutor(max_workers=10) as executor:
		# параллельно постараться ответить на все вопросы из .csv
		future_to_idx = {executor.submit(fetch_answer, idx): idx for idx in df.index}
		with tqdm(total=len(future_to_idx), desc="Processing") as pbar:
			for future in as_completed(future_to_idx):
				idx, answer = future.result()
				idx2answers[idx] = answer
				pbar.update(1)

	with_answers_df = pd.DataFrame(columns=['q', 'a', 'true_a'])
	with_answers_df['a'] = df.index.map(idx2answers)
	with_answers_df['q'] = df[cols['q']]
	with_answers_df['true_a'] = df[cols['a']]
	pt_to_save = experiment_path / f'answers_{experiment_path.name}.csv'
	with_answers_df.to_csv(
		pt_to_save,
		index=False
	)
	return with_answers_df

from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

class ScoreResponse(BaseModel):
	relevance: float = Field(ge=0, le=100)
	completeness: float = Field(ge=0, le=100)
	accuracy: float = Field(ge=0, le=100)
	comment: str

def parse_score_response(
	response_text: str
) -> tp.Optional[ScoreResponse]:
	try:
		res = ScoreResponse.model_validate_json(response_text)
		return res
	except Exception as e:
		print(response_text)
		return None


def get_score(
	client,
	LLM_MODEL: str,
	q: str,
	true_a: str,
	a: str
) -> Optional[ScoreResponse]:
	"""
	q - question from the csv, a - answer of the model, true_a - correct answer
	"""
	messages = [
		{
			"role": "system",
			"content": (
					"Strictly evaluate the answer against the ground truth. You need to provide most informative answer to compare different RAG systems"
					"Provide scores for relevance, completeness, accuracy (0 to 100), and a brief comment explaining deductions. "
					"Respond ONLY in JSON format: {\"relevance\": number, \"completeness\": number, \"accuracy\": number, \"comment\": string}. Don't need to wrap it with ```"
			)
		},
		{
			"role": "user",
			"content": f"Question: {q}\n\n RAG Answer to Score: {a}\n\nGround truth (to compare with): {true_a}"
		}
	]
	try:
		response = client.chat.completions.create(
			model=LLM_MODEL,
			messages=messages,
			extra_body={"guided_json": ScoreResponse.model_json_schema()},
			response_format={
					'type': 'json_object'
			},
		)
		response_text = response.choices[0].message.content
		print(response_text)
		return parse_score_response(response_text)
	except Exception as e:
		print(e)
		return None
	

    
def average_scores(
	df,
	column_to_score="answer"
):
	column_to_score = column_to_score.replace(" ", "_")

	relevance_col = f"{column_to_score}_relevance"
	completeness_col = f"{column_to_score}_completeness"
	accuracy_col = f"{column_to_score}_accuracy"
	
	avg_relevance = df[relevance_col].mean()
	avg_completeness = df[completeness_col].mean()
	avg_accuracy = df[accuracy_col].mean()

	return {
		'Average scores for': column_to_score,
		'Relevance': f"{avg_relevance:.2f}",
		'Completeness': f"{avg_completeness:.2f}",
		'Accuracy': f"{avg_accuracy:.2f}",
		'Overall': f"{(avg_relevance + avg_completeness + avg_accuracy) / 3:.2f}",
	}


class Logger:
	def __init__(self, should_print:bool=True, indent_str: str="    ", indent_level=0):
		self.indent_str = indent_str
		self.indent_level = indent_level
		self._indent = ""
		self.should_print = should_print
		self._update_indent()
	
	def _update_indent(self):
		"""Update the current indentation string based on level"""
		self._indent = self.indent_str * self.indent_level
	
	def incr(self):
		"""Increase indentation level by 1"""
		self.indent_level += 1
		self._update_indent()
		return self
	def get_indent(self) -> str:
		return self._indent
	
	def decr(self):
		"""Decrease indentation level by 1 (minimum 0)"""
		self.indent_level = max(0, self.indent_level - 1)
		self._update_indent()
		return self
	
	def print(self, *args, **kwargs):
		"""
		Print with current indentation.
		Supports all the same arguments as built-in print().
		"""
		# Convert all arguments to strings
		message = " ".join(str(arg) for arg in args)
		
		# Add indentation to each line
		indented_message = "\n".join(
			self._indent + line if line.strip() else line  # Preserve empty lines
			for line in message.split("\n")
		)
		
		# Use the built-in print with remaining kwargs
		print(indented_message, **kwargs)
	
	# Alias for print to make it even more print-like
	__call__ = print

"""
summarizing all the code above in one .py pipeline
that takes inside path to (benchmark/) folder

iterates through the all experiments
take inside it config with llm's and save it's answers to the folders
and after that can score the results
"""
import openai
import typing as tp

import openai
import typing as tp
from pathlib import Path
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from langchain.schema import Document

class c1Rag:
	def __init__(
		self,
		chunker_wrapper: tp.Any,
		indexer_wrapper: tp.Any,
		llm: tuple[openai.OpenAI, str],
		eval_llm: tuple[openai.OpenAI, str],
		embedding_client: tp.Any,
		v: bool=False,
	):
		self.chunker = chunker_wrapper
		self.indexer = indexer_wrapper

		self.llm, self.llm_nm = llm[0], llm[1]
		self.eval_llm, self.eval_llm_nm = eval_llm[0], eval_llm[1]

		self.embedding_client = embedding_client
		self.files_hash: dict[str, Path] = {}
		self.chunks: list[Document] = []
		self.vectorchunks: dict[str, list[ChunkWithVector]] = {}
		self.logger = Logger(should_print=v)
			
	def process_documents(
		self,
		pt: Path
	) -> list[Document]:
		"""Read and process all documents in a folder"""
		contents = []
		for content, f_nm in read_all_documents_in_folder(
			pt=pt,
			extract_f=default_extract_cnt,
			extract_f_kwargs={"v": True},
			v=True
		):
			file_hash = uuid4().hex
			self.files_hash[file_hash] = f_nm
			doc = Document(
				page_content=content,
				metadata={
					'source': str(f_nm.name),
					'document_id': file_hash,
					'has_table': False
				}
			)
			contents.append(doc)
		return contents
	
	def evaluate_answers(
		self,
		df: pd.DataFrame,
		experiment_path: Path,
		cols: dict[str, str]
	) -> pd.DataFrame:
		"""
		Evaluate generated answers against ground truth
		"""
		df_with_score = score_column_and_save(
			df,
			experiment_path,
			cols
		)
		return df_with_score
	
	def process_set(
		self,
		set_pt: Path,
		cols: dict[str,str],
		# model name
		n_requests: tp.Optional[int] = None,
		v: bool = False
	):
		"""
		взять все вопросы из файла бенчмарка
		и сделать запрос к LLM, чтобы она ответила
		"""
		# resulting dataset
		with_answers_df = pd.DataFrame(columns=['q', 'a', 'true_a'])
		q_df = pd.read_csv(set_pt / 'bench.csv')
		if not n_requests is None:
			q_df = q_df[:n_requests]
		idx2answers = {}
		def fetch_answer(idx):
			q = q_df.loc[idx, cols['q']]
			right_answer = q_df.loc[idx, cols['a']]

			try:
				q_emb = np.array(self.embedding_client.embed_batch([q]))
				relevant_chunks = self.indexer.search_chunks(q_emb, limit=5)
				resp = generate_response_from_chunks(
					client=self.llm,
					model_name=self.llm_nm,
					q=q,
					chunks=relevant_chunks,
					docs=self.files_hash,
				)
			except Exception as e:
				resp = "<Exception>"
			if v:
				self.logger.print(f"ВОПРОС: {q}\nОТВЕТ:\n{resp}\nПРАВИЛЬНЫЙ ОТВЕТ: {right_answer}\n")
				self.logger.print("-" * 100)
			return idx, resp

		with ThreadPoolExecutor(max_workers=10) as executor:
			# параллельно постараться ответить на все вопросы из .csv
			future_to_idx = {executor.submit(fetch_answer, idx): idx for idx in q_df.index}
			with tqdm(total=len(future_to_idx), desc=self.logger.get_indent()+"Processing") as pbar:
				for future in as_completed(future_to_idx):
					idx, answer = future.result()
					idx2answers[idx] = answer
					pbar.update(1)
		
		with_answers_df['a'] = q_df.index.map(idx2answers)
		with_answers_df['q'] = q_df[cols['q']]
		with_answers_df['true_a'] = q_df[cols['a']]
		return with_answers_df

	def process(
		self,
		exp_pt: Path,
		cols: dict[str, str],
		n_requests: tp.Optional[int] = None,
		# v: bool = False
	) -> dict[str,dict[str,str]]:
		self.logger.print(f"pt:{exp_pt.resolve()}")

		pts = get_benchmark_sets(exp_pt)
		self.logger.print(f"exp pts:{pts}\n\n")

		scores_for_sets: dict[str, dict[str,str]] = {}
		for set_nm, set_pt in pts.items():
			# for each experiment:
			# 1. read question dataset
			self.logger.print(f"set \"{set_nm}\": pt={set_pt.resolve()}")
			self.logger.print("-" * 80)
			self.logger.incr()
			self.logger.print(f"question col: {cols['q']}, answer col: {cols['a']}\n")

			corresp_fs: list[Document] = self.process_documents(pt=set_pt/'files')
			self.logger.print(f"fs:\n- {',\n- '.join([f.metadata['source'] + ' ' + f.metadata['document_id'] for f in corresp_fs])}\n")
			self.logger.print(f"hashes:{self.files_hash}")

			# 2. chunk all corresponede documents inside set/files/ folder
			self.logger.print(f"Chunking")
			chunks_for_all_docs = self.chunker.process(
				corresp_fs,
				launch_f=lambda chuner_cls, txts, kwargs: chuner_cls.chunk_document(txts, **kwargs),
				chunker_kwargs={},
			)
			self.logger.print(f"len(chunks)={len(chunks_for_all_docs)}\n")

			# 3. create indeces for that files
			self.logger.print(f"Creating indeces for {set_pt / 'files'}")
			# create embeddings for these text chunks
			embs_for_all_chunks = self.embedding_client.embed_batch([
				c.text for c in chunks_for_all_docs
			])
			# create chunks with vector representation
			self.vectorchunks[set_pt.name] = [
				c.with_vector(emb)
				for c, emb in zip(
					chunks_for_all_docs,
					embs_for_all_chunks
				)
			]

			self.logger.print(f"collected: {len(self.vectorchunks[set_pt.name])}-chunks")
			self.logger.print(f"vector size in chunk:\n{len(self.vectorchunks[set_pt.name][0].vector)}\n")
			# 4. upload chunks
			self.indexer.upload_chunks(self.vectorchunks[set_pt.name])
			# 5. generate answers
			self.logger.incr()
			df_with_answers = self.process_set(
				set_pt,
				cols,
				n_requests=n_requests,
			)

			# 6. store answers
			pt_to_save = set_pt / f'rag_work_on_{set_pt.name}.csv'
			df_with_answers.to_csv(
				pt_to_save,
				index=False
			)
			# 7. validate answers
			df_with_answers = score_column_and_save(
				self.eval_llm,
				self.eval_llm_nm,
				df_with_answers,
			)
			self.logger.decr()

			# 8. save validation
			df_with_answers.to_csv(
				pt_to_save,
				index=False
			)
			# 9. return validation review
			scores_for_sets[set_nm] = average_scores(
					df_with_answers,
					column_to_score='a'
				)
			self.logger.decr()
		return scores_for_sets

__all__ = [
	'c1Rag',
	'EmbeddingClient',
	'FAISSCollection',
	'ChunkerWrapper',
]