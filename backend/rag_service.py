import os
import asyncio
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

from . import db_models
from .rag_graph import RAGGraph


class PerformanceRAG:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )
        # LangGraph-powered pipeline for insights
        self.graph = RAGGraph(self.vector_store)

    


    def index_utterances(self, utterances):
        if not utterances:
            print("No utterances to index.")
            return
        
        #create a list of langchain documents
        documents = []
        for utterance in utterances:
            scores_text = str({**utterance.predictions, **utterance.aggregated_scores})
            page_content = (
                f"On {utterance.date} at {utterance.timestamp}, {utterance.speaker} said: "
                f"'{utterance.text}'\\n"
                f"The scores for this utterance were: {scores_text}"
            )
            
            doc = Document(
                page_content=page_content,
                metadata={
                    "speaker": utterance.speaker,
                    "date": utterance.date,
                    "timestamp": utterance.timestamp,
                    #"scores": scores_text,
                    "source_id": utterance.id #store original DB for reference
                }
            
            )
            documents.append(doc)
        if not documents:
            return
        
        #BATCH IN PARALLEL FOR EMBEDDING PURPOSES
        batch_size = int(os.getenv("BATCH_SIZE", 50))
        total_docs = len(documents)
        print(f"Starting to index {total_docs} documents in parallel batches of {batch_size}...")

        #tasks = []
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            # async task for each batch
            # task = asyncio.create_task(self.vector_store.add_documents(batch))
            # tasks.append(task)
            try:
                self.vector_store.add_documents(batch)
                print(f"Successfully indexed batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}...")
            except Exception as e:
                print(f"An error occurred during batch indexing: {e}")
                continue
        
        # #run concurrently
        # if tasks:
        #     await asyncio.gather(*tasks)

        try:
            self.vector_store.persist()
        except Exception as e:
            print(f"Warning: failed to persist vector store: {e}")
        print(f"Finished indexing {total_docs} utterances.")

        # self.vector_store.add_documents(documents)
        # print(f"Indexed {len(documents)} new utterances with rich content.")


    def query_insights(self, question, session_id: str | None = None, filters: dict | None = None):
        print(f"Querying RAG graph with: '{question}'")
        try:
            result = self.graph.run(question=question, session_id=session_id, filters=filters or {})
            return result
        except Exception as e:
            print(f"RAG graph error: {e}")
            return {
                "answer": "Sorry, I could not process that question.",
                "bullets": [],
                "metrics_summary": [],
                "citations": [],
                "follow_ups": [],
                "metadata": {"error": str(e)},
            }
