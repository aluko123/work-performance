import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

from . import db_models


class PerformanceRAG:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.embeddings = OpenAIEmbeddings()

        self.vector_store =  Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4o-mini"),
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )

    


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

        self.vector_store.add_documents(documents)
        print(f"Indexed {len(documents)} new utterances with rich content.")


    def query_insights(self, question):
        print(f"Querying RAG chain with: '{question}'")

        result = self.qa_chain({"query": question})
        
        source_docs = result.get("source_documents", [])
        if source_docs:
            print(f"Retrieved {len(source_docs)} source documents.")

        return result.get("result", "Sorry I could not find an answer to that question.")    