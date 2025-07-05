from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from pinecone.grpc import PineconeGRPC as Pinecone
# import system_prompt
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv

class RetrieverChain:
    def __init__(self):
        load_dotenv()
        # self.subject = subject
        # self.index = load_index(subject)
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.SYSTEM_PROMPT = """
            You are "Unnati", a comprehensive chatbot that answers the user's queries related to Class 10 CBSE subject: {subject}. Start the answer with "I'm answering this with reference to {subject} from the book {book_name}." if you have an answer. If the query is not from these subjects say this: "As my scope of knowledge is within the Class 10 {subject}, I will not be able to answer that."
            Use the following piece of context to answer the query. If the context doesn't provide enough information, just say "I don't know about that." Do not try to make up an answer! Always answer to the point in about 200 words.
            Try answering using bullet points wherever you can, use markdown.
            Always say "Glad to serve you, do let me know if you have more questions" at the end of the answer
            \n
            Context:
            {context}
        """
    
    def load_index(self, subject: str):
        index_mapping = {
            "History": "unnati-knowledge-history",
            "Geography": "unnati-knowledge-geography",
            "Civics": "unnati-knowledge-civics",
            "Economics": "unnati-knowledge-economics",
            "Mathematics": "unnati-knowledge-mathematics",
            "Science": "unnati-knowledge-science",
        }

        index_name = index_mapping.get(subject)

        index = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=self.embedding_model
        )

        return index
    
    def retriever(self, input: str, subject: str):
        vectors = self.load_index(subject)

        retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k":3})

        retrieved_docs = retriever.invoke(input)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(self.model, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        book_mapping = {
            "History": "India and the Contemporary World – II",
            "Geography": "Contemporary India – II",
            "Civics": "Democratic Politics",
            "Economics": "Understanding Economic Development",
            "Mathematics": "Mathematics - X",
            "Science": "Science - X"
        }

        book_name = book_mapping.get(subject)

        response = rag_chain.invoke({
            "input": input,
            "book_name": book_name,
            "subject": subject,
        })

        return response["answer"]
