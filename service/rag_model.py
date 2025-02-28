import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGModel:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """Initialize the model and tokenizer for word embeddings and generation."""
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.document_embeddings = {}
        self.document_texts = {}

    def get_embedding(self, text):
        """Get the embedding for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # Using the last hidden state of the last token as the embedding
        embedding = outputs.hidden_states[-1][:, -1, :].cpu().numpy()
        return embedding

    def index_document(self, document_id, document_text):
        """Index a document and store its embedding."""
        embedding = self.get_embedding(document_text)
        self.document_embeddings[document_id] = embedding
        self.document_texts[document_id] = document_text
        print(f"Indexed document: {document_id}")

    def retrieve_document(self, query, top_k=1):
        """Retrieve the most relevant documents based on the query."""
        query_embedding = self.get_embedding(query)
        similarities = {}
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            similarities[doc_id] = similarity

        sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        return sorted_similarities[:top_k]

    def generate_response(self, query, documents):
        """Generate a response based on the query and retrieved documents."""
        if not documents:
            return "No relevant documents found."

        retrieved_document_id = documents[0][0]
        retrieved_document_text = self.document_texts[retrieved_document_id]

        # Construct the context for the model
        context = f"Document: {retrieved_document_text}\n\nQuery: {query}:"
        # print(context)
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=150,  # Adjust as needed
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # remove the context from the generated response.
        response = response.replace(context, "").strip()

        return response

    def generate_embedding(self, document_name, document_text):
        self.index_document(document_name, document_text)

    def query_model(self, question):
        retrieved_docs = self.retrieve_document(question, top_k=1)
        if retrieved_docs:
            response = self.generate_response(question, retrieved_docs)
            return response
        else:
            return None
