import torch
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer


class CAGModel:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """Initialize the model and tokenizer."""
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Fix padding token issue
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="mps").to(self.device)
        self.kv_cache = None
        self.document_ids = None  # store document token ids

    def precompute_kv_cache(self, document):
        """Generate and store KV cache from a document."""
        inputs = self.tokenizer(document, return_tensors="pt", truncation=True, padding=True, max_length=4096).to(
            self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)  # Precompute KV Cache

        self.kv_cache = outputs.past_key_values
        self.document_ids = inputs["input_ids"]  # store the document token ids
        return self.kv_cache

    def save_kv_cache(self, filename="kv_cache.pkl"):
        """Save KV cache to disk."""
        if self.kv_cache:
            with open(filename, "wb") as f:
                pickle.dump((self.kv_cache, self.document_ids), f)
            print(f"KV Cache saved to {filename}.")
        else:
            print("No KV Cache to save!")

    def load_kv_cache(self, filename="kv_cache.pkl"):
        """Load KV cache from disk."""
        try:
            with open(filename, "rb") as f:
                self.kv_cache, self.document_ids = pickle.load(f)
            self.kv_cache = tuple(tuple(t.to(self.device) for t in layer) for layer in
                                  self.kv_cache)  # move loaded cache to correct device.
            self.document_ids = self.document_ids.to(self.device)
            print(f"KV Cache loaded from {filename}.")
        except FileNotFoundError:
            print("KV Cache file not found. Run precompute_kv_cache first.")

    def generate_response(self, query):
        """Generate response using the precomputed KV cache."""
        if self.kv_cache is None:
            print("No KV Cache found. Generating without cache...")
            return self.generate_response_no_cache(query)

        query_inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(
            self.device)
        input_ids = torch.cat([self.document_ids, query_inputs["input_ids"][:, 1:]],
                              dim=-1)  # merge document and query. remove first token of query.
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=self.kv_cache,
                max_new_tokens=100,
                pad_token_id=self.tokenizer.pad_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_response_no_cache(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(
            self.device)
        attention_mask = inputs["attention_mask"] if "attention_mask" in inputs else None
        with torch.no_grad():
            output = self.model.generate(
                inputs["input_ids"],
                attention_mask=attention_mask,
                max_new_tokens=100,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_kv_cache(self, document_name, document_text):
        cache_name = 'cache_files/' + document_name+'.pkl'
        # Precompute and store KV cache
        self.precompute_kv_cache(document_text)
        self.save_kv_cache(cache_name)
        # Load KV Cache
        self.load_kv_cache(cache_name)

    def query_model(self, question):
        response = self.generate_response(question)
        return response


# Example Usage
# if __name__ == "__main__":
#     model = CAGModel(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
#     document_text = "This is an important document for customer support AI."
#
#     # Precompute and store KV cache
#     kv_cache = model.precompute_kv_cache(document_text)
#     model.save_kv_cache()
#
#     # Load KV Cache and use it for response generation
#     model.load_kv_cache()
#     response = model.generate_response("What does the document say about AI?")
#     print("Response:", response)
