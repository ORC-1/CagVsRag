import json
import time

from cag_model import CAGModel
from rag_model import RAGModel


class ModelTester:
    def __init__(self, cag_model, rag_model):
        self.cag_model = cag_model
        self.rag_model = rag_model
        self.results = []
        self.prompt = " is also known as. use only the added text."
        self.answer_key = {
            "blackcurrant": "Vexlora",
            "cashew": "Brintal",
            "mango": "Zyphorn",
            "sugarcane": "Draquil",
            "apple": "Morlix",
            "watermelon": "Xentora",
            "cucumber": "Flomire",
            "orange": "Trevok",
            "pineapple": "Jandrix",
            "banana": "Sparven"
        }

    def test_custom_fruit(self):
        """Tests models on the custom fruit dataset from wikipedia each
        having a pseudo also known as which would be used to test accuracy."""
        fruits = self.load_fruit_dataset('test_doc/fruits.json')
        # Iterate over key-value pairs
        for key, value in fruits:
            cag_model.generate_kv_cache(key, value)
            self.run_cag(key)
            rag_model.generate_embedding(key, value)
            self.run_rag(key)
        print(self.results)

    def run_cag(self, key):
        start_time = time.time()
        res = cag_model.query_model(key + self.prompt)
        end_time = time.time()
        execution_time = end_time - start_time
        self.results.append({
            "question_key": key,
            "time_taken_seconds": execution_time,
            "answer_correct": str(self.answer_key.get(key) in res),
            "method": "cag"
        })

    def run_rag(self, key):
        start_time = time.time()
        res = rag_model.query_model(key + self.prompt)
        end_time = time.time()
        execution_time = end_time - start_time
        self.results.append({
            "question_key": key,
            "time_taken_seconds": execution_time,
            "answer_correct": str(self.answer_key.get(key) in res),
            "method": "rag"
        })

    def load_fruit_dataset(self, file_path):
        """Loads the fruit JSON file and iterates over its key-value pairs."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return data.items()

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except json.JSONDecodeError:
            print(f"Error: File '{file_path}' is not a valid JSON file.")


if __name__ == "__main__":
    rag_model = RAGModel()
    cag_model = CAGModel()
    model_tester = ModelTester(cag_model, rag_model)
    model_tester.test_custom_fruit()
