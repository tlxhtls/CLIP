import openai
import torch
import clip
import os
from io import BytesIO
import boto3
from dotenv import load_dotenv
load_dotenv()

accessKey = os.getenv('AWS_ACCESS_KEY')
secretKey = os.getenv('AWS_SECRET_KEY')
region = os.getenv('AWS_REGION')
bucket_name = os.getenv('AWS_BUCKET_NAME')

openai.api_key = os.getenv('OPENAI_API_KEY')

class ClipSearch:

    def __init__(self, query, user_id, top_n=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        self.gpt_model = "gpt-3.5-turbo"
        self.query = query
        self.user_id = user_id
        self.s3 = self.start_s3()
        self.top_n = top_n


    def start_s3(self):
        # Create a session using your AWS credentials
        session = boto3.Session(
          aws_access_key_id=accessKey,
          aws_secret_access_key=secretKey,
          region_name=region
        )
        s3 = session.client('s3')
        return s3

    def translate_query(self):
        messages = [{
            "role": "user",
            "content": f"Translate the text delimited by triple backticks into English.```{self.query}```"
        }]
        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"]


    def load_image_features(self):
        # Specify the paths
        s3_image_features_path = f"{self.user_id}/processed/image_features.pt"
        s3_image_paths_csv_path = f"{self.user_id}/processed/image_paths.csv"

        # Download and load the image features
        byte_stream = BytesIO()
        self.s3.download_fileobj(bucket_name, s3_image_features_path, byte_stream)
        byte_stream.seek(0)
        image_features = torch.load(byte_stream)

        # Download and load the image paths
        byte_stream = BytesIO()
        self.s3.download_fileobj(bucket_name, s3_image_paths_csv_path, byte_stream)
        byte_stream.seek(0)
        image_paths = byte_stream.read().decode('utf-8').split('/n')
        return image_features, image_paths

    def text_vectorizer(self):
        translated_query = self.translate_query()
        text_inputs = clip.tokenize([translated_query]).to(self.device)
        text_features = self.model.encode_text(text_inputs)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features_norm

    def searching_by_cos_similarity(self, image_features, text_features):
        similarity = (text_features @ image_features.T).squeeze(0)
        top_results = similarity.argsort(descending=True)[:self.top_n]
        return top_results

    def extract_top_results(self):
        image_features, image_paths = self.load_image_features()
        text_features = self.text_vectorizer()
        top_results = self.searching_by_cos_similarity(image_features, text_features)
        top_results_list = [image_paths[result] for result in top_results]
        return top_results_list