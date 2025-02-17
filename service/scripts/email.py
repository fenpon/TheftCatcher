# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
 

 
# .env 파일 로드
load_dotenv()
# 환경 변수 가져오기
_api_version = os.getenv("dalle_api_version")
_azure_endpoint = os.getenv("dalle_azure_endpoint")

_api_key = os.getenv("dalle_api_key")


def get_email_img():
    client = AzureOpenAI(
        api_version=_api_version,
        azure_endpoint=_azure_endpoint,
        api_key=_api_key,
    )
    
    result = client.images.generate(
        model="dall-e-3", # the name of your DALL-E 3 deployment
        prompt="A cute cat wearing glasses, holding a report in its paws, standing confidently as if presenting the report. The background is light blue with soft gradients, giving a professional yet friendly atmosphere. Cartoon style, clean and minimalistic design, high-quality illustration.",
        n=1
    )
    
    image_url = json.loads(result.model_dump_json())['data'][0]['url']
    
    print(image_url)
    return image_url