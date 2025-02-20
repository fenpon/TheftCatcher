# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
import os

from openai import AzureOpenAI
import json
import logging
 

 



def get_email_img(version,endpoint,apikey):
    client = AzureOpenAI(
        api_version=version,
        azure_endpoint=endpoint,
        api_key=apikey,
    )
    
    result = client.images.generate(
        model="dall-e-3", # the name of your DALL-E 3 deployment
        prompt="A cute cat wearing glasses, holding a report in its paws, standing confidently as if presenting the report. The background is light blue with soft gradients, giving a professional yet friendly atmosphere. Cartoon style, clean and minimalistic design, high-quality illustration.",
        n=1
    )
    
    image_url = json.loads(result.model_dump_json())['data'][0]['url']
    
    logging.info(image_url)
    return image_url

def get_fail_img(version,endpoint,apikey):
    client = AzureOpenAI(
        api_version=version,
        azure_endpoint=endpoint,
        api_key=apikey,
    )
    
    result = client.images.generate(
        model="dall-e-3", # the name of your DALL-E 3 deployment
        prompt="A cute cat wearing glasses, bowing its head apologetically as if saying sorry. The background is light red with soft gradients, giving a professional yet friendly atmosphere. Cartoon style, clean and minimalistic design, high-quality illustration.",
        n=1
    )
    
    image_url = json.loads(result.model_dump_json())['data'][0]['url']
    
    logging.info(image_url)
    return image_url