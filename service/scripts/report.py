import requests
import json
import textwrap
import tempfile
import cv2
import io
import os
import platform
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, ListFlowable,Frame, PageTemplate,PageBreak,Image
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.enums import TA_JUSTIFY
from datetime import datetime,timedelta
from PIL import Image as PILImage
import pytz
import markdown
import sys
import subprocess
sys.stdout.reconfigure(encoding='utf-8')


import re

import tempfile

from azure.storage.blob import  BlobServiceClient, generate_blob_sas, BlobSasPermissions

import logging



# .env 파일 로드
load_dotenv()


# 환경 변수 가져오기
API_KEY = os.getenv("API_KEY")
ENDPOINT = os.getenv("ENDPOINT")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")


base_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(base_dir, "font", "NanumGothic.ttf")
print(font_path)

pdfmetrics.registerFont(TTFont("NanumGothic", font_path))
pdfmetrics.registerFontFamily("NanumGothic", normal="NanumGothic", bold="NanumGothic", italic="NanumGothic")

FONT_NAME = "NanumGothic"
FONT_SIZE = 12  # 본문 폰트 크기
TITLE_SIZE = 18  # 제목 폰트 크기
SUBTITLE_SIZE = 14  # 부제목 폰트 크기
LINE_SPACING = 16  # 줄 간격
MARGIN = 50  # 좌우 여백


def report_analyze(video_report, location):
    logging.info(f"report_analyze : {video_report} // {location}")

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }

    payload = {
         "messages": [
            {"role": "system", "content": "CCTV 영상을 분석한 결과를 주면 이에 맞쳐 절도 방지 대책을 제시해 드립니다."},
            {"role": "user", "content": f"무인매장인 {location}의 CCTV 영상을 인공지능이 분석한 결과 아래와 같이 방문객의 절도 동작이 탐지되었다는 메시지가 나오고 절도동작이 탐지가 되었어 "},
            {"role": "user", "content": f"AI 분석 결과: {video_report}"},
            {"role": "user", "content": "위 메시지와 지역을 기반으로 절도 탐지 대책을 알려줘."}
        ],
        "max_tokens": 7000
    }

    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)

        # 응답 상태 코드 확인
        if response.status_code != 200:
            logging.error(f"요청 실패: {response.status_code}, 응답: {response.text}")
            return {"error": f"Request failed with status {response.status_code}"}

        # 응답 인코딩 설정
        response.encoding = "utf-8"

        # 응답이 비어 있는지 확인
        if not response.text.strip():
            logging.error("서버 응답이 비어 있습니다.")
            return {"error": "Empty response from server"}

        # JSON 변환 시도
        try:
            data = response.json()
            result = data['choices'][0]['message']['content']
            # UTF-8로 변환
            result = result.encode('utf-8').decode('utf-8')  
        except json.JSONDecodeError:
            logging.error(f"JSON 디코딩 실패. 응답 내용: {response.text}")
            return {"error": "Invalid JSON response"}

        logging.info("- OpenAI 분석 완료 -")
        return result

    except requests.RequestException as e:
        logging.error(f"요청 중 오류 {e}")





TITLE_SIZE = 18  # 제목 폰트 크기
SUBTITLE_SIZE = 16  # 부제목 폰트 크기
FONT_SIZE = 16  # 폰트 크기
LINE_SPACING = 20  # 줄 간격
MARGIN = 50  # 좌우 여백
def add_cover_page(c, doc):
        # ✅ 첫 페이지에 이미지 추가
    cover_image_path = os.path.join(base_dir, "img", "cover.png") # ✅ 여기에 이미지 경로 입력
    c.saveState()
      # 배경 이미지 그리기 (A4 크기에 맞게 조정)
    c.drawImage(
            cover_image_path,  # 배경 이미지 경로
            0, 0,
            width=A4[0], height=A4[1],  # 페이지 크기에 맞춤
            preserveAspectRatio=True
    )
    c.restoreState()
def add_background(c,doc):
    bg_img_path = os.path.join(base_dir, "img", "background.png")
    c.saveState()
      # 배경 이미지 그리기 (A4 크기에 맞게 조정)
    c.drawImage(
            bg_img_path,  # 배경 이미지 경로
            0, 0,
            width=A4[0], height=A4[1],  # 페이지 크기에 맞춤
            preserveAspectRatio=True
    )
    c.restoreState()

def convert_markdown_to_paragraphs(md_text, style):
    """ Markdown을 ReportLab용 Paragraph 리스트로 변환 """
    html_text = markdown.markdown(md_text)  # Markdown을 HTML로 변환
    lines = html_text.split("\n")  # 줄 단위 분리
    
    paragraphs = []
    for line in lines:
        line = line.strip()
        
        if line.startswith("<h3>"):  # 제목 변환
            title_text = line.replace("<h3>", "").replace("</h3>", "").strip()
            title_style = ParagraphStyle(
                "TitleStyle",
                parent=style,
                fontSize=14,
                spaceAfter=10,
                alignment=TA_JUSTIFY,
                textColor="black",
                bold=True
            )
            paragraphs.append(Paragraph(title_text, title_style))

        elif line.startswith("<ul>"):  # 리스트 변환
            list_items = []
            for item in line.replace("<ul>", "").replace("</ul>", "").split("</li>"):
                if "<li>" in item:
                    list_items.append(Paragraph(item.replace("<li>", "").strip(), style))
            if list_items:
                paragraphs.append(ListFlowable(list_items, bulletType="bullet"))

        elif line:  # 일반 문단 처리
            paragraphs.append(Paragraph(line, style))
        
        paragraphs.append(Spacer(1, 10))  # 문단 간 간격 추가

    return paragraphs

#GhostScripts 설치 필요
def convert_to_pdfa(input_pdf):
    """PDF를 PDF/A로 변환"""
    output_pdfa = input_pdf.replace(".pdf", "_pdfa.pdf")
    
    # Windows는 gswin64c, 다른 OS는 gs 사용
    gs_executable = "gswin64c" if platform.system() == "Windows" else "gs"
    gs_command = [
        gs_executable,
        "-dPDFA",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=pdfwrite",
        "-sOutputFile=" + output_pdfa,
        "-dPDFACompatibilityPolicy=1",
        "-dEmbedAllFonts=true",
        "-dSubsetFonts=false",
        input_pdf
    ]
    
    try:
        subprocess.run(gs_command, check=True)
        print(f"✅ PDF/A 변환 성공: {output_pdfa}")
        return output_pdfa
    except subprocess.CalledProcessError as e:
        print("⚠️ PDF/A 변환 실패:", e)
        return input_pdf  # 변환 실패 시 원본 반환

def make_pdf(text,predicts_img):
    title = "절도 방지 대책 보고서"  # 문서 제목
    # ✅ 임시 PDF 파일 생성
    temp_fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(temp_fd)

    # ✅ PDF 문서 설정
    doc = SimpleDocTemplate(temp_pdf_path, pagesize=A4,
                            rightMargin=MARGIN, leftMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN,encoding="utf-8")
    
    styles = getSampleStyleSheet()
    # ✅ 한글이 깨지지 않도록 스타일 수정
    style_korean = ParagraphStyle(
        "KoreanStyle",
        parent=styles["Normal"],
        fontName=FONT_NAME,  # ✅ 한글 폰트 사용
        fontSize=12,
        leading=18,  # 줄 간격 설정
        alignment=TA_JUSTIFY  # 양쪽 정렬
    )

    # ✅ Markdown을 ReportLab Paragraph로 변환
    content = convert_markdown_to_paragraphs(text, style_korean)

    # ✅ 첫 번째 페이지 (표지) 프레임
    cover_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='cover')

    # ✅ 본문 페이지 프레임
    content_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')

    # ✅ 페이지 템플릿 설정
    cover_template = PageTemplate(id='cover', frames=[cover_frame], onPage=add_cover_page)  # 첫 페이지 (표지)
    content_template = PageTemplate(id='normal', frames=[content_frame], onPage=add_background, onPageEnd=add_background)  # 모든 페이지 배경 적용

    doc.addPageTemplates([cover_template, content_template])  # 템플릿 추가
    

    ii = 0
    # ✅ OpenCV 이미지 변환 후 추가
    
    for img_cv in predicts_img:
        if ii % 2 == 0:
            ii = 0
            content.append(PageBreak())
          
            title_style = ParagraphStyle(
                "TitleStyle",
                parent=style_korean,
                fontSize=14,
                spaceAfter=10,
                alignment=TA_JUSTIFY,
                textColor="black",
                bold=True
            )
            content.append(Paragraph("절도 탐지 장면", title_style))

            content.append(Spacer(1, 10))  # 문단 간 간격 추가

        # OpenCV 이미지를 PIL 이미지로 변환
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환
        height, width, _ = img_rgb.shape
        ratio = height/width  

        pil_img = PILImage.fromarray(img_rgb)

        # PIL 이미지를 BytesIO에 저장
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # ReportLab 이미지 객체 생성
        img = Image(img_buffer, width=480, height=480*ratio)  # 이미지 크기 조절 가능
       
        content.append(img)
        content.append(Spacer(1, 20))  # 문단 간 간격 추가
        #content.append(PageBreak())  # 각 이미지마다 페이지 나누기 (필요에 따라 조절 가능)
        ii += 1
  

    # ✅ 문서 빌드 (표지 → 빈 페이지 → 본문)
    doc.build(
        [PageBreak()] + content, 
        onFirstPage=add_cover_page,
        onLaterPages=add_background
    )

    # ✅ PDF/A 변환
    pdfa_path = convert_to_pdfa(temp_pdf_path)
    
    # 서울 (KST) 시간
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.now(kst)
   
    download_link = upload_to_blob_storage(pdfa_path, f"report_{now_kst.strftime('%Y-%m-%d_%H-%M-%S')}.pdf")
    
    print(f"PDF 생성 완료: {pdfa_path}")
    return download_link

def generate_blob_download_link(file_name,blob_service_client, expiry_hours=72):
    #72 시간 다운 가능
    try:
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=CONTAINER_NAME,
            blob_name=file_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),  # 읽기 권한만 부여
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours)  # 만료시간 설정
        )

        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{CONTAINER_NAME}/{file_name}?{sas_token}"
        return blob_url
    except Exception as e:
        logging.error(f"다운로드 링크 생성 오류: {str(e)}")
        return None
    
def upload_to_blob_storage(file_path, file_name):
    try:
        print("PDF 업로드 진행")
        # Blob Service Client 생성
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=file_name)

        # 파일을 Blob Storage에 업로드
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        logging.info(f"✅ Blob Storage에 업로드 완료: {file_name}")
        sas_url = generate_blob_download_link(file_name,blob_service_client=blob_service_client)
        return sas_url
    except Exception as e:
        logging.error(f"업로드 오류: {str(e)}")
        return None

