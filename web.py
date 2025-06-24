import os
import pathlib
import uuid
import json
import pandas as pd
from werkzeug.utils import secure_filename
from azure.core.credentials import AzureKeyCredential
from flask import Flask, request, render_template, redirect, url_for
from dotenv import load_dotenv
import requests
from azure.storage.blob import BlobServiceClient
import azure.cognitiveservices.speech as speechsdk
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem
from azure.ai.vision.face import FaceAdministrationClient, FaceClient
from azure.ai.vision.face.models import FaceAttributeTypeRecognition04, FaceDetectionModel, FaceRecognitionModel, QualityForRecognition
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from werkzeug.utils import secure_filename
from azure.ai.formrecognizer import DocumentAnalysisClient, DocumentAnalysisApiVersion
from azure.ai.formrecognizer import FormContentType
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import TextCategory, AnalyzeTextOptions
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from openai import AzureOpenAI  # ✅ 使用新版 openai 套件方式
from scipy import stats

# Load environment variables
load_dotenv(pathlib.Path(".env"))

app = Flask(__name__)
os.makedirs("static", exist_ok=True)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Azure service keys and endpoints
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_KEY = os.getenv("VISION_KEY")
FACE_ENDPOINT = os.getenv("FACE_ENDPOINT")
FACE_KEY = os.getenv("FACE_KEY")
TRANSLATOR_KEY = os.getenv("TRANSLATOR_KEY")
TRANSLATOR_REGION = os.getenv("TRANSLATOR_REGION")
TRANSLATOR_ENDPOINT = os.getenv("TRANSLATOR_ENDPOINT")
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")
TEXT_API_KEY = os.getenv("TEXT_API_KEY")
TEXT_API_ENDPOINT = os.getenv("TEXT_API_ENDPOINT")
DOC_KEY = os.getenv("DOC_INTELLIGENCE_KEY")
DOC_ENDPOINT = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
CONTENT_SAFETY_KEY = os.getenv("CONTENT_SAFETY_KEY")
CONTENT_SAFETY_ENDPOINT = os.getenv("CONTENT_SAFETY_ENDPOINT")
AZURE_BLOB_CONN_STR = os.getenv("AZURE_BLOB_CONN_STR")
BLOB_CONTAINER = "photos"
AZURE_MAPS_KEY = os.getenv("AZURE_MAPS_KEY")
MAPS_CLIENT_ID = os.getenv("AZURE_MAPS_CLIENT_ID")
ANOMALY_KEY = os.getenv("ANOMALY_KEY")
ANOMALY_ENDPOINT = os.getenv("ANOMALY_ENDPOINT")

# Azure Blob client
blob_service = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
container_client = blob_service.get_container_client(BLOB_CONTAINER)
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
)
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")  # 如 gpt4bot、chatbot 等
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/image_tool", methods=["GET", "POST"])
def index():
    image_url = ""
    description = ""
    tags = []
    tags_zh = []
    translation = ""
    blob_image_url = ""
    blob_audio_url = ""
    face_info = ""

    if request.method == "POST":
        image_url = request.form.get("image_url")
        pair_id = uuid.uuid4().hex

        # Computer Vision
        vision_url = VISION_ENDPOINT.rstrip("/") + "/vision/v3.2/analyze"
        headers = {"Ocp-Apim-Subscription-Key": VISION_KEY, "Content-Type": "application/json"}
        params = {"visualFeatures": "Description,Tags"}
        body = {"url": image_url}
        vision_resp = requests.post(vision_url, headers=headers, params=params, json=body)
        vision_data = vision_resp.json()
        description = vision_data.get("description", {}).get("captions", [{}])[0].get("text", "")
        tag_objs = vision_data.get("tags", [])
        if tag_objs:
            top_tag = sorted(tag_objs, key=lambda x: x.get("confidence", 0), reverse=True)[0]
            tags = [top_tag["name"]]

        # Translate description
        translator = TextTranslationClient(
            endpoint=TRANSLATOR_ENDPOINT,
            credential=TranslatorCredential(TRANSLATOR_KEY, TRANSLATOR_REGION)
        )
        translation_result = translator.translate(
            content=[InputTextItem(text=description)],
            from_parameter="en",
            to=["zh-Hant"]
        )
        translation = translation_result[0].translations[0].text

        # Translate tag
        if tags:
            tag_items = [InputTextItem(text=tag) for tag in tags]
            tag_result = translator.translate(content=tag_items, from_parameter="en", to=["zh-Hant"])
            tags_zh = [r.translations[0].text for r in tag_result]

        # TTS
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        audio_filename = f"speech_{pair_id}.mp3"
        audio_path = os.path.join("static", audio_filename)
        audio_output = speechsdk.audio.AudioOutputConfig(filename=audio_path)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)
        synthesizer.speak_text_async(translation).get()

        with open(audio_path, "rb") as audio_file:
            container_client.upload_blob(name=audio_filename, data=audio_file, overwrite=True)
        blob_audio_url = f"{container_client.url}/{audio_filename}"

        # Upload image
        image_resp = requests.get(image_url)
        image_filename = f"image_{pair_id}.jpg"
        container_client.upload_blob(name=image_filename, data=image_resp.content, overwrite=True)
        blob_image_url = f"{container_client.url}/{image_filename}"

        # Save metadata
        metadata = {
            "id": pair_id,
            "image": blob_image_url,
            "audio": blob_audio_url,
            "tags": tags,
            "tags_zh": tags_zh
        }
        tagfile = f"meta_{pair_id}.json"
        container_client.upload_blob(name=tagfile, data=json.dumps(metadata), overwrite=True)

    return render_template("index.html",
                           image_url=blob_image_url,
                           description=description,
                           tags=tags,
                           tags_zh=tags_zh,
                           translation=translation,
                           audio_url=blob_audio_url)

@app.route("/history")
def history():
    blobs = container_client.list_blobs()
    grouped = {}
    for blob in blobs:
        if blob.name.startswith("meta_") and blob.name.endswith(".json"):
            content = container_client.download_blob(blob).readall()
            data = json.loads(content)
            tags_zh = data.get("tags_zh", [])
            zh_tag = tags_zh[0] if tags_zh else "未知"
            grouped.setdefault(zh_tag, []).append(data)
    return render_template("history.html", grouped=grouped)


@app.route("/delete/<item_id>", methods=["POST"])
def delete_item(item_id):
    try:
        container_client.delete_blob(f"image_{item_id}.jpg")
        container_client.delete_blob(f"speech_{item_id}.mp3")
        container_client.delete_blob(f"meta_{item_id}.json")
    except Exception as e:
        print("刪除失敗：", e)
    return redirect(url_for("history"))

@app.route("/go_history")
def go_history():
    return redirect(url_for("history"))

@app.route("/text_sentiment", methods=["GET", "POST"])
def text_sentiment():
    text_input = ""
    sentiment = ""
    confidence = ""
    warning_message = ""
    content_safety_results = {
        "hate": "",
        "self_harm": "",
        "sexual": "",
        "violence": ""
    }

    if request.method == "POST":
        text_input = request.form.get("user_text", "").strip()
        if not text_input:
            sentiment = "❌ 請輸入有效文字"
            return render_template("sentiment.html", text=text_input, sentiment=sentiment, confidence="")

        # Sentiment analysis
        credential = AzureKeyCredential(TEXT_API_KEY)
        text_client = TextAnalyticsClient(endpoint=TEXT_API_ENDPOINT, credential=credential)
        documents = [{"id": "1", "text": text_input}]
        response = text_client.analyze_sentiment(documents=documents)[0]
        sentiment_map = {"positive": "正面", "neutral": "中立", "negative": "負面"}
        sentiment = sentiment_map.get(response.sentiment, response.sentiment)
        confidence = (
            f"正面: {response.confidence_scores.positive:.2f}, "
            f"中立: {response.confidence_scores.neutral:.2f}, "
            f"負面: {response.confidence_scores.negative:.2f}"
        )

        # Content safety analysis
        try:
            safety_client = ContentSafetyClient(CONTENT_SAFETY_ENDPOINT, AzureKeyCredential(CONTENT_SAFETY_KEY))
            safety_result = safety_client.analyze_text(AnalyzeTextOptions(text=text_input))
            SEVERITY_THRESHOLD = 2

            for item in safety_result.categories_analysis:
                if item.category == TextCategory.HATE:
                    content_safety_results["hate"] = f"仇恨內容嚴重程度：{item.severity}"
                    if item.severity >= SEVERITY_THRESHOLD:
                        warning_message = "⚠️ 文字中含有仇恨語言，請嘗試輸入其他內容。"
                elif item.category == TextCategory.SELF_HARM:
                    content_safety_results["self_harm"] = f"自我傷害內容嚴重程度：{item.severity}"
                    if item.severity >= SEVERITY_THRESHOLD:
                        warning_message = "⚠️ 文字中出現自我傷害傾向，請重新輸入健康內容。"
                elif item.category == TextCategory.SEXUAL:
                    content_safety_results["sexual"] = f"性內容嚴重程度：{item.severity}"
                    if item.severity >= SEVERITY_THRESHOLD:
                        warning_message = "⚠️ 文字中包含敏感性內容，請嘗試輸入其他內容。"
                elif item.category == TextCategory.VIOLENCE:
                    content_safety_results["violence"] = f"暴力內容嚴重程度：{item.severity}"
                    if item.severity >= SEVERITY_THRESHOLD:
                        warning_message = "⚠️ 文字中包含暴力相關字眼，請嘗試輸入其他內容。"
        except Exception as e:
            content_safety_results["error"] = f"⚠️ 安全性分析失敗: {e}"

    return render_template(
        "sentiment.html",
        text=text_input,
        sentiment=sentiment,
        confidence=confidence,
        content_safety=content_safety_results,
        warning=warning_message
    )


doc_client = DocumentAnalysisClient(endpoint=DOC_ENDPOINT, credential=AzureKeyCredential(DOC_KEY))

@app.route("/pdf_summary", methods=["GET", "POST"])
def pdf_summary():
    summary = ""
    filename = ""

    if request.method == "POST":
        if "pdf_file" not in request.files:
            return render_template("pdf_summary.html", summary="沒有上傳檔案")

        file = request.files["pdf_file"]
        if file.filename == "":
            return render_template("pdf_summary.html", summary="請選擇檔案")

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            with open(file_path, "rb") as f:
                poller = doc_client.begin_analyze_document(
                    model_id="prebuilt-read",
                    document=f,
                    
                )

                result = poller.result()
                if result.pages:
                    text = "\n".join([line.content for page in result.pages for line in page.lines])
                else:
                    text = "⚠️ 無法擷取內容"

                summary = text[:1000] + "..." if len(text) > 1000 else text

    return render_template("pdf_summary.html", summary=summary, filename=filename)

vision_client = ComputerVisionClient(VISION_ENDPOINT, CognitiveServicesCredentials(VISION_KEY))

@app.route("/ocr_tool", methods=["GET", "POST"])
def ocr_tool():
    extracted_text = ""
    filename = ""

    if request.method == "POST":
        if "image_file" not in request.files:
            return render_template("ocr_tool.html", text="沒有上傳檔案", filename="")

        file = request.files["image_file"]
        if file.filename == "":
            return render_template("ocr_tool.html", text="請選擇檔案", filename="")

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            with open(file_path, "rb") as f:
                image_data = f.read()

            try:
                headers = {
                    "Ocp-Apim-Subscription-Key": VISION_KEY,
                    "Content-Type": "application/octet-stream"
                }
                params = {"api-version": "2023-10-01"}
                ocr_url = f"{VISION_ENDPOINT}/computervision/imageanalysis:analyze?features=read"
                response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
                result = response.json()

                print("[DEBUG] OCR API 回傳：", json.dumps(result, indent=2, ensure_ascii=False))

                lines = []
                blocks = result.get("readResult", {}).get("blocks", [])
                for block in blocks:
                    for line in block.get("lines", []):
                        text = line.get("text", "")
                        if text:
                            lines.append(text)

                extracted_text = "\n".join(lines) if lines else "⚠️ 沒有偵測到文字"

            except Exception as e:
                extracted_text = f"⚠️ 文字擷取失敗：{str(e)}"

    return render_template("ocr_tool.html", text=extracted_text, filename=filename)

# OCR route + Azure Maps integration
@app.route("/ocr_map_tool", methods=["GET", "POST"])
def ocr_map_tool():
    extracted_text = ""
    filename = ""
    map_coords = None
    query_address = ""

    print("🔍 [INFO] 使用者進入 /ocr_map_tool")

    if request.method == "POST":
        print("📩 [INFO] 收到 POST 請求")

        if "image_file" in request.files:
            file = request.files["image_file"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                print(f"🖼️ [INFO] 上傳圖片儲存成功：{file_path}")

                with open(file_path, "rb") as f:
                    image_data = f.read()

                try:
                    headers = {
                        "Ocp-Apim-Subscription-Key": VISION_KEY,
                        "Content-Type": "application/octet-stream"
                    }
                    params = {"api-version": "2023-10-01"}
                    ocr_url = f"{VISION_ENDPOINT}/computervision/imageanalysis:analyze?features=read"
                    response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
                    result = response.json()

                    print("📤 [DEBUG] Azure OCR 回傳內容：", result)

                    lines = []
                    read_result = result.get("readResult")
                    if read_result and "blocks" in read_result:
                        blocks = read_result["blocks"]
                        for block in blocks:
                            for line in block.get("lines", []):
                                lines.append(line.get("text", ""))
                        extracted_text = "\n".join(lines) if lines else "⚠️ 沒有偵測到文字"
                        print("📄 [INFO] 擷取文字成功：", extracted_text)
                    else:
                        extracted_text = f"⚠️ 無法擷取文字：{result.get('error', {}).get('message', '未知錯誤')}"
                        print("⚠️ [ERROR] OCR 無法擷取文字")

                except Exception as e:
                    extracted_text = f"⚠️ 文字擷取失敗：{str(e)}"
                    print("❌ [EXCEPTION] OCR 發生錯誤：", e)

        if request.form.get("map_search"):
            query_address = request.form.get("address", "")
            print(f"📍 [INFO] 使用者輸入地址：{query_address}")
            if query_address:
                try:
                    maps_url = "https://atlas.microsoft.com/search/address/json"
                    maps_params = {
                        "api-version": "1.0",
                        "subscription-key": AZURE_MAPS_KEY,
                        "query": query_address
                    }
                    maps_resp = requests.get(maps_url, params=maps_params)
                    maps_data = maps_resp.json()
                    print("🗺️ [DEBUG] Azure Maps 回傳：", maps_data)

                    position = maps_data.get("results", [{}])[0].get("position", {})
                    if position:
                        map_coords = {
                            "lat": position.get("lat"),
                            "lon": position.get("lon")
                        }
                        print("✅ [INFO] 查詢座標成功：", map_coords)
                    else:
                        print("⚠️ [WARNING] 查無座標")

                except Exception as e:
                    extracted_text += f"\n⚠️ 查詢地圖錯誤：{e}"
                    print("❌ [EXCEPTION] 查詢地圖錯誤：", e)

    print("📦 [DEBUG] 傳入模板的 map_coords：", map_coords)
    print("🔑 [DEBUG] 使用的 AZURE_MAPS_KEY 是否存在：", bool(AZURE_MAPS_KEY))

    return render_template(
        "ocr_map_tool.html",
        text=extracted_text,
        filename=filename,
        map_coords=map_coords,
        query_address=query_address,
        azure_maps_key=AZURE_MAPS_KEY  # 加這行很關鍵！
    )

# 主頁：飲食推薦
@app.route("/recommand", methods=["GET", "POST"])
def diet_recommend():
    result = ""
    if request.method == "POST":
        height = request.form.get("height")
        weight = request.form.get("weight")
        goal = request.form.get("goal")
        preference = request.form.get("preference")

        prompt = f"""
        使用者身高 {height} 公分，體重 {weight} 公斤，
        飲食目標是「{goal}」，飲食偏好是「{preference}」。
        請用繁體中文推薦今天的三餐，列出每一餐的內容與推薦理由。
        """

        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content
        except Exception as e:
            result = f"❗ 發生錯誤：{e}"

    return render_template("recommand.html", result=result)

@app.route("/favicon.ico")
def favicon():
    return '', 204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)