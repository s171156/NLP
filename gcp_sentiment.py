import os
import requests
import pickle
# Imports the Google Cloud client library
from google.cloud import language_v1
import settings

credential_path = 'resources/GoogleCloudPlatform/Sentiment-Analysis-476a306a41f5.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
API_KEY = settings.GCP
api_endpoint = 'https://language.googleapis.com/v1/documents:analyzeSentiment?key=' + API_KEY


def analyze_sentiment_by_client(text):
    """
    NOTICE

    analyzeSentiment関数は戻り値としてAnalyzeSentimentResponseを返す。
    このオブジェクトは__dict__属性を持たないため、JSONに直接シリアライズできない。
    """
    # Instantiates a client
    # クライアントインスタンの生成
    client = language_v1.LanguageServiceClient()
    # リクエストのセット
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = 'ja'
    document = {'content': text, 'type': type_, 'language': language}
    encoding_type = language_v1.EncodingType.UTF8
    response = client.analyze_sentiment(
        request={'document': document, 'encoding_type': encoding_type})
    # 感情分析の実行
    sentiment = response.document_sentiment
    print("Text: {}".format(text))
    print("Sentiment: {}, {}".format(sentiment.score, sentiment.magnitude))


def analyze_sentiment_by_requests(text):
    """
    requestsとREST APIによる実装。
    レスポンスとしてJSONを取得できる。
    """
    # リクエストのセット
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = 'ja'
    document = {'content': text, 'type': type_, 'language': language}
    encoding_type = language_v1.EncodingType.UTF8
    response = requests.post(
        api_endpoint, json={'document': document, 'encodingType': encoding_type})
    print(response.text)


if __name__ == "__main__":
    # text = "安物かとおもったら付属品だけでもお得感がありました。マイク付きのケーブルとマイク無しのオーディオケーブルにこのヘッドホンとケーブルの接続構造は凄い。プラスチックだけど簡単には壊れないだろう。スマホps4で問題無く使えてます。持ち運びに便利な巾着がとても良いです。"
    # analyze_sentiment_by_requests(text)
    pass
