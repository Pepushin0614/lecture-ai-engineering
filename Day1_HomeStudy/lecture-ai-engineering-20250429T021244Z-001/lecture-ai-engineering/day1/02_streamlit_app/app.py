# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder

# --- アプリケーション設定 ---
st.set_page_config(page_title="Rinna Chatbot", layout="wide")

st.markdown(
    """
    <style>
    /* 背景をミディアムパープルに */
    [data-testid="stApp"] {
        background-color: #9370db; /* ミディアムパープル */
    }

    /* タイトルを可愛く */
    .stMarkdown h1 {
        font-family: 'Comic Sans MS', cursive, sans-serif;
        font-size: 40px;
        color: #ff69b4;
    }


    /* 本文テキストを黒に */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] div {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()

# LLMモデルのロード（キャッシュを利用）
# モデルをキャッシュして再利用
@st.cache_resource
# def load_model():
#    """LLMモデルをロードする"""
#    try:
#        device = "cuda" if torch.cuda.is_available() else "cpu"
#        pipe = pipeline(
#            model=MODEL_NAME,
#            device=device
#        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
#    except Exception as e:
#        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
#pipe = llm.load_model()



def load_model():
    """トークナイザーとモデルをロードする"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        st.success(f"モデル '{MODEL_NAME}' をロードしました。")
        return tokenizer, model

    except Exception as e:
        st.error(f"モデルロード失敗: {e}")
        return None, None

# ここも修正（自分のload_modelを呼ぶ）
tokenizer, model = llm.load_model()

# --- Streamlit アプリケーション ---
st.title("🌸 りんなちゃんチャットボット 🌸")
st.write("あなたの心に寄り添う、小さなAIアシスタントです。疲れたとき、嬉しいとき、何でも話しかけてね！😊")
st.markdown("---")

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
page = st.sidebar.radio(
    "ページ選択",
   # ["チャット", "履歴閲覧", "サンプルデータ管理"]
    ["チャット"]
)


# --- メインコンテンツ ---
if page == "チャット":
    if tokenizer and model:
        ui.display_chat_page(tokenizer, model)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")

elif page == "履歴閲覧":
    ui.display_history_page()

elif page == "サンプルデータ管理":
    ui.display_data_page()

# --- フッターなど（任意） ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: [Pepushin0085]")