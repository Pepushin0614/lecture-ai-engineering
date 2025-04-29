# llm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
from config import MODEL_NAME

@st.cache_resource
def load_model():
    """トークナイザーとモデルをロードする"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=False  # rinna推奨
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"  # 自動でデバイスマッピング
        )

        st.success(f"モデル '{MODEL_NAME}' をロードしました。")
        return tokenizer, model

    except Exception as e:
        st.error(f"モデルロード失敗: {e}")
        return None, None

def generate_response(tokenizer, model, user_input):
    """ユーザー入力に対して応答を生成し、元気づけるメッセージを付加して返す"""
    if tokenizer is None or model is None:
        return "モデルがロードされていないため応答できません。", 0

    import time
    start_time = time.time()

    try:
        # ユーザー入力をそのままプロンプトに（シンプル）
        prompt = f"ユーザー: {user_input}<NL>システム: "

        # トークナイズ
        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        # モデルにより応答生成
        with torch.no_grad():
            output_ids = model.generate(
                token_ids,
                do_sample=True,
                max_new_tokens=128,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 出力のデコード
        output = tokenizer.decode(
            output_ids[0][token_ids.shape[1]:],
            skip_special_tokens=True
        )
        output = output.replace("<NL>", "\n")  # <NL>を普通の改行に変換

        end_time = time.time()
        response_time = end_time - start_time

        # --- ここで固定の元気づけメッセージを付加 ---
        greeting = "🌟 今日も頑張ってくださいね！\n\n"
        final_output = greeting + output.strip()

        return final_output, response_time

    except Exception as e:
        st.error(f"応答生成中にエラーが発生しました: {e}")
        return "エラーが発生しました。", 0