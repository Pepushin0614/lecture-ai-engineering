# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:54:49 2025

@author: cloud
"""

from dotenv import load_dotenv, find_dotenv
import os
import re
def extract_username():
    # 現在のユーザーフォルダのパスを取得
    path = os.path.expanduser("~")    
    # 正規表現パターンの定義
    
    pattern = r"C:\\Users\\([^\\]+)"
    # 正規表現を使用してユーザー名を抽出
    match = re.search(pattern, path)
    if match:
        username = match.group(1)
        if username in ["gr0469ih", "cloud"]:
            return username
        else:
            return "指定されたユーザー名ではありません。"
    else:
        return "ユーザー名を抽出できませんでした。"

username = extract_username()

project_root = "C:\\Users\\"+username+"\\OneDrive - 学校法人立命館\\博士後期課程資料\\GCIエンジニアリング\\lecture-ai-engineering\\day1"

os.chdir(project_root)

#%%
load_dotenv(find_dotenv())

#%%
os.chdir(project_root+"\\01_streamlit_UI")

#%%
if not os.path.exists("C:\\temp"):
    os.makedirs("C:\\temp")

from pyngrok import ngrok

public_url = ngrok.connect(8501).public_url
print(f"公開URL: {public_url}")
