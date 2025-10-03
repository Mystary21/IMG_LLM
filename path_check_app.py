# -*- coding: utf-8 -*-
import streamlit as st
import os

st.set_page_config(layout="wide", page_title="路徑偵錯工具")

st.title("🔬 容器內部路徑偵錯工具")

# --- 我們要檢查的核心路徑 ---
# 這是我們期望模型所在的完整路徑
MODEL_PATH = "/root/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0"
# 這是掛載的根目錄，我們來看看裡面有什麼
HUB_PATH = "/root/.cache/huggingface/hub"

st.header("1. 檢查模型資料夾是否存在")
st.write(f"腳本正在檢查以下路徑是否存在：`{MODEL_PATH}`")

if os.path.exists(MODEL_PATH):
    st.success(f"✅ 成功！指定的模型路徑存在。")
else:
    st.error(f"❌ 失敗！在容器內找不到指定的模型路徑。")
    st.info("這通常代表您的 `podman run -v` 指令中的主機路徑有誤，或者該路徑下沒有模型。")

st.markdown("---")

st.header(f"2. 查看掛載目錄 `{HUB_PATH}` 的內容")
st.write("以下是容器實際看到的檔案清單：")

if os.path.exists(HUB_PATH):
    try:
        # 列出 HUB_PATH 裡的所有檔案和資料夾
        contents = os.listdir(HUB_PATH)
        if contents:
            st.code("\n".join(contents))
        else:
            st.warning("警告：`hub` 資料夾存在，但裡面是空的。")
    except Exception as e:
        st.error(f"讀取 `{HUB_PATH}` 時發生錯誤: {e}")
else:
    st.error(f"失敗！連掛載的根目錄 `{HUB_PATH}` 都找不到。")