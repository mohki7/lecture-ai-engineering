# llm.py
import os
import re
import torch
from transformers import pipeline
import streamlit as st
import time
from config import MODEL_NAME
from huggingface_hub import login

# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:

        # アクセストークンを保存
        hf_token = st.secrets["huggingface"]["token"]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

def generate_response(pipe, user_question, num_candidates=3):
    """LLMを使用して質問に対する回答をnum_candidates個生成し、
    その中からベストなものをモデルに選ばせる"""
    if pipe is None:
        return {
            "best_answer": "モデルがロードされていないため、回答を生成できません。",
            "response_time": 0.0,
            "candidates": [],
            "best_index": -1,
            "justification": ""
        }

    start_time = time.time()
    message = [{"role": "user", "content": user_question}]

    # 候補の作成
    candidates = []
    for _ in range(num_candidates):
        outputs = pipe(
            message,
            max_length=512,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            num_return_sequences=1
        )
        answer = ""

        if outputs and isinstance(outputs, list):
            gen = outputs[0].get('generated_text', "")
            if isinstance(gen, list):
                last = gen[-1]
                if last.get("role") == "assistant":
                    answer = last.get("content", "").strip()
            else:
                answer = gen.strip()
        candidates.append(answer)


    # ベストな回答を選択
    eval_prompt = (
        "以下のユーザーの質問と3つの回答候補を読み、最も優れた回答を1つ選んでください。\n"
        "まず 'index:' の後に 1,2,3 の番号を示し、改行して '理由:' の後にその選択理由を述べてください。\n\n"
        f"ユーザーの質問:\n{user_question}\n\n"
        f"回答候補:\n"
        f"1) {candidates[0]}\n\n"
        f"2) {candidates[1]}\n\n"
        f"3) {candidates[2]}"
    )

    eval_outputs = pipe(
        eval_prompt,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,)

    eval_text = ""
    if eval_outputs and isinstance(eval_outputs, list):
        eval_text = eval_outputs[0].get('generated_text', "")
        if isinstance(eval_text, list):
            eval_text = eval_text[-1].get("content", "")
    m_idx = re.search(r'index[:：]?\s*(\d)', eval_text, re.IGNORECASE)
    best_index = int(m_idx.group(1)) - 1 if m_idx else 0
    m_reason = re.search(r'(?:理由|justification)[:：]?\s*(.*)', eval_text, re.DOTALL | re.IGNORECASE)
    justification = m_reason.group(1).strip() if m_reason else eval_text.strip()

    best_answer = candidates[best_index]
    response_time = time.time() - start_time

    return {
        "best_answer": best_answer,
        "response_time": response_time,
        "candidates": candidates,
        "best_index": best_index,
        "justification": justification
    }