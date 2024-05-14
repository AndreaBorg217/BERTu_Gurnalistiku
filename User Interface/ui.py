import streamlit as st
import pandas as pd
from streamlit_tags import st_tags
from transformers import pipeline, BertForQuestionAnswering, AutoTokenizer
import torch
import time

st.set_page_config(layout="wide")

# Initialize session state to only load model and corpus at the start
if "qa" not in st.session_state:
    ## QA PIPELINE
    progress_text = "Loading model..."
    model_pb = st.progress(0, text=progress_text)
    model_name = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_pb.progress(25, text=progress_text)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_pb.progress(50, text=progress_text)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    # model.to_bettertransformer()
    model_pb.progress(75, text=progress_text)
    qa = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)
    model_pb.progress(100, text=progress_text)
    st.session_state.qa = qa


    ## LOADING DATA
    progress_text = "Loading data..."
    data_pb = st.progress(0, text=progress_text)
    corpus = list()
    lines = open('Korpus Ġurnalistiku.txt', 'r', encoding='utf-8').read().splitlines()
    ten_percentile = len(lines) // 10
    corpus = list()
    for i in range(0, len(lines)):
        line = lines[i].strip()
        if (len(line) > 0 and not line.isspace()): corpus.append(line)
        # progress bar is updated every 10% loaded 
        if i % ten_percentile == 0: data_pb.progress(10 * (i//ten_percentile), text=progress_text)
    st.session_state.corpus = corpus
    model_pb.empty() # remove model progress bar when page has loaded
    data_pb.empty() # remove data progress bar when page has loaded


st.title("BERTu Ġurnalistiku")

url = f"https://aclanthology.org/2022.deeplo-1.10/"
st.markdown(
    f"A <a href={url}>Maltese BERT</a> further pre-trained on a corpus of local news articles then fine-tuned for extractive QA using SQuAD",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")
with col1: 
    keywords = st_tags(
        label='#### Enter Keywords:',
        text='Press enter to add more',
        value=[],
        maxtags = -1,
    )

if 'filtered_list' not in st.session_state:
    st.session_state['filtered_list'] = list()


with col2: 
    spacer = st.markdown("<br>", unsafe_allow_html=True) # spacer to center metric
    metric_placeholder = st.empty()
    if len(keywords) > 0:
        if 'keywords' not in st.session_state or st.session_state['keywords'] != keywords:
            st.session_state['keywords'] = keywords
            st.session_state['filtered_list'].clear()
            spacer = st.markdown("<br>", unsafe_allow_html=True) # spacer to center spinner
            with st.spinner('Filtering the data...'):
                for s in st.session_state.corpus:
                    if any(keyword in s for keyword in keywords):
                        st.session_state['filtered_list'].append(s)
                spacer.empty()
            st.session_state.context = "".join(st.session_state['filtered_list'])
        reduction = (len(st.session_state.corpus) - len(st.session_state['filtered_list'])) / len(st.session_state.corpus)
        metric_placeholder.metric(label="Reduced the search space by", value=f"{round(reduction * 100, 2)}%", delta="")
    else:
        st.session_state.context = "".join(st.session_state.corpus)
        spacer = st.markdown("<br>", unsafe_allow_html=True)
        no_keyword_warning = st.markdown("**Not using keywords can cause the chatbot to run very slow**")


st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    default = "Staqsini mistoqsijiet fuq aħbarijiet mill-2013 sal-2023"
    st.session_state.messages = [{"role": "assistant", "content": default, "score": None}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if(message["role"]) == "user":
            st.markdown(message["content"])
        else: 
            ans, confidence = st.columns(2, gap="large")
            with ans: 
                st.markdown(message["content"])
            with confidence:
                if message.get("score"):
                    if message["score"] < 0.5:
                        color = "red"
                    elif 0.5 <= message["score"] <= 0.7:
                        color = "orange"
                    else:
                        color = "green"             
                    st.markdown(f'<span style="color:{color}">{message["score"]}</span>', unsafe_allow_html=True)

# Accept user input
if question := st.chat_input():        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner(): 
            start = time.time()
            response = st.session_state.qa(question, st.session_state.context) # QA PIPELINE
            # response = {"answer": "Din hija t-tweġiba", "score": 0.97207254854}
            end = time.time()
            difference = int(end - start)
            duration = time.strftime("%H:%M:%S", time.gmtime(difference)) 

            
            score = response["score"]
            if round(score, 1) == 0.0:
                answer = "Skużani ma nistax nwieġeb dik il-mistoqsija"
            else:
                answer = response["answer"]
            
        col1, col2 = st.columns(2, gap="large")
        with col1: 
            st.markdown(answer)
        with col2:
            rounded = round(score, 2)
            if rounded < 0.5:
                color = "red"
            elif 0.5 <= rounded <= 0.7:
                color = "orange"
            else:
                color = "green"
            st.markdown(f'<span style="color:{color}">{rounded}</span>', unsafe_allow_html=True)
        st.markdown(f"***{duration}***")
        # Add answer with confidence score to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer, "score": round(score, 2)})