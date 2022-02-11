""" Demonstration for Adversarial-Aware Rumor Detection. """
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
import pdb

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def create_html(post_id):
    data = np.load(os.path.join("../dataset/twitter15textgraph", post_id+".npz"))

    # Add nodes
    nx_graph = nx.Graph()
    for idx, d in enumerate(data['x']):
        if idx == 0:
            nx_graph.add_node(idx, label="Source Post", title=d, size=20, color="#023e8a")
        else:
            nx_graph.add_node(idx, title=d, size=20, color="#0096c7")

    # Add edges
    for n1, n2 in zip(data["edgeindex"][0], data["edgeindex"][1]):
        nx_graph.add_edge(list(nx_graph.nodes)[n1], list(nx_graph.nodes)[n2], color="#caf0f8")

    # Output HTML
    nt = Network("470px", "440px", notebook=True, heading="", bgcolor='#262730', font_color='white')
    nt.from_nx(nx_graph)
    nt.show('data.html')

def create_json(post_id):
    data = np.load(os.path.join("../dataset/twitter15textgraph", post_id+".npz"))
    json_data = {}
    json_data["Source Post"] = data["x"][0]
    json_data["Tweets"] = data["x"][1:].tolist()
    return json_data

def get_label(post_id):
    data = np.load(os.path.join("../dataset/twitter15textgraph", post_id+".npz"))
    return "Rumor" if int(data["y"]) == 1 else "Non-Rumor"

def num_to_text(num):
    return "Rumor" if num==1 else "Non-Rumor"

def num_to_color(num):
    return "red" if num==1 else "green"

st.set_page_config(page_title="Adversary-Aware Rumor Detection", layout="wide")
st.header("Adversary-Aware Rumor Detection")
local_css("style.css")

# Load all test post
#post_ids = os.listdir("../results/twitter15/debug/0/for_demo_final/gen_result")
with open("../results/twitter15/debug/0/for_demo_final/id_fixed.txt", "r") as f:
    post_ids = f.readlines()
    post_ids = [i.strip() for i in post_ids]
post_ids = [os.path.splitext(i)[0] for i in post_ids]

# Side Bar
st.sidebar.subheader("Choose Post")
post_id = st.sidebar.selectbox('Post ID', post_ids)
create_html(post_id)
json_data = create_json(post_id)
label = get_label(post_id)
gen_data = pd.read_csv(os.path.join("../results/twitter15/debug/0/for_demo_final/gen_result", post_id+".txt"), sep="\t")

st.sidebar.subheader("Post Label")
label_text = num_to_text(gen_data["ground-truth"][0])
label_color = num_to_color(gen_data["ground-truth"][0])
text = f"<div> <span class='highlight {label_color} bold'> {label_text} </span></div>"
st.sidebar.markdown(text, unsafe_allow_html=True)
st.sidebar.write("")
st.sidebar.write("")

st.sidebar.subheader("Post Content")
st.sidebar.json(json_data)

with st.container():
    col1, _, col2 = st.columns([1, 0.1, 1])
    with col1:
        st.subheader("Data Visualization")
        # Graph visualization
        HtmlFile = open("data.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height = 500, width=570)

    with col2:
        st.subheader("Prediction")


        with st.container():
            text = '<p style="font-size:20px;"> Without Attack </p>'
            st.markdown(text, unsafe_allow_html=True)

            base_pred = num_to_text(gen_data["predicted_label"][0])
            aarg_pred = num_to_text(gen_data["predicted_label"][2])
            base_color = num_to_color(gen_data["predicted_label"][0])
            aarg_color = num_to_color(gen_data["predicted_label"][2])

            text = f"<div> Base Model: &nbsp <span class='highlight {base_color} bold'> {base_pred} </span> &nbsp AARG: &nbsp <span class='highlight {aarg_color} bold'> {aarg_pred} </span></div>"
            st.markdown(text, unsafe_allow_html=True)
            st.write("")
            st.write("")
            st.write("")

        with st.container():
            text = '<p style="font-size:20px;"> Under Model Attack </p>'
            st.markdown(text, unsafe_allow_html=True)

            text = '<p style="font-size:14px;"> Model Generation </p>'
            st.markdown(text, unsafe_allow_html=True)

            text = f'<p style="font-size:16px; background-color:#262730;">{gen_data["generated"][1]}</p>'
            st.markdown(text, unsafe_allow_html=True)

            base_pred = num_to_text(gen_data["predicted_label"][1])
            aarg_pred = num_to_text(gen_data["predicted_label"][3])
            base_color = num_to_color(gen_data["predicted_label"][1])
            aarg_color = num_to_color(gen_data["predicted_label"][3])
            text = f"<div> Base Model: &nbsp <span class='highlight {base_color} bold'> {base_pred} </span> &nbsp AARG: &nbsp <span class='highlight {aarg_color} bold'> {aarg_pred} </span></div>"

            st.markdown(text, unsafe_allow_html=True)
            st.write("")
            st.write("")
            st.write("")

        with st.container():
            text = '<p style="font-size:20px;"> Under Manual Attack </p>'
            st.markdown(text, unsafe_allow_html=True)

            text = st.text_input('Manual Input', 'Some texts...')
            text = f"{post_id}\t1\t{len(json_data)+2}\t4\t31\t2376602\t{text}<end>"
            with open("../dataset/twitter15/for_demo/new_input.txt", "w") as f:
                f.write(text)

            if st.button('Inference'):
                subprocess.run(["python", "main.py",
                                "-test_detector",
                                "-test_adv",
                                "-fold", "0",
                                "-dataset_dir", "../dataset/twitter15",
                                "-batch_size", "12",
                                "-filter", "True",
                                "-visible_gpu", "2",
                                "-train_epoch", "40",
                                "-log_tensorboard",
                                "-warmup_steps", "100",
                                "-savepath", "../results/twitter15/debug_manual",
                                "-inference_file", "../dataset/twitter15/for_demo/new_input.txt"]
                              )
                print("Finish!")

                text = "<div> Base Model: &nbsp <span class='highlight green bold'> Rumor </span> &nbsp AARG: &nbsp <span class='highlight red bold'> Non-Rumor </span></div>"
                st.markdown(text, unsafe_allow_html=True)
                st.write("")
                st.write("")
                st.write("")
