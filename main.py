import gc
import numpy as np
import streamlit as st
import torch
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from streamlit_pdf_viewer import pdf_viewer
from transformers import AutoTokenizer, AutoModel

st.set_page_config(
    page_title="Error Code Search App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Set the CUDA device to GPU 1
if torch.cuda.is_available():
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

@st.cache_resource
def load_base_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path + '/' + 'tokenizer')
    base_model = AutoModel.from_pretrained(model_path + '/' + 'model')

    base_model.eval()  # Set in inference mode. Turns off dropout etc.
    base_model.to(device)  # use GPU 1 or CPU mode
    return base_model, tokenizer


@st.cache_resource
def load_finetune_weight(model_path, weight_path):
    finetune_model = AutoModel.from_pretrained(model_path + '/' + 'model')
    weights = torch.load(weight_path, map_location=device)
    model_state = finetune_model.embeddings.state_dict()
    model_state['word_embeddings.weight'] = weights.T
    finetune_model.embeddings.load_state_dict(model_state)
    finetune_model.to(device)  # use GPU 1 or CPU mode
    return finetune_model


@st.cache_data
def load_knowledge_file(filepath):
    reader = PdfReader(filepath)
    texts = np.array([page.extract_text() for page in reader.pages])
    return texts


def get_opensource_embeddings(inputs: list[str], model=None, tokenizer=None):
    """Get embeddings with bge-base-en-v1.5 model."""
    model.eval()
    if isinstance(inputs, np.ndarray):
        inputs = inputs.tolist()
    if isinstance(inputs, str):
        inputs = [inputs]

    ems = []
    for i_batch in range(0, len(inputs), 32):
        text_batch = inputs[i_batch:i_batch + 32]
        encoded_input = tokenizer(text_batch, padding=True, truncation=True, return_tensors='pt').to(device)

        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy()

        ems.extend(sentence_embeddings)
        gc.collect()
        torch.cuda.empty_cache()
    return np.asarray(ems)


@st.cache_data
def semantic_search(query_embedding, chunks_embedding, knowledge_texts):
    """Find relevant page from document based on user query"""
    semantic_similarities = (query_embedding @ chunks_embedding.T).flatten()  # Flatten from (1, 280) to (280)
    semantic_sim_idx_ordered = np.argsort(semantic_similarities)[::-1]  # Reverse the order
    semantic_top_5_matches_idx = semantic_sim_idx_ordered[:5]

    most_similar_page = knowledge_texts[semantic_top_5_matches_idx][0]
    page_no = semantic_top_5_matches_idx[0]
    return most_similar_page, page_no, semantic_top_5_matches_idx


@st.cache_data
def bm25_tokenizer(sentence):
    """Tokenize sentence to individual words. Goal is to get meaningful words."""
    list_split_by_space = sentence.split(' ')
    list_of_lists_by_newline = [token.split('\n') for token in list_split_by_space]
    corpus = [word for word_list in list_of_lists_by_newline for word in word_list]
    corpus = [word.lower() for word in corpus if len(word) > 2]
    return corpus


@st.cache_data
def keyword_search(user_query, knowledge_texts):
    tokenized_corpus = [bm25_tokenizer(doc) for doc in knowledge_texts]
    tokenized_query = bm25_tokenizer(user_query)
    bm25 = BM25Okapi(tokenized_corpus)
    bm_scores = bm25.get_scores(tokenized_query)
    keyword_top_5_matches_idx = np.argsort(bm_scores)[::-1][:5]
    most_matching_page = knowledge_texts[keyword_top_5_matches_idx][0]
    page_no = keyword_top_5_matches_idx[0]
    return most_matching_page, page_no, keyword_top_5_matches_idx


@st.cache_data
def rrf(all_rankings: list[list[int]]):
    """Takes in list of rankings produced by multiple retrieval algorithms,
    and returns newly of ranked and scored items."""
    scores = {}  # key is the index and value is the score of that index
    # 1. Take every retrieval algorithm ranking
    for algorithm_ranks in all_rankings:
        # 2. For each ranking, take the index and the ranked position
        for rank, idx in enumerate(algorithm_ranks):
            # 3. Calculate the score and add it to the index
            if idx in scores:
                scores[idx] += 1 / (60 + rank)
            else:
                scores[idx] = 1 / (60 + rank)

    # 4. Sort the indices based on accumulated scores
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_scores

def get_openai_response(messages):
    """Get OpenAI response through REST API.
    REST API is preferred, as the Python client can introduce breaking
    changes whenever."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages)
    answer = response.choices[0].message.content

    return answer
def get_openai_rag_response(query: str, sources: list[str]):
    """Get RAG response based on query and list of strings as sources."""
    messages = [{'role': 'system', 'content': f"""
    You are chatbot for Tesla customer support. You must answer based on the relevant sources, delimited by three backticks:

    Sources: ```{sources}```


    Answer only based on the sources.
    """}]
    messages += [{"role": "user", "content": query}]
    answer = get_openai_response(messages)

    return answer


def answer_question(user_query, knowledge_texts, model, tokenizer, only_semantic=False):
    chunks_embedding = get_opensource_embeddings(knowledge_texts, model=model, tokenizer=tokenizer)
    query_embedding = get_opensource_embeddings(user_query, model=model, tokenizer=tokenizer)

    relevant_page_semantic, page_no_semantic, top_5_idx_semantic = semantic_search(query_embedding, chunks_embedding, knowledge_texts)
    if only_semantic:
        response = get_openai_rag_response(user_query, [relevant_page_semantic])
        return [int(result) + 1 for result in top_5_idx_semantic], int(page_no_semantic) + 1, response
    else:
        relevant_page_keyword, page_no_keyword, top_5_idx_keyword = keyword_search(user_query, knowledge_texts)
        rrf_result = rrf([top_5_idx_semantic, top_5_idx_keyword])
        the_best_page_no = int(rrf_result[0][0]) + 1
        print(knowledge_texts[the_best_page_no-1])
        response = get_openai_rag_response(user_query, [knowledge_texts[the_best_page_no-1]])
        return [result[0] + 1 for result in rrf_result[:5]], the_best_page_no, response

base_model, tokenizer = load_base_model(model_path="./model")
finetune_model = load_finetune_weight(model_path="./model",
                                      weight_path="./data/embedding_weights_H.pt")
knowledge_texts = load_knowledge_file(filepath="./data/Owners_Manual_tesla.pdf")
client = OpenAI(
    ## TODO add gm api key
    api_key="gm key",
    organization="gluon-meson",
    base_url="http://bj.private.gluon-meson.tech:11000/model-proxy/v1")

print("Loaded knowledge texts !!!!!!!!!")

if 'finetune_model_page' not in st.session_state:
    st.session_state.finetune_model_page = 1

if 'base_model_page' not in st.session_state:
    st.session_state.base_model_page = 1

if 'semantic_page' not in st.session_state:
    st.session_state.semantic_page = 1

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.container(height=600):
    col1, col2 = st.columns([3,1])

    with col1:
        st.title("Enterprise Search Powered by GenAI")
        messages = st.container(height=300)
        for message in st.session_state.messages:
            with messages.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Input a question about Tesla error code?",):
            messages.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # # Display user message in chat message container
            # with st.chat_message("user"):
            #     st.markdown(prompt)
            print("calling finetune model !!!")
            finetune_model_matching_page_list, finetune_model_page_no,finetune_model_answer = answer_question(user_query=prompt,
                                                                                        knowledge_texts=knowledge_texts,

                                                                                        model=finetune_model,
                                                                                        tokenizer=tokenizer)
            print("calling base model !!!")
            base_model_matching_page_list, base_model_page_no,base_model_answer = answer_question(user_query=prompt,
                                                                                knowledge_texts=knowledge_texts,

                                                                                model=base_model,
                                                                                tokenizer=tokenizer)
            print("calling only semantic !!!")
            only_semantic_matching_page_list, only_semantic_page_no,only_semantic_answer = answer_question(user_query=prompt,
                                                                                knowledge_texts=knowledge_texts,
                                                                                model=base_model,
                                                                                tokenizer=tokenizer,
                                                                                only_semantic=True)

            st.session_state.finetune_model_page = int(finetune_model_page_no)
            st.session_state.base_model_page = int(base_model_page_no)
            st.session_state.semantic_page = int(only_semantic_page_no)
            with messages.chat_message("assistant"):
                finetune_model_response = f'''**Finetune Model**: The best matching page list is {','.join(str(page) for page in finetune_model_matching_page_list)}  
                               **Answer**: {finetune_model_answer}'''

                st.markdown(finetune_model_response)
                st.session_state.messages.append({"role": "assistant", "content": finetune_model_response})

            with messages.chat_message("assistant"):
                base_model_response = f'''**Base Model**: The best matching page list is {','.join(str(page) for page in base_model_matching_page_list)}   
                               **Answer**: {base_model_answer}'''

                st.markdown(base_model_response)
                st.session_state.messages.append({"role": "assistant", "content": base_model_response})

            with messages.chat_message("assistant"):
                semantic_response = f'''**Only Semantic**: The best matching page list is {','.join(str(page) for page in only_semantic_matching_page_list)}
                                        **Answer**: {only_semantic_answer}'''

                st.markdown(semantic_response)
                st.session_state.messages.append({"role": "assistant", "content": semantic_response})
    with col2:
        text = "Error code used to finetune model"
        highlighted_text = f"<h2 style='color: gray'>{text}</h2>"

        st.markdown(highlighted_text, unsafe_allow_html=True)
        error_code = "APP_w009, APP_w048, APP_w207, APP_w218, APP_w221, APP_w222, APP_w224, APP_w304, BMS_a066, BMS_a067, BMS_a068, BMS_a069, CC_a001, CC_a002, CC_a003, CC_a004, CC_a005, CC_a006, CC_a007, CC_a008, CC_a009, CC_a010, CC_a011, CC_a012, CC_a013, CC_a014, CC_a015, CC_a016, CC_a017, CC_a018, CC_a019, CC_a020, CC_a021, CC_a022, CC_a023, CC_a024, CC_a025, CC_a026, CC_a027, CC_a028, CC_a029, CC_a030, CC_a041, CC_a043, CP_a004, CP_a010, CP_a043, CP_a046, CP_a051, CP_a053, CP_a054, CP_a055, CP_a056, CP_a058, CP_a066, CP_a078, CP_a079, CP_a101, CP_a102, CP_a143, CP_a151, DI_a138, DI_a166, DI_a175, DI_a184, DI_a185, DI_a190, DI_a245, DIF_a251, DIR_a251, EPBL_a195, EPBR_a195, ESP_a118, PCS_a016, PCS_a017, PCS_a019, PCS_a032, PCS_a052, PCS_a053, PCS_a054, PCS_a073, PCS_a090, PM_a092, PMR_a092, PMF_a092, UI_a004, UI_a006, UI_a013, UI_a014, UI_a137, UMC_a001, UMC_a002, UMC_a004, UMC_a005, UMC_a007, UMC_a008, UMC_a009, UMC_a010, UMC_a011, UMC_a012, UMC_a013, UMC_a014, UMC_a015, UMC_a016, UMC_a017, UMC_a018, UMC_a019, VCFRONT_a180, VCFRONT_a182, VCFRONT_a191, VCFRONT_a192, VCFRONT_a216, VCFRONT_a220, VCFRONT_a402, VCFRONT_a496, VCFRONT_a592, VCFRONT_a593, VCFRONT_a596, VCFRONT_a597, VCSEC_a221, VCSEC_a228"
        text_format = f"<span style='font-style: italic'>{error_code}</span>"
        st.markdown(text_format, unsafe_allow_html=True)
with st.container():
    show_button = st.button("Show PDF page", type="primary")
    if show_button:
        finetune_bm25_col, base_bm25_col, only_semantic_col = st.columns([1,1,1])
        print(st.session_state.finetune_model_page)
        print(st.session_state.base_model_page)
        print(st.session_state.semantic_page)
        with finetune_bm25_col:
            st.title("Finetune + BM25")
            pdf_viewer(key="finetune model", input="./data/Owners_Manual_tesla.pdf", height=500, width=700,
                       pages_to_render=[st.session_state.finetune_model_page])
        with base_bm25_col:
            st.title("Base + BM25")
            pdf_viewer(key="base model", input="./data/Owners_Manual_tesla.pdf", height=500, width=700,
                       pages_to_render=[st.session_state.base_model_page])
        with only_semantic_col:
            st.title("Only Semantic")
            pdf_viewer(key="semantic", input="./data/Owners_Manual_tesla.pdf", height=500, width=700,
                       pages_to_render=[st.session_state.semantic_page])
