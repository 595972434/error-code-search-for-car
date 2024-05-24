import gc

import numpy as np
import torch
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel


def load_base_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    base_model.eval()  # Set in inference mode. Turns off dropout etc.
    base_model.to("cpu")  # use cpu mode
    return base_model, tokenizer


def load_finetune_weight(model_name, weight_path):
    finetune_model = AutoModel.from_pretrained(model_name)
    weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model_state = finetune_model.embeddings.state_dict()
    model_state['word_embeddings.weight'] = weights.T
    finetune_model.embeddings.load_state_dict(model_state)
    return finetune_model


def load_knowledge_file(filepath):
    reader = PdfReader(filepath)
    texts = np.array([page.extract_text() for page in reader.pages])
    return texts


def get_opensource_embeddings(inputs: list[str], model=None, tokenizer=None):
    """Get embeddings with bge-base-en-v1.5 model.
    """
    model.eval()
    if isinstance(inputs, np.ndarray):
        inputs = inputs.tolist()
    if isinstance(inputs, str):
        inputs = [inputs]

    ems = []
    for i_batch in range(0, len(inputs), 32):
        text_batch = inputs[i_batch:i_batch + 32]
        encoded_input = tokenizer(text_batch, padding=True, truncation=True, return_tensors='pt').to("cpu")
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]

        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy()
        ems.extend(sentence_embeddings)
        gc.collect()
        torch.cuda.empty_cache()
    return np.asarray(ems)


def semantic_search(query_embedding, chunks_embedding, knowledge_texts):
    """Find relevant page from document based on user query"""
    semantic_similarities = (query_embedding @ chunks_embedding.T).flatten() # Flatten from (1, 280) to (280)
    semantic_sim_idx_ordered = np.argsort(semantic_similarities)[::-1] # Reverse the order
    semantic_top_5_matches_idx = semantic_sim_idx_ordered[:5]

    most_similar_page = knowledge_texts[semantic_top_5_matches_idx][0]
    page_no = semantic_top_5_matches_idx[0]
    return most_similar_page, page_no, semantic_top_5_matches_idx


def bm25_tokenizer(sentence):
    """Tokenize sentence to individual words. Goal is to get meaningful words."""
    list_split_by_space = sentence.split(' ')
    list_of_lists_by_newline = [token.split('\n') for token in list_split_by_space]
    corpus = [word for word_list in list_of_lists_by_newline for word in word_list]
    corpus = [word.lower() for word in corpus if len(word) > 2]
    return corpus


def keyword_search(user_query, knowledge_texts):
    tokenized_corpus = [bm25_tokenizer(doc) for doc in knowledge_texts]
    tokenized_query = bm25_tokenizer(user_query)
    bm25 = BM25Okapi(tokenized_corpus)
    bm_scores = bm25.get_scores(tokenized_query)
    keyword_top_5_matches_idx = np.argsort(bm_scores)[::-1][:5]
    most_matching_page = knowledge_texts[keyword_top_5_matches_idx][0]
    page_no = keyword_top_5_matches_idx[0]
    return most_matching_page, page_no, keyword_top_5_matches_idx


def rrf(all_rankings: list[list[int]]):
    """Takes in list of rankings produced by multiple retrieval algorithms,
    and returns newly of ranked and scored items."""
    scores = {} # key is the index and value is the score of that index
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


def answer_question(user_query, knowledge_texts, model, tokenizer):
    chunks_embedding = get_opensource_embeddings(knowledge_texts, model=model, tokenizer=tokenizer)
    query_embedding = get_opensource_embeddings(user_query, model=model, tokenizer=tokenizer)

    relevant_page_semantic, page_no_semantic, top_5_idx_semantic = semantic_search(query_embedding, chunks_embedding, knowledge_texts)
    print(top_5_idx_semantic)
    relevant_page_keyword, page_no_keyword, top_5_idx_keyword = keyword_search(USER_QUERY, knowledge_texts)
    rrf_result = rrf([top_5_idx_semantic, top_5_idx_keyword])
    return rrf_result


if __name__ == "__main__":
    USER_QUERY = "I see code app_w222 on my dashboard what to do?"
    base_model, tokenizer = load_base_model(model_name="BAAI/bge-base-en-v1.5")
    finetune_model = load_finetune_weight(model_name="BAAI/bge-base-en-v1.5", weight_path="./data/embedding_weights_H.pt")
    knowledge_texts = load_knowledge_file(filepath="./data/Owners_Manual_tesla.pdf")

    base_model_answer = answer_question(user_query=USER_QUERY, knowledge_texts=knowledge_texts, model=base_model, tokenizer=tokenizer)
    # finetune_model_answer = answer_question(user_query=USER_QUERY, knowledge_texts=knowledge_texts, model=finetune_model, tokenizer=tokenizer)

    print(base_model_answer)
    # print(finetune_model_answer)