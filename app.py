import gc

import torch
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging
import numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_opensource_embeddings(texts: list[str], is_s2p=False):
    """Get embeddings with bge-base-en-v1.5 model.
    Use argument `is_s2p=True` if `texts` are short query sentences.
    """
    model.eval()
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    if isinstance(texts, str):
        texts = [texts]
    # If the sentences are query sentences, add prefix that was used during training
    if is_s2p:
        texts = ["Represent this sentence for searching relevant passages: " + text for text in texts]

    # Batch the inputs

    ems = []
    for i_batch in range(0, len(texts), 32):
        text_batch = texts[i_batch:i_batch + 32]
        encoded_input = tokenizer(text_batch, padding=True, truncation=True, return_tensors='pt').to("cpu")
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]

        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy()
        ems.extend(sentence_embeddings)
        gc.collect()
        torch.cuda.empty_cache()

    return np.asarray(ems), None


def find_relevant_page_from_document(query, passage_chunks):
    """Find relevant page from document based on user query"""
    embeddings, _ = get_opensource_embeddings(query, is_s2p=True)

    semantic_similarities = (embeddings @ passage_chunks.T).flatten() # Flatten from (1, 280) to (280)
    semantic_sim_idx_ordered = np.argsort(semantic_similarities)[::-1] # Reverse the order
    semantic_top_5_matches_idx = semantic_sim_idx_ordered[:5]

    most_similar_page = texts[semantic_top_5_matches_idx][0]
    page_no = semantic_top_5_matches_idx[0]
    return most_similar_page, page_no, semantic_top_5_matches_idx

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


def bm25_tokenizer(sentence):
    """Tokenize sentence to individual words. Goal is to get meaningful words.
    Stemming or lemmatization could be used here to improve the performance."""
    list_split_by_space = sentence.split(' ')
    list_of_lists_by_newline = [token.split('\n') for token in list_split_by_space]
    corpus = [word for word_list in list_of_lists_by_newline for word in word_list]
    # Remove empty strings, assuming word must have at least 3 characters
    corpus = [word.lower() for word in corpus if len(word) > 2]
    return corpus
if __name__ == '__main__':
    logging.set_verbosity_info()
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')

    model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
    model.eval()  # Set in inference mode. Turns off dropout etc.
    model.to(DEVICE)  # Change which memory model resides

    reader = PdfReader("./data/Owners_Manual_tesla.pdf")
    texts = np.array([page.extract_text() for page in reader.pages])
    # chunks, outs = get_opensource_embeddings(texts)

    # print(chunks)

    # Load the fine-tuned weights
    weights = torch.load("./data/embedding_weights_H.pt", map_location=torch.device('cpu'))
    old_state_dict = model.embeddings.state_dict()
    old_state_dict['word_embeddings.weight'] = weights.T
    model.embeddings.load_state_dict(old_state_dict)

    old_sim_mat = (model.embeddings.word_embeddings.weight.H.T @ model.embeddings.word_embeddings.weight.H).detach().cpu().numpy()
    new_sim_mat = (weights.T @ weights).detach().cpu().numpy()

    USER_QUERY = "I see code app_w304 on my dashboard what to do?"  # User question

    chunks, outs = get_opensource_embeddings(texts)
    relevant_page, page_no, X = find_relevant_page_from_document(USER_QUERY, passage_chunks=chunks)


    print(USER_QUERY)
    print(str(page_no))

    from rank_bm25 import BM25Okapi



    # Initialize BM25
    tokenized_corpus = [bm25_tokenizer(doc) for doc in texts]
    tokenized_query = bm25_tokenizer(USER_QUERY)
    bm25 = BM25Okapi(tokenized_corpus)

    # Get top matches with BM25
    bm_scores = bm25.get_scores(tokenized_query)
    keyword_top_5_matches_idx = np.argsort(bm_scores)[::-1][:5]

    # Get top matches with semantic similarity
    relevant_page, page_no, semantic_top_5_matches_idx = find_relevant_page_from_document(USER_QUERY,
                                                                                          passage_chunks=chunks)

    rrf_score = rrf([keyword_top_5_matches_idx, semantic_top_5_matches_idx])

    print(f"Top 5 matches BM25: {keyword_top_5_matches_idx}")
    print(f"Top 5 matches semantic: {semantic_top_5_matches_idx}")
    print(f"RRF matches and scores: {rrf_score}")