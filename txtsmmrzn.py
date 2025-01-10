
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import nltk
nltk.download("stopwords")
nltk.download("punkt")  # Add this if needed for tokenization

# Function to calculate sentence similarity
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

# Function to build the similarity matrix
def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

# Function to summarize text
def generate_summary(input_text, top_n=5):
    stop_words = stopwords.words('english')
    sentences = [sentence.split() for sentence in input_text.split(". ") if sentence]

    if len(sentences) == 0:
        return "No valid sentences found."

    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_n = min(top_n, len(ranked_sentence))

    summarize_text = [" ".join(ranked_sentence[i][1]) for i in range(top_n)]

    return ". ".join(summarize_text)

# Streamlit Interface
def main():
    st.title("Text Summarization using NLP")
    st.write("Enter your text below, and click 'Generate Summary'.")

    # Text Input
    input_text = st.text_area("Input Text", height=200, placeholder="Paste your text here...")
    
    # Fixed number of sentences for summary
    top_n = 3  # You can adjust this number based on your preference
    
    # Button to generate summary
    if st.button("Generate Summary"):
        if input_text.strip():
            summary = generate_summary(input_text, top_n)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
