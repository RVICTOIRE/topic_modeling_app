import streamlit as st
import pandas as pd
import re
import nltk
import numpy as np
import tweepy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis.gensim_models
import pyLDAvis
import streamlit.components.v1 as components

nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(page_title="Topic Modeling Sénégal", layout="wide")

# ---- CONFIGURATION TWITTER ----
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une étape :", ["📥 Collecte des Tweets", "🧠 Analyse des sujets", "📊 Visualisation"])

# --- Initialisation de session ---
if "tweets_df" not in st.session_state:
    st.session_state.tweets_df = None
if "lda_model" not in st.session_state:
    st.session_state.lda_model = None
if "dictionary" not in st.session_state:
    st.session_state.dictionary = None
if "corpus" not in st.session_state:
    st.session_state.corpus = None
if "texts_tokenized" not in st.session_state:
    st.session_state.texts_tokenized = None

# ---- 1. COLLECTE DES TWEETS ----
if page == "📥 Collecte des Tweets":
    st.header("📥 Collecte des Tweets")

    collecte_mode = st.radio("Choisissez le mode de collecte :", ["🌐 Collecte via Tweepy", "📂 Charger un fichier CSV"])

    if collecte_mode == "🌐 Collecte via Tweepy":
        BEARER_TOKEN = st.text_input("Entrez votre Bearer Token Twitter :", type="password")

        query = st.text_input("Entrez votre requête (ex: Sénégal OR Sonko OR Macky)", "Sénégal OR Sonko OR Macky OR éducation")

        max_tweets = st.slider("Nombre de tweets à collecter :", 10, 100, 50)

        if st.button("Lancer la collecte avec Tweepy"):
            if BEARER_TOKEN == "":
                st.error("Veuillez entrer un Bearer Token valide.")
            else:
                client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

                tweets = client.search_recent_tweets(query=query, max_results=max_tweets, tweet_fields=["text", "lang"])

                tweet_list = []
                for tweet in tweets.data:
                    if tweet.lang == 'fr':
                        tweet_list.append(tweet.text)

                df = pd.DataFrame(tweet_list, columns=["tweet"])
                st.session_state.tweets_df = df
                st.success(f"{len(df)} tweets collectés et enregistrés en mémoire.")
                st.dataframe(df)

    elif collecte_mode == "📂 Charger un fichier CSV":
        uploaded_file = st.file_uploader("Chargez votre fichier CSV contenant les tweets", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Vérifier si la colonne existe
            st.write("Colonnes disponibles :", df.columns.tolist())

            tweet_column = st.selectbox("Sélectionnez la colonne contenant les tweets :", df.columns)

            st.session_state.tweets_df = df[[tweet_column]].rename(columns={tweet_column: 'tweet'})
            st.success("Fichier chargé et colonne sélectionnée avec succès.")
            st.dataframe(st.session_state.tweets_df)



# ---- 2. ANALYSE DES SUJETS ----
elif page == "🧠 Analyse des sujets":
    st.header("🧠 Analyse des Sujets - Topic Modeling")

    if st.session_state.tweets_df is None:
        st.warning("Veuillez d'abord collecter les tweets dans la section Collecte.")
    else:
        df = st.session_state.tweets_df
        texts = df['tweet'].dropna().tolist()

        # Nettoyage
        stop_words = set(stopwords.words('french'))
        lemmatizer = WordNetLemmatizer()

        def clean_tweet(tweet):
            tweet = tweet.lower()
            tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet)
            tweet = re.sub(r'@\w+', '', tweet)
            tweet = re.sub(r'#', '', tweet)
            tweet = re.sub(r'rt\s+', '', tweet)
            tweet = re.sub(r'[^a-zA-Zàâçéèêëîïôûùüÿñæœ\s]', '', tweet)
            tweet = re.sub(r'\s+', ' ', tweet)
            return tweet.strip()

        def preprocess(text):
            tokens = text.split()
            tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
            return tokens

        cleaned_texts = [clean_tweet(t) for t in texts]
        texts_tokenized = [preprocess(t) for t in cleaned_texts]

        st.session_state.texts_tokenized = texts_tokenized

        # LDA
        dictionary = corpora.Dictionary(texts_tokenized)
        corpus = [dictionary.doc2bow(text) for text in texts_tokenized]

        n_topics = st.slider("Nombre de topics :", 2, 10, 5)

        lda_model = models.LdaModel(corpus=corpus,
                                    id2word=dictionary,
                                    num_topics=n_topics,
                                    passes=10,
                                    alpha='auto',
                                    random_state=42)

        # Sauvegarde dans session_state
        st.session_state.lda_model = lda_model
        st.session_state.dictionary = dictionary
        st.session_state.corpus = corpus
        st.session_state.n_topics = n_topics  

        st.subheader("📄 Topics extraits :")
        for idx, topic in lda_model.print_topics(-1):
            st.markdown(f"**Topic {idx+1}** : {topic}")


# ---- 3. VISUALISATION ----
elif page == "📊 Visualisation":
    st.header("📊 Visualisation des Topics")

    if st.session_state.lda_model is None:
        st.warning("Veuillez d'abord exécuter l'analyse des sujets.")
    else:
        lda_model = st.session_state.lda_model
        corpus = st.session_state.corpus
        dictionary = st.session_state.dictionary
        texts_tokenized = st.session_state.texts_tokenized
        n_topics = st.session_state.n_topics

        # 4. Visualisation WordCloud
        st.subheader("☁️ Nuage de mots par topic")

        for idx, topic in lda_model.show_topics(num_topics=n_topics, num_words=20, formatted=False):
            dict_words = dict(topic)
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab10').generate_from_frequencies(dict_words)
            plt.figure(figsize=(8,4))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

        # 5. Histogramme des topics
        st.subheader("📈 Répartition des topics dans le corpus")

        topic_counts = np.zeros(n_topics)
        for bow in corpus:
            topic_probs = lda_model.get_document_topics(bow)
            dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
            topic_counts[dominant_topic] += 1

        plt.figure(figsize=(8,5))
        plt.bar(range(1, n_topics+1), topic_counts, color='skyblue')
        plt.xlabel("Topic")
        plt.ylabel("Nombre de tweets")
        plt.title("Distribution des topics dans le corpus")
        st.pyplot(plt)

        # 6. Visualisation pyLDAvis
        st.subheader("🔍 Visualisation interactive avec pyLDAvis")

        try:
            import pyLDAvis.gensim_models
            import pyLDAvis
            vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
            html_string = pyLDAvis.prepared_data_to_html(vis)
            components.html(html_string, height=800, scrolling=True)
        except Exception as e:
            st.warning("Visualisation pyLDAvis désactivée sur Streamlit Cloud. Utilisez WordCloud et Histogramme.")


