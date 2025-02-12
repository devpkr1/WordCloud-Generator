import streamlit as st
import preprocessing_pipeline as pp

def main():
    st.title("WordCloud Generator")
    text = st.text_area("Enter text for WordCloud")

    if st.button("Generate") and text:

        processed_text = pp.preprocess_text(text)
        
        if not processed_text:
            st.warning("No valid words found after preprocessing. Please enter more text or check for punctuation.")

            return

        st.subheader("Unigram WordCloud")
        unigram_wc = pp.generate_wordcloud(processed_text, ngram=1)
        if unigram_wc:
           st.image(unigram_wc.to_array(), use_container_width=True)

        st.subheader("Bigram WordCloud")
        bigram_wc = pp.generate_wordcloud(processed_text, ngram=2)
        if bigram_wc:
            st.image(bigram_wc.to_array(), use_container_width=True)

if __name__ == "__main__":
    main()