import streamlit as st
from pypdf import PdfReader


def main():
    st.title("PDF Reader Example")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Load PDF
        reader = PdfReader(uploaded_file)
        # Extract text from the first page
        page = reader.pages[0]
        text = page.extract_text()
        st.text_area("Extracted Text", text)


if __name__ == "__main__":
    main()
