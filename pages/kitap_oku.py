# pages/kitap_oku.py
# Note: Create a 'pages' folder in the same directory as app.py and place this file there.
# Also, install streamlit-pdf-viewer if not already: pip install streamlit-pdf-viewer
# For base64 alternative, see commented code below.

import streamlit as st
import os

st.set_page_config(page_title="Kitap Oku", page_icon="📖", layout="wide")

# Şifre Kontrolü (app.py'deki gibi)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Giriş Yapın")
    st.markdown("Siteye erişim için şifre girin.")
    password = st.text_input("Şifre:", type="password")
    if st.button("Giriş Yap"):
        if password == "yediulya":
            st.session_state.authenticated = True
            st.success("Giriş başarılı! Yeniden yüklüyor...")
            st.rerun()
        else:
            st.error("Yanlış şifre. Tekrar deneyin.")
else:
    pdf_folder = "pdfler"

    # Parse authors and books similar to main app
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    authors = set()
    books_by_author = {}
    for pdf_file in pdf_files:
        base_name = pdf_file.replace(".pdf", "").replace("_", " ")
        if "-" in base_name:
            book_part, author_part = base_name.split("-", 1)
            book_name = book_part.strip().title()
            author = author_part.strip().title() + " Hz.leri"
        else:
            book_name = base_name.title()
            author = "Bilinmeyen Mürşid"
        authors.add(author)
        if author not in books_by_author:
            books_by_author[author] = []
        books_by_author[author].append((book_name, pdf_file))

    authors = sorted(authors)

    st.sidebar.title("Kitap Oku")
    selected_author = st.sidebar.selectbox("Şahsiyet Seçin", authors)

    if selected_author:
        books = sorted(books_by_author[selected_author], key=lambda x: x[0])
        selected_book = st.sidebar.selectbox("Kitap Seçin", [b[0] for b in books])
        
        if selected_book:
            pdf_file = next(p[1] for p in books if p[0] == selected_book)
            file_path = os.path.join(pdf_folder, pdf_file)
            
            st.subheader(f"{selected_book} - {selected_author}")
            
            # Using streamlit-pdf-viewer (recommended)
            try:
                from streamlit_pdf_viewer import pdf_viewer
                pdf_viewer(file_path, width=1000, height=800)
            except ImportError:
                st.error("streamlit-pdf-viewer yüklü değil. Lütfen 'pip install streamlit-pdf-viewer' komutunu çalıştırın.")
            
            # Alternative: Base64 iframe (uncomment if not using pdf_viewer)
            # import base64
            # with open(file_path, "rb") as f:
            #     base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
            # st.markdown(pdf_display, unsafe_allow_html=True)