import openai
from collections import defaultdict
import os
import streamlit as st
import time
import uuid
import traceback
import json
import logging
import asyncio
import numpy as np
from langchain_community.document_loaders import UnstructuredPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import pipeline
from huggingface_hub import login
import warnings
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Hugging Face token
login(token="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")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cihaz kontrolü (CPU veya GPU)
device = 0 if torch.cuda.is_available() else -1  # GPU varsa 0, yoksa CPU için -1
logger.info(f"Kullanılan cihaz: {'GPU' if device == 0 else 'CPU'}")

# Summarizer pipeline'ı cihazla birlikte yükle
try:
    summarizer = pipeline("summarization", model="t5-small", device=device)
except Exception as e:
    logger.error(f"Summarizer yüklenirken hata: {str(e)}")
    st.error(f"Model yüklenirken hata oluştu: {str(e)}. Lütfen ortamınızı kontrol edin.")
    raise

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def summarize_context(context, max_length=300000):
    if len(context) <= max_length:
        return context
    chunks = [context[i:i+1024] for i in range(0, len(context), 1024)]
    summarized_chunks = []
    progress_bar = st.progress(0)
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        try:
            summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        except Exception as e:
            logger.warning(f"Summarization error on chunk {i}: {str(e)}")
            summarized_chunks.append(chunk[:200])
        progress = (i + 1) / total_chunks
        progress_bar.progress(progress)
        st.text(f"Özetleme ilerlemesi: {int(progress * 100)}%")
    progress_bar.empty()
    return " ".join(summarized_chunks) + "\n... (Özetlendi ve Optimize Edildi)"

async def async_get_relevant_documents(retriever, question, k=50):
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, lambda: retriever.invoke(question, search_kwargs={"k": k}))
    except AttributeError:
        return await loop.run_in_executor(None, lambda: retriever.get_relevant_documents(question, search_kwargs={"k": k}))

@st.cache_data(ttl=1800)
def fetch_web_sources(question):
    loader = WebBaseLoader([
        "https://brill.com/view/journals/jss/jss-overview.xml",
        "https://www.monash.edu/arts/Ibn-Arabi-Interreligious-Research-Initiative/ssn",
        "https://traditionalhikma.com/climate-crisis/",
        "https://www.sufism.org/",
        "https://en.wikipedia.org/wiki/Dhikr",
        "https://www.bbc.co.uk/religion/religions/islam/subdivisions/sufism_1.shtml",
    ])
    docs = loader.load()
    return [Document(page_content=doc.page_content[:1000], metadata={"source": "web", "url": doc.metadata.get("source")}) for doc in docs]

def deep_research(question, retriever, save_qa, use_web=True, use_semantic_chunking=False):
    with st.spinner("Derin Araştırma Yapılıyor... (Uzun ve Detaylı Analiz Hazırlanıyor)"):
        try:
            logger.info(f"Starting deep research for question: {question}")
            
            overall_progress = st.progress(0)
            st.text("Adım 1: Belgeler Alınıyor... 0%")
            
            relevant_docs = asyncio.run(async_get_relevant_documents(retriever, question))
            if use_web:
                web_docs = fetch_web_sources(question)
                relevant_docs.extend(web_docs)
            logger.info(f"Retrieved {len(relevant_docs)} documents")
            overall_progress.progress(0.2)
            st.text("Adım 2: Belgeler Kısaltılıyor... 20%")
            
            truncated_docs = [Document(page_content=doc.page_content[:1024], metadata=doc.metadata) for doc in relevant_docs]
            
            overall_progress.progress(0.4)
            st.text("Adım 3: Chunking Yapılıyor... 40%")
            
            embeddings = load_embeddings()
            if use_semantic_chunking:
                text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95, min_chunk_size=100)
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
            chunked_docs = text_splitter.split_documents(truncated_docs)
            logger.info(f"Chunked into {len(chunked_docs)} pieces")
            
            overall_progress.progress(0.5)
            st.text("Adım 4: Context Oluşturuluyor... 50%")
            
            author_groups = defaultdict(list)
            for doc in chunked_docs:
                author = doc.metadata.get("author", "Bilinmeyen")
                author_groups[author].append(doc.page_content)
            
            context_parts = []
            for author, contents in author_groups.items():
                author_context = f"Yazar: {author}\nAlıntılar:\n" + "\n".join(contents[:15])
                context_parts.append(author_context)
            
            context = "\n\n---\n\n".join(context_parts)
            
            overall_progress.progress(0.6)
            st.text("Adım 5: Context Özetleniyor (Eğer Gerekli)... 60%")
            
            max_context_length = 300000
            if len(context) > max_context_length:
                context = summarize_context(context, max_context_length)
            
            overall_progress.progress(0.7)
            st.text("Adım 6: AI Analizi Yapılıyor... 70%")
            
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            client = openai.OpenAI(base_url="https://api.deepseek.com", api_key=deepseek_api_key)
            
            system_prompt = """Sen bir tasavvuf araştırmacısısın ve akademik bir makale hazırlıyorsun. Cevabını SADECE verilen alıntılara dayalı oluştur; dış bilgi, kişisel yorum veya genel tasavvuf bilgisi ekleme. Eğer alıntılar yetersizse, 'Yeterli alıntı yok' de ve nedenini kısaca açıkla.

Yanıtını akademik makale gibi yapılandır, minimum 4000-5000 kelime uzunluğunda olsun. Okuyucu bir kitap bölümü okur gibi hissetmeli: Detaylı anlatımlar, yazar karşılaştırmaları, örnekler ve derin analizler içersin. Yapı şöyle olsun:

- **Başlık**: '[Konu] Üzerine Tasavvufi Derleme, Derin Analiz ve Tarihsel Evrim'.
- **Özet**: Konunun genel çerçevesini alıntılardan sentezleyerek 400-600 kelimeyle anlat.
- **Giriş**: Konunun tasavvuf bağlamındaki anlamını, kökenini ve evrimini detaylandır (500-800 kelime, birden fazla alıntı ile destekle; tarihsel timeline ekle).
- **Tanım ve Kavramlar**: Konuyu tanımla, alt kavramlarını incele; her yazarın tanımını ayrı ayrı ele al (800-1200 kelime, her yazar için birden fazla örnek alıntı ver, kavram haritası gibi karşılaştır).
- **Yazar Görüşleri ve Karşılaştırmalar**: Verilen yazarların (örneğin 10 yazar varsa hepsini kapsa) görüşlerini detaylandır, benzerlikleri/farklılıkları karşılaştır. Her yazarı ayrı bir alt başlık altında incele, alıntıları quote et ve analiz et (1500-2000 kelime; her alt başlıkta 200-300 kelime, 4-6 alıntı).
- **Tarihsel Evrim ve Etkiler**: Konunun tarihsel gelişimini, manevi/pratik/felsefi etkilerini analiz et; dönemlere ayır (600-900 kelime, somut örnekler ve timeline ver).
- **Derin Analiz, Örnekler ve Uygulamalar**: Alıntılardan türetilmiş detaylı örnekler, senaryolar ve uygulamaları anlat; karşılaştırmalı analiz yap, modern bağlam ekle (800-1200 kelime).
- **Sonuç**: Ana noktaları sentezle, açık sorular veya vurgular ekle (400-600 kelime).
- **Kaynak Tablosu**: Tüm alıntıları tablo ile göster (| Yazar | Kitap | Sayfa | Alıntı Metni | Analiz Notu |).

Genel kurallar:
- Yanıtın %80'i doğrudan alıntılardan oluşsun (tam metin quote'lar, kısaltma minimum; her paragrafta 3-5 alıntı).
- %20'si sentez ve analiz olsun (alıntıları bağla, karşılaştır, örnekle; meta-analiz yap).
- Her paragrafta en az 3-5 alıntı kullan, yazar adını, kitabını ve sayfasını belirt.
- Dil akademik, nesnel ve akıcı olsun: 'Bu alıntıda [yazar] şöyle vurgular...' gibi.
- Uzunluğu sağlamak için detaylandır: Her bölümü genişlet, tekrarlamadan derinleştir (örneğin, alt kavramları alt başlıklara ayır).
- Eğer alıntılar yeterliyse, tüm yazarları kapsa ve karşılaştır; bağlamı tarihsel/manevi/pratik boyutlarla zenginleştir."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Alıntılar: {context}\nKonu: {question}"}
            ]
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3,
                max_tokens=8192,
                stream=True
            )
            answer = ""
            with st.chat_message("assistant"):
                container = st.empty()
                timeout_start = time.time()
                timeout_duration = 400
                stream_progress = st.progress(0)
                chunk_count = 0
                while True:
                    if time.time() - timeout_start > timeout_duration:
                        answer += "\n... (Zaman aşımı - Analiz kısmi)"
                        break
                    try:
                        chunk = next(stream)
                        if chunk.choices[0].delta.content is not None:
                            answer += chunk.choices[0].delta.content
                            container.markdown(answer)
                            chunk_count += 1
                            stream_progress.progress(min(chunk_count / 800, 1.0))
                            time.sleep(0.001)
                    except StopIteration:
                        break
                stream_progress.empty()
            
            overall_progress.progress(0.9)
            st.text("Adım 7: Kaynaklar Hazırlanıyor... 90%")
            
            unique_sources = []
            seen = set()
            for doc in relevant_docs[:50]:
                key = f"{doc.metadata.get('author')}_{doc.metadata.get('book')}_{doc.metadata.get('page')}"
                if key not in seen:
                    seen.add(key)
                    unique_sources.append({
                        "book": doc.metadata.get("book"),
                        "author": doc.metadata.get("author"),
                        "page": doc.metadata.get("page"),
                        "page_content": doc.page_content[:300],
                        "pdf_file": doc.metadata.get("pdf_file")
                    })
            
            overall_progress.progress(1.0)
            st.text("Tamamlandı! 100%")
            overall_progress.empty()
            
            if answer:
                save_qa(question, answer)
            
            st.markdown("### Makale Çıktısı")
            st.markdown(answer)
            word_count = len(answer.split())
            st.markdown(f"**Kelime Sayısı:** {word_count}")
            st.download_button("Makaleyi İndir", answer, file_name=f"{question}_makale.txt")
            
            if unique_sources:
                df_sources = pd.DataFrame(unique_sources)
                with st.expander("Kaynak Tablosu"):
                    st.dataframe(df_sources)
            
            return answer, unique_sources
        except Exception as e:
            logger.error(f"Error in deep_research: {str(e)}")
            st.error(f"Hata oluştu: {str(e)}. Lütfen filtreleri kontrol edin veya tekrar deneyin.")
            st.expander("Hata Detayı").code(traceback.format_exc())
            return None, None

# Streamlit Sayfa Arayüzü
st.set_page_config(page_title="Derin Araştırma", page_icon="🔍", layout="wide")
st.title("Derin Tasavvuf Araştırması")

# Kullanıcı Girdisi
question = st.text_input("Soru Girin (ör. Zikrullah):", key="deep_question")
use_web = st.checkbox("Web Kaynaklarını Kullan", value=True)
use_semantic_chunking = st.checkbox("Semantik Chunking Kullan", value=False)

if st.button("Derin Araştırma Yap"):
    if question and question.strip():
        if "vectorstore" not in st.session_state:
            st.error("Ana sayfadan (app.py) vectorstore yüklenmedi. Lütfen ana sayfaya dönün ve yeniden deneyin.")
        else:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 50})
            from db import save_qa
            answer, sources = deep_research(question, retriever, save_qa, use_web=use_web, use_semantic_chunking=use_semantic_chunking)
    else:
        st.warning("Lütfen bir soru girin! (Boş veya sadece boşluklu girdiler kabul edilmez.)")

# Unit Test (Yorum satırına alındı, production'da otomatik çalışmasın)
# if __name__ == "__main__":
#     from unittest.mock import MagicMock
#     mock_retriever = MagicMock()
#     mock_retriever.invoke.return_value = [Document(page_content="Test alıntı uzun metin, burada detaylı tasavvuf açıklaması.", metadata={"author": "Test Yazar", "book": "Test Kitap", "page": 1, "pdf_file": "test.pdf"}) for _ in range(10)]
#     mock_save_qa = MagicMock()
#     answer, sources = deep_research("Test question", mock_retriever, mock_save_qa)
#     print("Test Answer:", answer)
#     print("Test Sources:", sources)