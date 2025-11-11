# Temel ve Streamlit Kütüphaneleri
import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv # .env dosyasından environment değişkenlerini okumak için
import json                 # JSON dosyalarını (csv_configs.json) okumak için
import uuid                 # Her kullanıcı oturumu için benzersiz bir ID oluşturmak için

# Veri Yükleme ve İşleme
from PyPDF2 import PdfReader # PDF dosyalarından metin okumak için

# Google ve Gemini Entegrasyonu
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# LangChain Kütüphaneleri (RAG Mimarisi için)
from langchain.text_splitter import RecursiveCharacterTextSplitter # Metinleri parçalara ayırmak için
from langchain_community.vectorstores import Chroma              # Vektör veritabanı için
from langchain.chains.question_answering import load_qa_chain    # Soru-cevap zinciri oluşturmak için
from langchain.prompts import PromptTemplate                       # LLM'e gönderilecek talimat şablonu için

# IT Sınıflandırma Modeli (Hugging Face)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch # Transformers kütüphanesinin arka planda kullandığı temel kütüphane

# Google Sheets Entegrasyonu
import gspread
from google.oauth2.service_account import Credentials

# Uygulamanın ana mantığını oluşturan, tekrar tekrar kullanılan fonksiyonlar.
@st.cache_data 
def load_csv_configs():
    """csv_configs.json dosyasını yükler."""
    try:
        with open("csv_configs.json", "r", encoding="utf-8") as f:
            return json.load(f)
    # Eğer konfigürasyon dosyası bulunamazsa, her şeyi genel bir formatla işleyecek
    # varsayılan bir şablon döndürür. Bu, uygulamanın çökmesini engeller.
    except FileNotFoundError:
        return {"_default": {"template": "{__ALL_COLUMNS__}."}}

def process_row_with_config(row: pd.Series, file_name: str, configs: dict) -> str:
    """
    Bir CSV satırını, yüklenen konfigürasyondaki şablona göre anlamlı bir cümleye dönüştürür.
    Bu, "Kod Yerine Konfigürasyon" prensibinin temelidir.
    """
    config = configs.get(file_name, configs["_default"])
    template = config["template"]
    row_dict = row.to_dict()
    # Şablondaki özel komutları işle
    if "{__COLUMN_0__}" in template:
        template = template.replace("{__COLUMN_0__}", str(row.iloc[0]))
    if "{__ALL_COLUMNS__}" in template:
        all_cols_text = ", ".join([f"{k}: {v}" for k, v in row_dict.items() if pd.notna(v)])
        template = template.replace("{__ALL_COLUMNS__}", all_cols_text)

    try:
        # Şablondaki {SütunAdı} gibi yer tutucuları, satırdaki gerçek değerlerle doldur.
        return template.format_map({k: v if pd.notna(v) else "" for k, v in row_dict.items()})
    except KeyError as e:
        print(f"Şablonda hata: CSV'de olmayan bir sütun isteniyor -> {e}")
        return ""

def get_documents_text():
    """Tüm PDF ve CSV dosyalarını okuyup, tek bir metin bloğuna dönüştürür."""
    csv_configs = load_csv_configs()
    docs_path = "./data/"
    if not os.path.exists(docs_path): return ""
    
    file_paths = [os.path.join(docs_path, f) for f in os.listdir(docs_path)]
    raw_text = ""

    # PDF'leri işle
    for path in filter(lambda p: p.endswith('.pdf'), file_paths):
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                raw_text += page.extract_text() or ""
        except Exception as e:
            print(f"PDF okuma hatası {path}: {e}")

    # CSV'leri işle
    for path in filter(lambda p: p.endswith('.csv'), file_paths):
        try:
            file_name = os.path.basename(path)
            df = pd.read_csv(path, sep=';', on_bad_lines='skip')
            for index, row in df.iterrows():
                sentence = process_row_with_config(row, file_name, csv_configs)
                raw_text += sentence + "\n"
        except Exception as e:
            print(f"CSV okuma hatası {path}: {e}")
            
    return raw_text

def get_text_chunks(text): #"""Uzun metinleri, LLM'in işleyebileceği parçalara böler."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks): #"""Metin parçalarından vektör veritabanını oluşturur veya yükler."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store_path = "chroma_db"
    if os.path.exists(vector_store_path):
        return Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    else:
        vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory=vector_store_path)
        vector_store.persist()
        return vector_store

def get_conversational_chain(): 
    """LLM ile RAG konuşma zincirini oluşturur (Hafıza ile)."""
    prompt_template = """
    Sen X-Company'nin yardımsever bir kurumsal asistanısın. Cevaplarını, sana verilen bağlam (context) ve önceki konuşma geçmişini (chat history) dikkate alarak oluştur. 
    Eğer bir soru önceki konuşmayla ilgiliyse, bu bağlantıyı kurarak cevap ver. 
    Cevap, verilen bağlamda bulunmuyorsa, "Bu konuda bilgi sahibi değilim." de. Kendi bilgini kullanma.

    Context:\n{context}\n
    Chat History:\n{chat_history}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def handle_user_input(user_question, vector_store, chat_history): 
    """Kullanıcının sorusunu ve sohbet geçmişini RAG pipeline'ından geçirerek cevap üretir."""
    docs = vector_store.similarity_search(user_question, k=5)
    chain = get_conversational_chain()
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:-1]])
    response = chain({"input_documents": docs, "chat_history": history_text, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def predict_it_ticket_category(text, tokenizer, model): 
    """
    Verilen metnin bir IT sorunu olup olmadığını ve hangi kategoriye ait olduğunu
    eğitilmiş Transformer modeli ile tahmin eder.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).flatten()
    predicted_class_id = probabilities.argmax().item()
    confidence = probabilities[predicted_class_id].item()
    return model.config.id2label[predicted_class_id], confidence

def save_ticket_to_gsheet(konu, detay, aciliyet, kategori, user_email):
    """Doldurulan IT Destek Formu bilgilerini Google Sheets'e yeni bir satır olarak kaydeder."""
    try:
        scopes = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open("X-Company IT Talepleri").sheet1
        new_row = [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),user_email, kategori, konu, detay, aciliyet, "Yeni Talep"]
        sheet.append_row(new_row)
        return True
    except Exception as e:
        st.error(f"Veritabanına yazılırken bir hata oluştu: {e}")
        return False

# ÖNBELLEĞE ALINACAK FONKSİYONLAR

@st.cache_resource
def load_rag_vector_store():
    """Tüm veri kaynaklarını yükler, işler ve vektör veritabanını hazırlar."""
    print("RAG Veritabanı yükleniyor...")
    raw_text = get_documents_text()
    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(text_chunks)
    return vector_store

@st.cache_resource
def load_classification_model():
    """IT sınıflandırma modelini ve tokenizer'ı Hugging Face Hub'dan yükler."""
    model_path = "gismo-o/x-company-it-ticket-classifier"
    print(f"IT sınıflandırma modeli {model_path} adresinden yükleniyor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("Model başarıyla yüklendi.")
        return tokenizer, model
    except Exception as e:
        st.error(f"Hugging Face Hub'dan model yüklenirken hata oluştu: {e}")
        return None, None

# SABİTLER 
IT_CATEGORIES = [
    "Ağ", "Donanım", "Yazılım", "Şifre", "Yazıcı Sorunu", "Donanım Ağı",
    "VPN", "Email", "Veri ve Dosya Yönetimi", "Sistem Güncellemeleri",
    "Yazıcı / Tarayıcı / Periferik", "Web ve Uygulama Erişimi",
    "Güvenlik ve Antivirus", "Ses ve Görüntü", "Hesap ve Yetki",
    "Toplantı / Video Konferans"
]

# ANA UYGULAMA FONKSİYONU
def main(): #
    # API ANAHTARI VE SAYFA YAPILANDIRMASI
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except (FileNotFoundError, KeyError):
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    st.set_page_config(page_title="X-Company Dijital Asistan", page_icon="🤖")

    # VERİ VE MODELLERİ YÜKLEME
    rag_vector_store = load_rag_vector_store()
    it_tokenizer, it_model = load_classification_model()
    if it_model is None: return

    # OTURUM VE GİRİŞ KONTROLÜ
    if "user_email" not in st.session_state:
        st.header("X-Company Dijital Asistan'a Hoş Geldiniz")
        st.write("Lütfen devam etmek için kurumsal e-posta adresinizi girin.")

        with st.form("login_form"):
            email = st.text_input("E-posta Adresi", placeholder="ad.soyad@xcompany.com").lower()
            if st.form_submit_button("Giriş Yap"):
                if "@xcompany.com" in email:
                    st.session_state.user_email = email
                    st.session_state.messages = [{"role": "assistant", "content": f"Merhaba! Size nasıl yardımcı olabilirim, {email}?"}]
                    st.session_state.show_form = False
                    st.rerun()
                else:
                    st.error("Lütfen geçerli bir @xcompany.com e-posta adresi girin.")
    
    # KULLANICI GİRİŞ YAPTIYSA, SOHBET ARAYÜZÜNÜ GÖSTER
    else:
        st.header("💬 X-Company Dijital Asistan")

        # 1. Mevcut sohbet geçmişini ekrana yazdır
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        # 2. Form gönderme ve gösterme mantığı
        if st.session_state.get("ticket_submitted", False):
            st.success("Destek talebiniz başarıyla IT ekibine iletildi!")
            st.session_state.show_form = False
            del st.session_state["ticket_submitted"]

        if st.session_state.get("show_form", False):
            category = st.session_state.current_category
            prompt_val = st.session_state.current_prompt
            st.warning(f"ℹ️ Anlaşılan bir **{category}** konusuyla karşı karşıyasınız. Lütfen formu doldurun.")
            with st.expander("IT Destek Formu", expanded=True):
                st.text_input("Konu:", value=prompt_val, disabled=True, key="ticket_konu")
                st.text_area("Sorunun Detayları:", placeholder="Lütfen daha fazla detay verin...", key="ticket_detay")
                st.selectbox("Aciliyet Seviyesi:", ["Düşük", "Normal", "Yüksek", "Kritik"], key="ticket_aciliyet")
                if st.button("Destek Talebi Gönder"):
                    with st.spinner("Talebiniz gönderiliyor..."):
                        success = save_ticket_to_gsheet(
                            st.session_state.ticket_konu,
                            st.session_state.ticket_detay,
                            st.session_state.ticket_aciliyet,
                            category,
                            st.session_state.user_email
                        )
                        if success:
                            st.session_state.ticket_submitted = True
                            st.rerun()
        
        # 3. Yeni kullanıcı girdisini al ve süreci tetikle
        if prompt := st.chat_input("Sorunuzu buraya yazabilirsiniz..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Yeni mesajın henüz işlenmediğini belirtmek için bir bayrak ayarla
            st.session_state.message_processed = False
            st.rerun()

        # 4. Eğer son mesaj kullanıcıya aitse VE HENÜZ İŞLENMEDİYSE, asistan cevabını üret
        if (st.session_state.messages and 
            st.session_state.messages[-1]["role"] == "user" and 
            not st.session_state.get("message_processed", False)):
            
            # Mesajı "işlendi" olarak işaretle ki bir sonraki rerun'da bu blok tekrar çalışmasın
            st.session_state.message_processed = True

            with st.chat_message("assistant"):
                with st.spinner("Yanıt hazırlanıyor, lütfen bekleyin..."):
                    user_prompt = st.session_state.messages[-1]["content"]
                    
                    # Niyet Tespiti: Gelen soruyu önce IT modeline sor.
                    category, confidence = predict_it_ticket_category(user_prompt, it_tokenizer, it_model)
                    is_it_category = category in IT_CATEGORIES and confidence > 0.59
                    
                    if is_it_category: # IT sorunu ise: Formu göstermek için bayrakları ayarla ve sayfayı yenile.
                        st.session_state.show_form = True
                        st.session_state.current_category = category
                        st.session_state.current_prompt = user_prompt
                        st.rerun()
                    else: # IT sorunu değilse: RAG ile cevap üret, cevabı kaydet ve sayfayı yenile.
                        response = handle_user_input(user_prompt, rag_vector_store, st.session_state.messages)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()

# UYGULAMAYI BAŞLATMA
if __name__ == "__main__":
    main()