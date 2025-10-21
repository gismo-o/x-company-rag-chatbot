import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gspread
import json
from google.oauth2.service_account import Credentials




@st.cache_data # Bu konfigürasyonun sadece bir kez okunmasını sağlar
def load_csv_configs():
    """csv_configs.json dosyasını yükler."""
    try:
        with open("csv_configs.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Eğer konfigürasyon dosyası yoksa, sadece varsayılan bir yapı döndür
        return {
            "_default": {
                "template": "{__ALL_COLUMNS__}."
            }
        }
    
def process_row_with_config(row: pd.Series, file_name: str, configs: dict) -> str:
    """
    Verilen bir CSV satırını, konfigürasyon dosyasına göre bir cümleye dönüştürür.
    """
    # Dosya adı için özel bir konfigürasyon var mı kontrol et, yoksa _default kullan
    config = configs.get(file_name, configs["_default"])
    template = config["template"]
    
    # Satır verilerini bir sözlüğe çevir 
    row_dict = row.to_dict()

    # Şablondaki özel yer tutucuları işle
    if "{__COLUMN_0__}" in template:
        template = template.replace("{__COLUMN_0__}", str(row.iloc[0]))
    
    if "{__ALL_COLUMNS__}" in template:
        all_cols_text = ", ".join([f"{k}: {v}" for k, v in row_dict.items() if pd.notna(v)])
        template = template.replace("{__ALL_COLUMNS__}", all_cols_text)

    # Geri kalan tüm normal {SütunAdı} yer tutucularını doldur
    # .format(**row_dict) metodu, sözlükteki anahtarlarla şablondaki yer tutucuları eşleştirir
    try:
        return template.format(**row_dict)
    except KeyError as e:
        print(f"Şablonda hata: CSV'de olmayan bir sütun isteniyor -> {e}")
        return "" # Hata durumunda boş string döndür



# Tüm dökümanlardan metinleri çıkaran ana fonksiyon
def get_documents_text():
    # İlk olarak CSV konfigürasyonunu yükle
    csv_configs = load_csv_configs()

    docs_path = "./data/"
    if not os.path.exists(docs_path):
        return ""
        
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

# Metni yönetilebilir parçalara bölen fonksiyon
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Metin parçalarından vektör deposu oluşturan ve kaydeden fonksiyon
def get_vector_store(text_chunks):
    # Google'ın embedding modelini yükle
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Chroma DB'yi diskte saklamak için bir dizin belirt. Bu sayede her seferinde baştan oluşturmak gerekmez
    vector_store_path = "chroma_db"
    
    # Eğer daha önce oluşturulmuş bir veritabanı varsa, onu yükle
    if os.path.exists(vector_store_path):
        vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    # Yoksa, metin parçalarından yeni bir veritabanı oluştur
    else:
        vector_store = Chroma.from_texts(
            text_chunks, 
            embedding=embeddings, 
            persist_directory=vector_store_path
        )
        vector_store.persist()
        
    return vector_store


# LLM ile konuşma zincirini oluşturan fonksiyon
def get_conversational_chain():
    # Prompt şablonu: LLM'e nasıl davranması gerektiğini söylüyoruz
    prompt_template = """
    Sen X-Company'nin yardımsever bir kurumsal asistanısın. Cevaplarını sadece aşağıda verilen bağlama (context) dayanarak, kısa ve öz bir şekilde oluştur. 
    Eğer cevap verilen bağlamda bulunmuyorsa, "Bu konuda bilgi sahibi değilim." de. Kendi bilgini kullanma.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    # LLM modelini yapılandır
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # Prompt'u ve modeli bir araya getir
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Soru-cevap zincirini yükle
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain


# Kullanıcı sorusunu işleyen ana fonksiyon
def handle_user_input(user_question, vector_store):
    # Kullanıcının sorusuna en çok benzeyen dökümanları vektör deposunda bul
    # k=5, en alakalı 5 metin parçasını getirmesini söyler.
    docs = vector_store.similarity_search(user_question, k=5)
    
    # Soru-cevap zincirini al
    chain = get_conversational_chain()
    
    # Zinciri dökümanlar ve soru ile çalıştır
    response = chain(
        {"input_documents": docs, "question": user_question}, 
        return_only_outputs=True
    )
    
    return response["output_text"]



# IT Sınıflandırma modelini ve tokenizer'ı yükleyen fonksiyon
@st.cache_resource
def load_classification_model():
    """IT sınıflandırma modelini ve tokenizer'ı Hugging Face Hub'dan yükler."""
    model_path = "gismo-o/x-company-it-ticket-classifier" #Hugging Face Yolu
    
    print(f"IT sınıflandırma modeli {model_path} adresinden yükleniyor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("Model başarıyla yüklendi.")
        return tokenizer, model
    except Exception as e:
        # Hata durumunda Streamlit arayüzünde net bir hata mesajı göster
        st.error(
            f"Hugging Face Hub'dan model yüklenirken bir hata oluştu: {e}\n\n"
            f"Lütfen kontrol edin:\n"
            f"1. Model reposunun adı doğru mu? ('{model_path}')\n"
            f"2. Repo 'public' olarak ayarlı mı?\n"
            f"3. İnternet bağlantınızda bir sorun var mı?"
        )
        # Hata durumunda uygulamanın devam etmesini engelle
        return None, None

# Kullanıcının girdisini sınıflandıran fonksiyon
def predict_it_ticket_category(text, tokenizer, model):
    # Metni token'lara çevir
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Modelden tahmin al
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # Olasılıkları hesapla
    probabilities = torch.nn.functional.softmax(logits, dim=-1).flatten()
    
    # En yüksek olasılığa sahip etiketi ve skorunu bul
    predicted_class_id = probabilities.argmax().item()
    confidence = probabilities[predicted_class_id].item()
    predicted_class_label = model.config.id2label[predicted_class_id]
    
    return predicted_class_label, confidence


# Google Sheets'e destek talebini kaydeden fonksiyon
def save_ticket_to_gsheet(konu, detay, aciliyet, kategori):
    try:
        # Streamlit'in secrets yönetiminden kimlik bilgilerini al
        scopes = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
        client = gspread.authorize(creds)
        
        # Google Sheet'i adıyla aç ve ilk çalışma sayfasını seç
        sheet = client.open("X-Company IT Talepleri").sheet1
        
        # Yeni satır olarak eklenecek veriyi hazırla
        new_row = [
            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            kategori,
            konu,
            detay,
            aciliyet,
            "Yeni Talep"
        ]
        sheet.append_row(new_row)
        return True
    except Exception as e:
        st.error(f"Veritabanına yazılırken bir hata oluştu: {e}")
        return False


# IT sorunu olarak kabul edilecek kategorilerin listesi
IT_CATEGORIES = [
    "Ağ", "Donanım", "Yazılım", "Şifre", "Yazıcı Sorunu", "Donanım Ağı",
    "VPN", "Email", "Veri ve Dosya Yönetimi", "Sistem Güncellemeleri",
    "Yazıcı / Tarayıcı / Periferik", "Web ve Uygulama Erişimi",
    "Güvenlik ve Antivirus", "Ses ve Görüntü", "Hesap ve Yetki",
    "Toplantı / Video Konferans"
] 


# Streamlit uygulamasını çalıştıran ana fonksiyon
def main():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    st.set_page_config(
        page_title="X-Company Kurumsal Asistan", 
        page_icon=":robot_face:"
    )

    # VERİ İŞLEME VE MODELLERİ YÜKLEME 
    @st.cache_resource
    def load_rag_vector_store():
        print("RAG Veritabanı yükleniyor...")
        raw_text = get_documents_text()
        text_chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)
        return vector_store
    
    rag_vector_store = load_rag_vector_store()
    it_tokenizer, it_model = load_classification_model()

    # ARAYÜZ ELEMENTLERİ
    st.header("💬 X-Company Dijital Asistan")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": (
            "Merhaba! Ben **X-Company Asistanıyım**. "
            "Şirket süreçleri, bilgiler veya genel sorularınız için buradayım. "
            "Bugün size nasıl yardımcı olabilirim?"
        )}
        ]

    # Sohbet geçmişini göster 
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)


    # Formun durumunu chat girdisi dışında yönetme
    # Eğer bir önceki adımda form gönderildiyse, başarı mesajını göster.
    # Bu, sayfa yenilense bile mesajın kalıcı olmasını sağlar.
    if st.session_state.get("ticket_submitted", False):
        st.success("Destek talebiniz başarıyla IT ekibine iletildi!")
        st.session_state.show_form = False  # Formu tekrar gösterme
        del st.session_state["ticket_submitted"] # Bayrağı temizle

    # Yeni bir chat girdisi varsa, onu işle ve state'i güncelle.
    if prompt := st.chat_input("Sorunuzu buraya yazabilirsiniz..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Yeni bir mesaj geldiğinde, önceki form durumunu temizle
        st.session_state.show_form = False
        
        with st.chat_message("assistant"):
            with st.spinner("Yanıt hazırlanıyor, lütfen bekleyin..."):
                category, confidence = predict_it_ticket_category(prompt, it_tokenizer, it_model)
                #st.info(f"DEBUG: Tahmin: '{category}', Güvenilirlik: {confidence:.2f}")

                is_it_category = category in IT_CATEGORIES and confidence > 0.59

                if is_it_category:
                    # Formu göstermek için sadece bir "bayrak" ayarla.
                    st.session_state.show_form = True
                    # Formun ihtiyaç duyacağı bilgileri state'e kaydet
                    st.session_state.current_category = category
                    st.session_state.current_prompt = prompt
                else:
                    # IT sorunu değilse, RAG ile cevap ver ve bitir.
                    response = handle_user_input(prompt, rag_vector_store)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    # Bayrak True ise, formu göster. Bu bölüm her etkileşimde kontrol edilir.
    if st.session_state.get("show_form", False):
        # State'den gerekli bilgileri al
        category = st.session_state.current_category
        prompt_val = st.session_state.current_prompt

        st.warning(f"ℹ️ Anlaşılan bir **{category}** konusuyla karşı karşıyasınız. İlgili formu doldurarak bize iletebilirsiniz.")
        
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
                        category
                    )
                    
                    if success:
                        # Başarılı olursa, bir sonraki çalıştırmada mesajı göstermek için bayrağı ayarla
                        st.session_state.ticket_submitted = True
                        # Değişikliklerin anında görünmesi için sayfayı yeniden çalıştır
                        st.rerun()
                    # Başarısız olursa, save_ticket_to_gsheet fonksiyonu zaten hata mesajı gösterecektir.

# Uygulamayı başlatmak için
if __name__ == "__main__":
    main()