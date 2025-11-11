# X-Company Dijital Asistan

## ![X-Company Dijital Asistan Arayüzü](https://github.com/user-attachments/assets/f172d96b-d148-4d77-b5e2-f82a04aad5d1)

### Proje Hakkında

Bu proje, **Akbank GenAI Bootcamp** kapsamında, `X-Company` adlı kurgusal bir şirket için geliştirilmiş, RAG (Retrieval-Augmented Generation) tabanlı bir kurumsal dijital asistandır. Asistan, şirket içi dokümanları (politikalar, kılavuzlar) ve yapısal verileri (çalışan ve yazılım listeleri) anlayarak çalışanların sorularına doğal dilde yanıt verir.

Asistan, sadece tekil soruları yanıtlamakla kalmaz; aynı zamanda her kullanıcı oturumunu benzersiz bir kimlikle (e-posta) takip eder ve konuşma geçmişini **kısa süreli bir bellek** olarak kullanarak, "peki onun detayları neler?" gibi bağlam gerektiren diyalogları da başarıyla yönetir.

#### Projenin Amacı

Projenin temel amacı, şirket içi bilgiye erişimi kolaylaştırmak, sıkça sorulan soruları otomatikleştirerek İK ve BT departmanlarının yükünü azaltmak ve çalışanlara 7/24 hizmet veren akıllı bir destek kanalı sunmaktır.

Proje, basit bir RAG modelinin ötesine geçerek, **hibrit bir yapı** kullanır:

1.  **Bilgi Asistanı (RAG):** PDF ve CSV dosyalarındaki bilgilere dayanarak genel soruları yanıtlar.
2.  **IT Destek Yönlendiricisi:** Kullanıcının niyetini analiz eder. Eğer bir IT sorunu tespit ederse, RAG'i atlayarak kullanıcıyı ilgili destek formuna yönlendirir ve talebi bir veritabanına kaydeder.

---

<!-- GÖRSELLERİN YAN YANA GÖSTERİLDİĞİ TABLO -->

<table align="center">
  <tr>
    <td align="center"><b>Uygulama Başlangıç Arayüzü</b></td>
    <td align="center"><b>Genel Sorgu Arayüzü</b></td>
    <td align="center"><b>IT Destek Formu</b></td>
    <td align="center"><b>Veritabanı Kaydı (Google Sheets)</b></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/gismo-o/x-company-rag-chatbot/main/assets/arayuz.png" width="250"></td>
    <td><img src="https://raw.githubusercontent.com/gismo-o/x-company-rag-chatbot/main/assets/IT-ticket.png" width="250"></td>
    <td><img src="https://raw.githubusercontent.com/gismo-o/x-company-rag-chatbot/main/assets/db.png" width="250"></td>
    <td><img src="https://raw.githubusercontent.com/gismo-o/x-company-rag-chatbot/main/assets/db.png" width="250"></td>
  </tr>
</table>

---

### Kullanılan Teknolojiler

- **Programlama Dili:** Python 3.10+
- **Web Arayüzü:** Streamlit
- **Dil Modeli (LLM):** Google Gemini 2.5 Flash
- **RAG ve AI Framework'leri:** LangChain, LangChain Community
- **Embedding Modeli:** Google `models/embedding-001`
- **Vektör Veritabanı:** ChromaDB
- **Niyet Sınıflandırma:** Hugging Face Transformers kütüphanesi ile fine-tune edilmiş `dbmdz/bert-base-turkish-cased` modeli.
- **Veri İşleme:** Pandas, **NumPy**
- **Veri Kaydı (Ticket Sistemi):** Google Sheets API
- **Niyet Sınıflandırma (NLP):** Hugging Face (`Transformers`, `Datasets`, `Evaluate`)
- **Model Eğitimi (Fine-Tuning):** **PyTorch**
- **Paket Yönetimi:** `uv`
- **Deney Takibi:** **Weights & Biases (`wandb`)**

---

### 📊 Veri Setleri

Projede kullanılan tüm veri setleri, `X-Company` adlı kurgusal şirket senaryosuna uygun olarak tarafımca sıfırdan oluşturulmuştur. Amaç, gerçek dünya kurumsal ortamını simüle eden bir bilgi tabanı yaratmaktır.

- **Politika Dokümanları (PDF):** `İK Politikası`, `BT Politikası`, `Ofis Yönetimi` ve `Seyahat Politikası` gibi PDF dosyaları, bir şirketin temel operasyonel kurallarını içerecek şekilde detaylı olarak hazırlanmıştır. Bu dokümanlar, RAG sisteminin yapılandırılmamış metin anlama yeteneğini test etmek için ana bilgi kaynağı olarak kullanılır.

- **Yapısal Veriler (CSV):**
  - `xcompany_calisan_listesi.csv`: Şirketin organizasyon şemasını, çalışanların unvanlarını, departmanlarını, yöneticilerini ve en önemlisi **uzmanlık alanlarını** içeren detaylı bir CSV dosyasıdır. Bu veri, "Python bilen kim var?" gibi spesifik yetenek bazlı sorgulara yanıt verebilmek için kritik öneme sahiptir.
  - `yazilimlar.csv`: Şirket bünyesinde kullanılan tüm yazılımları, kategorilerini, sorumlu departmanlarını, lisans türlerini ve kullanım amaçlarını içeren zengin bir envanterdir.

Bu CSV dosyalarının işlenmesi, `app.py` içine gömülü statik kurallar yerine, `csv_configs.json` adlı bir konfigürasyon dosyası ile dinamik olarak yönetilir. Bu şablon tabanlı yaklaşım, her CSV için özel bir metin formatı tanımlanmasına olanak tanır ve sisteme kod değişikliği yapmadan yeni CSV veri kaynakları eklenmesini sağlayarak yüksek esneklik ve bakım kolaylığı sunar.

### Özelleştirilmiş Sınıflandırma Modeli (Fine-Tuning)

Projenin hibrit yapısının temelini oluşturan niyet tespiti (intent detection), basit bir kural tabanlı sistem yerine, son teknoloji NLP (Doğal Dil İşleme) teknikleri kullanılarak tarafımca özelleştirilmiş (fine-tuned) bir Transformer modeline dayanmaktadır.

- **Temel Model (Base Model):** Türkçe metin sınıflandırma görevlerindeki kanıtlanmış başarısı ve dil yapımıza olan derin hakimiyeti nedeniyle `dbmdz/bert-base-turkish-cased` modeli temel olarak seçilmiştir.

- **Özel Veri Seti (Custom Dataset):** Modelin kurumsal IT sorunlarını yüksek doğrulukla sınıflandırabilmesi amacıyla, çeşitli kategorilerde (`Ağ`, `Donanım`, `Yazılım`, `Şifre`, `Diğer` vb.) yaklaşık **3000 satırlık, Türkçe IT destek talebi (ticket)** verisi tarafımca sıfırdan üretilmiştir. Bu veri seti, kullanıcıların bir sorunu ifade edebileceği farklı doğal dil kalıplarını, argo ve teknik terimleri içerecek şekilde tasarlanmıştır.

- **Eğitim Süreci (Fine-Tuning Pipeline):** Modelin eğitimi, `Hugging Face Transformers` ve `Datasets` kütüphaneleri kullanılarak uçtan uca bir pipeline ile gerçekleştirilmiştir:

  1.  **Veri Hazırlama:** `pandas` ile okunan 3000 satırlık CSV dosyası, `scikit-learn`'ün `LabelEncoder`'ı kullanılarak kategorik etiketlerden sayısal etiketlere dönüştürülmüştür. Ardından, veri `Hugging Face Datasets` formatına çevrilerek `train` (%75), `validation` (%12.5) ve `test` (%12.5) olmak üzere üç parçaya stratejik olarak bölünmüştür.
  2.  **Tokenizasyon:** `AutoTokenizer` kullanılarak, tüm metin verisi BERT modelinin anlayabileceği token ID'lerine, attention mask'lerine dönüştürülmüştür. Bu aşamada `truncation=True` ve dinamik padding için `DataCollatorWithPadding` kullanılmıştır.
  3.  **Model Eğitimi:** `Transformers.Trainer` API'si, yapılandırılmış `TrainingArguments` ile birlikte kullanılmıştır. Eğitim süreci; 3 epoch, 2e-5 öğrenme oranı (learning rate) ve `f1_macro` metriğini en iyi modelin seçimi için temel alan bir stratejiyle yürütülmüştür.
  4.  **Metrik Hesaplama:** Modelin performansını değerlendirmek için `Hugging Face Evaluate` kütüphanesi entegre edilmiştir. Her epoch sonunda `accuracy`, `macro F1-score`, `precision` ve `recall` metrikleri hesaplanmıştır.

- **Deney Takibi (Experiment Tracking):**
  Eğitim sürecinin şeffaflığını, tekrar edilebilirliğini ve analizini sağlamak amacıyla tüm metrikler, kayıp (loss) değerleri ve hiperparametreler **[Weights & Biases (wandb.ai)](https://wandb.ai/kozgizemm-/huggingface?nw=nwuserkozgizemm)** platformuna entegre edilmiştir. Bu sayede, `train/loss` ve `eval/loss` eğrileri gibi kritik görseller canlı olarak takip edilmiş, modelin öğrenme süreci ve potansiyel "overfitting" durumları anlık olarak analiz edilmiştir.

- **Sonuç ve Dağıtım:**
  Başarılı bir eğitim ve doğrulama sürecinin ardından, `validation` setinde en yüksek `f1_macro` skorunu elde eden modelin en iyi versiyonu kaydedilmiştir. Projenin dağıtımını kolaylaştırmak ve Git LFS bağımlılığını ortadan kaldırmak amacıyla, bu son ve optimize edilmiş model dosyaları [Hugging Face Hub](https://huggingface.co/gismo-o/x-company-it-ticket-classifier) üzerine yüklenmiştir. Streamlit uygulaması, modeli doğrudan bu platform üzerinden, `AutoModelForSequenceClassification.from_pretrained()` fonksiyonu aracılığıyla dinamik olarak çekmektedir.

### ⚙️ Kurulum ve Başlatma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

**1. Projeyi Klonlayın:**

```bash
git clone https://github.com/gismo-o/x-company-rag-chatbot.git
cd x-company-rag-chatbot
```

**2. Sanal Ortam Oluşturun ve Aktif Edin:**

```bash
# Sanal ortamı oluştur
uv venv

# Sanal ortamı aktif et (Windows)
.\venv\Scripts\activate

# Sanal ortamı aktif et (macOS/Linux)
source venv/bin/activate
```

**3. Gerekli Kütüphaneleri Yükleyin:**

```bash
uv pip install -r requirements.txt
```

**4. Hassas Bilgileri (Secrets) Yapılandırın:**
Projenin çalışması için API anahtarları gereklidir. Proje ana dizininde `.streamlit` adında bir klasör ve içinde `secrets.toml` ve `.env` adında dosya oluşturun.

**`.streamlit/secrets.toml` dosyasının içeriği şu formatta olmalıdır:**

```toml
# Google Cloud'dan indirilen servis hesabı .json dosyasının içeriği
[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = """-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"""
client_email = "..."
# ... (geri kalan tüm alanlar) ...
```

**`.env` dosyasının içeriği şu formatta olmalıdır:**

```env
# Google AI Studio'dan alınan Gemini API Anahtarı
GOOGLE_API_KEY = "API ANAHTARINIZ"
```

---

**5. Uygulamayı Başlatın:**

```bash
streamlit run app.py
```

Uygulama `http://localhost:8501` adresinde başlayacaktır.

---

### 📂 Proje Yapısı

```
x-company-rag-chatbot/
│
├── .streamlit/  # Streamlit Cloud deploy'u için hassas bilgiler (repo'ya dahil değil).
│   ├── config.toml
│   └── secrets.toml
│
├── assets/
│   ├── arayuz1.png           # README'de kullanılan proje görselleri.
│   ├── ticket.png
│   ├── aratuz2.png
│   └── db.png
│
├── data/                    # Chatbot'un bilgi kaynağı olan tüm PDF ve CSV dokümanları.
│   ├── BT_Politikasi.pdf
│   ├── xcompany_calisan_listesi.csv
│   ├── Finans_Politikasi.pdf
│   ├── Ofis_Yonetimi.pdf
│   ├── IK_Politikasi.pdf
│   └── yazilimlar.csv
│
├── .env                     # Yerel geliştirme için API anahtarları
├── .gitignore               # Git tarafından takip edilmeyecek dosya ve klasörlerin listesi.
├── app.py                   # Streamlit uygulamasının tüm mantığını içeren ana kod.
├── csv_configs.json         # CSV dosyalarının metne dönüştürülme şablonlarını içeren konfigürasyon dosyası.
├── IT_tickets_model.py      # IT niyet sınıflandırma modelini eğitmek için kullanılan dosya.
├── packages.txt             # Streamlit Cloud için gerekli olan sistem (apt-get) bağımlılıkları.
└── requirements.txt         # Projenin ihtiyaç duyduğu Python kütüphaneleri.
```

---

### Nasıl Çalışır? (Hibrit Model Mimarisi)

Uygulama, kullanıcıdan bir soru aldığında iki aşamalı bir mantıkla çalışır:

**Aşama 1: Niyet Tespiti**

- Kullanıcının sorusu, ilk olarak IT sorunlarını tespit etmek üzere eğitilmiş **BERT tabanlı sınıflandırma modeline** gönderilir.
- Model, sorunun güven skorunu ve kategorisini (`Ağ`, `Donanım`, `Yazılım`, `Diğer` vb.) tahmin eder.

**Aşama 2: Yönlendirme**

- **Eğer Soru Bir IT Sorunuysa:**
  1.  RAG süreci tamamen atlanır.
  2.  Kullanıcıya sorununun anlaşıldığına dair bir mesaj gösterilir.
  3.  Ekranda interaktif bir **"IT Destek Formu"** belirir.
  4.  Kullanıcı formu doldurup gönderdiğinde, talep **Google Sheets API** aracılığıyla bir e-tabloya kaydedilir.
- **Eğer Soru Genel Bir Bilgi Sorusuysa:**
  1.  Soru, **RAG (Retrieval-Augmented Generation)** pipeline'ına yönlendirilir.
  2.  **Kısa Süreli Bellek:** Sistemin bağlamı anlaması için, kullanıcının son birkaç mesajı da konuşma geçmişi olarak işleme dahil edilir.
  3.  **Vektör Arama:** Kullanıcının sorusu bir embedding modeline gönderilerek vektöre dönüştürülür ve **ChromaDB**'de en alakalı metin parçacıkları ("context") bulunur.
  4.  **Zenginleştirilmiş Sorgu:** Bulunan "context", konuşma geçmişi ve kullanıcının sorusu bir araya getirilerek **Google Gemini 1.5 Flash** modeline gönderilir.
  5.  **Cevap Üretimi:** Gemini, kendisine verilen bu zenginleştirilmiş bağlama sadık kalarak, konuşmanın akışını da dikkate alan bir cevap üretir.

---

### 💡 Örnek Sorular

**Genel Bilgi Soruları (RAG):**

- `Python ve SQL konusunda uzman olan kim var?`
- `Jira'nın alternatifi nedir?`
- `6 yıldır şirkette çalışıyorum. Yıllık izin hakkım kaç gün?`
- `Yurt içi seyahatlerde günlük yemek harcırahı ne kadar?`

**IT Destek Soruları (Sınıflandırıcı):**

- `Bilgisayarım açılmıyor.`
- `Outlook sürekli donuyor ve kapanıyor.`
- `İnternet bağlantım çok yavaş.`
- `Şifremi unuttum, nasıl sıfırlayabilirim?`

---

### 📝 Önemli Notlar

- **Modelin Yüklenmesi:** IT niyet sınıflandırma modeli, doğrudan [Hugging Face Hub](https://huggingface.co/gismo-o/x-company-it-ticket-classifier) üzerinden yüklenmektedir.
- **Veritabanı Oluşturma:** `chroma_db` vektör veritabanı, uygulama ilk kez çalıştığında `data/` klasöründeki dokümanları işleyerek oluşturulur. `data/` klasöründeki dosyaları güncellerseniz, deploy edilmiş uygulamanın önbelleğini temizlemeniz veya yerelde `chroma_db` klasörünü silmeniz gerekir.

---

## 🚀 Canlı Demo

Bu projenin canlı demosuna aşağıdaki linkten erişebilirsiniz:

**[https://x-company-rag-chatbot.streamlit.app/](https://x-company-rag-chatbot.streamlit.app/)**
