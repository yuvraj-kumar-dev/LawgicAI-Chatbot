
# ⚖️ LawgicAI-Chatbot

![License](https://img.shields.io/github/license/yuvraj-kumar-dev/LawgicAI-Chatbot)
![Issues](https://img.shields.io/github/issues/yuvraj-kumar-dev/LawgicAI-Chatbot)
![Stars](https://img.shields.io/github/stars/yuvraj-kumar-dev/LawgicAI-Chatbot)
![Forks](https://img.shields.io/github/forks/yuvraj-kumar-dev/LawgicAI-Chatbot)

> 💡 A Streamlit-based legal chatbot powered by **RAG architecture**, built to deliver **hallucination-free, document-grounded** answers — unlike generic LLMs.

---

## 🚀 Live Demo

👉 [Click here to try LawgicAI Chatbot](https://your-streamlit-url.com) 

## 📸 Demo

![ScreenRecording2025-04-17132309-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/ac0ab0e2-e099-450d-8811-169bf86aa9ed) 

---

## 🚀 Features

- 🧠 **RAG (Retrieval-Augmented Generation)** backed memory
- 🗃️ Custom **vectorstore using FAISS** for fast retrieval
- 🛡️ Avoids hallucinations — answers are based on your uploaded documents
- 🧩 Modular code (`create_memory.py`, `connect_memory.py`, `chatbot.py`)
- ⚡ Instant and interactive UI with **Streamlit**
- 📁 Clean folder structure with `data/` and `vectorstore/` separation

---

## 🧠 Architecture

```plaintext
📁 data/
   └── your uploaded PDFs
📁 vectorstore/db_faiss/
   └── vector index stored here
📄 create_memory.py
   └── Converts PDFs to chunks → embeds with LLM → stores vectors
📄 connect_memory.py
   └── Loads vectorstore and connects it to LLM
📄 chatbot.py
   └── Streamlit UI to chat with your knowledge base
```

---

## 💡 Why LawgicAI is Better

| Feature | LawgicAI | Traditional LLMs |
|--------|----------|------------------|
| Hallucination-Free | ✅ | ❌ |
| Domain-Aware (Legal) | ✅ | ❌ |
| Custom Data Injection | ✅ | ❌ |
| Modular Architecture | ✅ | ⚠️ |
| Streamlit UI | ✅ | ❌ |

---

## 🛠️ Installation

```bash
git clone https://github.com/yuvraj-kumar-dev/LawgicAI-Chatbot.git
cd LawgicAI-Chatbot
pip install -r requirements.txt
```

---

## ▶️ Running the App

1. **Create Vectorstore**
```bash
python create_memory.py
```

2. **Start Chatbot**
```bash
streamlit run chatbot.py
```

---

## 📂 Project Structure

```
├── data/                     ← Your PDFs go here
├── vectorstore/db_faiss/    ← Vector DB created by FAISS
├── create_memory.py         ← Create memory (RAG setup)
├── connect_memory__.py      ← Connect memory to LLM
├── chatbot.py               ← Streamlit interface
├── LICENSE
└── README.md
```

---

## 📃 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ✨ Contribution

If you find this useful, give it a ⭐ and feel free to contribute!

---

## 👨‍💻 Author

Built with ❤️ by [**Yuvraj Kumar**](https://github.com/yuvraj-kumar-dev)

---
