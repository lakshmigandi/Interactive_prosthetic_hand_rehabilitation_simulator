# 🦾 Interactive Prosthetic Hand Rehabilitation Simulator (KAG + YUOM)

### **Developer/Principal Investigator:** Dr. Lakshmi Gandi

---

## 📌 Project Overview
This repository hosts an advanced AI-powered Neuro-Prosthetic Rehabilitation Engineering Assistant. By blending the **KAG (Knowledge-Augmented Generation)** retrieval layer with a **YUOM multi-agent routing paradigm**, this system allows clinicians and therapists to safely guide, track, and monitor patient prosthetic training sessions.

---

## 🛡️ Core Safety Architecture: The Tele-Kinetic Sentinel
A cornerstone feature of this simulator is its strict physical safety compliance framework:
* **The 15-Newton Threshold Rule:** The system continuously monitors grip-force analytics. 
* **Spasm Protection:** If an uncommanded grip-force spike or involuntary muscle spasm exceeding **15 Newtons** is detected, the **Tele-Kinetic Sentinel protocol** instantly intercepts the command pipeline.
* **Closed-Loop Feedback:** It automatically triggers an emergency mechanical stall-release protocol, dispatches immediate haptic vibration pulses back to the user's residual limb interface, and logs a timestamped safety event directly into the clinical database.

---

## ⚙️ Core Capabilities
* 📚 **Prosthetic Knowledge Retrieval:** Semantic extraction from the technical 3D robotic prosthetic manual.
* 🎯 **Grasp Classification Guidance:** Step-by-step assistance for Power Hook, Precision Pincer, and active finger joint training.
* 📊 **Clinical Resource Management:** Agentic database integration to check real-time availability of rehabilitation tracking systems, assessment labs, and tele-rehabilitation monitoring interfaces.
* 🔒 **Rigid Privacy Masking:** Hardcoded compliance parameters ensuring no proprietary creator, author, or patient-identifiable biographical data is exposed out-of-context.

---

## 🛠️ Tech Stack & Architecture Components
* **Frontend UI:** Gradio (`gr.Blocks` custom layout)
* **Orchestration & State Machine:** LangGraph / LangChain ReAct Framework
* **Vector Store Database:** FAISS (Facebook AI Similarity Search) using `sentence-transformers/all-MiniLM-L6-v2`
* **Relational Database:** SQLite3 (`prosthetic_hand_suite.db`)
* **Core Foundation LLM:** Llama-3.1-8b-instant via Groq Cloud API
