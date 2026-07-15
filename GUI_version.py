# =====================================================================
# FINAL PROJECT: (Full Version)
# =====================================================================

# 1. IMPORTING TOOLS
import streamlit as st       
import pandas as pd          
import matplotlib.pyplot as plt  
import seaborn as sns        
from openai import OpenAI    

# 2. CONFIGURING THE WEB PAGE
st.set_page_config(page_title="Totem Dashboard", page_icon="🏭", layout="wide")

# =====================================================================
# 3. LOADING DATA FROM THE REAL FILE (CSV)
# =====================================================================
# Remember to have the 'production_data.csv' file in the same folder with the columns:
# Date,Time,Production_Time,Machine,Good_Parts,Defects,Sector,Operator
@st.cache_data
def load_data():
    try:
        table = pd.read_csv("production_data.csv")
        table["Date"] = pd.to_datetime(table["Date"])
        return table
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop() # Stops the program if data is not found

# =====================================================================
# 4. BUILDING THE WEBSITE VISUALS AND CHARTS
# =====================================================================
st.image("logo-1.png")
st.title("Advanced Production Control Center - AI Transformers")
st.markdown("Detailed monitoring by **Sector** and **Operator**.")
st.divider() 

# --- METRIC CARDS ---
c1, c2, c3, c4 = st.columns(4)
total_good = df["Good_Parts"].sum()
total_def = df["Defects"].sum()
total_time = df["Production_Time"].sum()
error_rate = (total_def / (total_good + total_def)) * 100

c1.metric("Total Good Parts", f"{total_good:,}".replace(",", "."))
c2.metric("Total Scrapped", f"{total_def:,}".replace(",", "."))
c3.metric("Production Time", f"{total_time} min")
c4.metric("Global Error Rate", f"{error_rate:.2f}%")

st.divider()

# --- SIDE-BY-SIDE CHARTS ---
g1, g2 = st.columns(2)

with g1:
    st.subheader("Production by Sector")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sector_prod = df.groupby("Sector")["Good_Parts"].sum().reset_index()
    sns.barplot(data=sector_prod, x="Sector", y="Good_Parts", palette="viridis", ax=ax1)
    st.pyplot(fig1)

with g2:
    st.subheader("Performance by Operator (Good Parts)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    op_prod = df.groupby("Operator")["Good_Parts"].sum().sort_values(ascending=True).reset_index()
    # Horizontal chart (y="Operator") to read names more easily
    sns.barplot(data=op_prod, x="Good_Parts", y="Operator", palette="magma", ax=ax2)
    st.pyplot(fig2)

st.divider()

st.subheader("Virtual Production Analyst")

# --- CLEAR CHAT BUTTON ---
col_tit, col_btn = st.columns([0.85, 0.15])
with col_btn:
    if st.button("Clear Chat"):
        if "chat_history" in st.session_state:
            del st.session_state["chat_history"] # Clears the memory
        st.rerun() # Restarts the page to clear the screen

# --- AUTHENTICATION ---
token = st.text_input("Enter your OpenAI API Key to enable the chat:", type="password")

if token:
    client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=token)

    # --- INITIALIZING MEMORY IF IT IS EMPTY ---
    if "chat_history" not in st.session_state:
        
        # Converts the Pandas table to text so the AI can read it
        data_text = df.to_csv(index=False)
        
        # Secret system instruction
        system_context = f"""You are a senior industrial analyst. 
        You have access to this factory production data: 
        
        {data_text}
        
        Help the user identify bottlenecks, best operators, and failures by sector.
        Respond in a clear, helpful, and direct manner."""
        
        # Create the list with the system instruction AND a visible welcome message
        st.session_state["chat_history"] = [
            {"role": "system", "content": system_context},
            {"role": "assistant", "content": "Hello! The history is clean. What would you like to analyze in today's production data?"}
        ]

    # --- RENDERING OLD MESSAGES TO THE SCREEN ---
    for msg in st.session_state["chat_history"]:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # --- RECEIVING NEW QUESTION ---
    question = st.chat_input("Ex: Who is the operator with the most defects in the Machining sector?")
    
    if question:
        # Show the question on screen
        with st.chat_message("user"):
            st.markdown(question)
        
        # Store in secret memory
        st.session_state["chat_history"].append({"role": "user", "content": question})
        
        # Send to the cloud
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state["chat_history"]
        )
        
        # Extract response
        ai_text = response.choices[0].message.content
        
        # Show response on screen
        with st.chat_message("assistant"):
            st.markdown(ai_text)
            
        # Store response in memory
        st.session_state["chat_history"].append({"role": "assistant", "content": ai_text})
