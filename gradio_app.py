import gradio as gr
from groq import Groq
from dotenv import load_dotenv
import json
import re
import os

load_dotenv() 
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)  

def process_message(msg):
    prompt = f"""
    You are an AI message processor. For the following customer message, produce **strict JSON only** with keys: Category, Sentiment, Auto-Reply.

    Category: one of [Complaint, Refund/Return, Sales Inquiry, Delivery Question, Account/Technical Issue, General Query, Spam]
    Sentiment: Positive, Neutral, Negative
    Auto-Reply: short professional reply (2-4 sentences)

    Message: "{msg}"
    """
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()

        # extract JSON if model adds extra text
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            return {"Category": "Error", "Sentiment": "Neutral", "Auto-Reply": "No valid JSON received"}
    except Exception as e:
        return {"Category": "Error", "Sentiment": "Neutral", "Auto-Reply": f"Error: {e}"}


# CSS
css = """
/* Hide all Gradio branding */
footer, .flag, .gr-button-secondary, .gr-footer { display: none !important; }

.gradio-container {
    background: #0b0c10 !important;
    color: #c5c6c7 !important;
    font-family: 'Poppins', sans-serif !important;
}

/* Textbox */
textarea, input {
    background: #1f2833 !important;
    color: #fff !important;
    border: 1px solid #45a29e !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 16px !important;
}

/* Limit width and center the textbox */
#msg-box {
    max-width: 1500px;
    margin-left: auto;
    margin-right: auto;
}

/* Output Box */
.output-box {
    background: #1f2833;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(102, 252, 241, 0.5);
    font-size: 18px;
    color: #ffffff;
    margin-top: 30px;
    line-height: 2;
}

/* Each row in output */
.output-box b {
    color: #ffffff !important;
    font-weight: 700;
    font-size: 19px !important;
}

/* Add spacing between each row */
.output-box br {
    content: "";
    margin-bottom: 10px;
    display: block;
}

/* Header */
h1 {
    font-size: 36px !important;
    background: linear-gradient(90deg, #66fcf1, #45a29e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}

/* Description */
h2, p {
    color: #c5c6c7 !important;
}

/* Buttons */
button {
    background: #45a29e !important;
    color: black !important;
    border-radius: 8px !important;
    padding: 12px 20px !important;
    font-weight: 600 !important;
    border: none !important;
    margin-top: 10px;
}

button:hover {
    background: #66fcf1 !important;
    color: black !important;
}

/* Center rows for buttons and add spacing */
.gradio-container > .row {
    justify-content: center;
    margin-bottom: 15px;
}
"""


# Gradio Function UI
def gradio_process(msg):
    if not msg.strip():
        return "<div style='color:#ff6b6b; font-weight:700; margin-bottom:15px;'>⚠ Please enter a message first!</div>"
    
    result = process_message(msg)
    return f"""
<div class='output-box'>
<div style='margin-bottom:12px;'><b>Category:</b> {result['Category']}</div>
<div style='margin-bottom:12px;'><b>Sentiment:</b> {result['Sentiment']}</div>
<div><b>Auto-Reply:</b> {result['Auto-Reply']}</div>
</div>
"""

# Clear function
def clear_all():
    return "", "<div style='color:#888; text-align:center;'>Message cleared</div>"

with gr.Blocks(css=css) as app:
    gr.HTML("<h1 style='text-align:center; margin-bottom:10px;'>⚡AI Customer Message Processor</h1>")
    gr.HTML("<p style='text-align:center; color:#bbb; margin-bottom:25px;'>Detect category, sentiment, and generate professional replies</p>")

    with gr.Row():
        txt = gr.Textbox(
            lines=3,
            placeholder="Type a customer message...",
            label="Customer Message",
            elem_id="msg-box"
        )

    with gr.Row():
        btn = gr.Button("Analyze Message")
        clear_btn = gr.Button("Clear", variant="secondary")

    out = gr.HTML()

    btn.click(gradio_process, inputs=txt, outputs=out)
    txt.submit(gradio_process, inputs=txt, outputs=out)
    clear_btn.click(lambda: ("", ""), inputs=None, outputs=[txt, out])

app.launch()
