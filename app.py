import gradio as gr
from unsloth import FastLanguageModel
import torch
import os

# Global model and tokenizer (loaded once)
model = None
tokenizer = None
MAX_SEQ_LENGTH = 2048 # Or your model's specific max sequence length
DEFAULT_DROPDOWN_OPTION = "(Select a predefined query or type your own)"

def load_dropdown_options_from_file(file_path: str) -> list:
    """Loads options from a text file, one option per line, ignoring empty lines and comments."""
    options = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    options.append(line)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return options

def load_model_globally():
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Loading Unsloth model...")
        try:
            model_name = "Nolan-Robbins/unsloth-gemma-3-4B-customer-support"
            loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(loaded_model)
            model = loaded_model
            tokenizer = loaded_tokenizer
            print("Unsloth model loaded successfully.")
        except Exception as e:
            print(f"Error loading Unsloth model: {e}")
            raise gr.Error(f"Failed to load model: {str(e)}")

try:
    load_model_globally()
except gr.Error as e:
    print(f"Startup Model Load Error: {e}")
    pass

def respond(message, chat_history, system_prompt, selected_dropdown_query, max_new_tokens, temperature, top_p):
    global model, tokenizer

    if model is None or tokenizer is None:
        print("Model not loaded. Attempting to load now (should have loaded at startup)...")
        try:
            load_model_globally()
            if model is None or tokenizer is None:
                 raise gr.Error("Model could not be loaded. Please check Space logs or restart.")
        except Exception as e:
            raise gr.Error(f"Model loading failed on demand: {str(e)}")

    actual_user_message = message
    if (not message or not message.strip()) and \
       selected_dropdown_query and \
       selected_dropdown_query != DEFAULT_DROPDOWN_OPTION:
        actual_user_message = selected_dropdown_query
    
    if not actual_user_message or not actual_user_message.strip():
        yield "Please provide a message or select a predefined query."
        return

    full_prompt_text = ""
    if system_prompt and system_prompt.strip():
        full_prompt_text += f"{system_prompt.strip()}\n\n"

    for user_msg, assistant_msg in chat_history: # Processes tuples from chat_history
        full_prompt_text += f"<start_of_turn>user\n{user_msg}<end_of_turn>\n"
        if assistant_msg:
            full_prompt_text += f"<start_of_turn>model\n{assistant_msg}<end_of_turn>\n"

    full_prompt_text += f"<start_of_turn>user\n{actual_user_message}<end_of_turn>\n"
    full_prompt_text += "<start_of_turn>model\n"

    inputs = tokenizer(
        [full_prompt_text],
        return_tensors="pt",
    ).to("cuda:0" if torch.cuda.is_available() else "cpu")

    generated_text_response = ""
    try:
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=int(max_new_tokens),
            temperature=max(temperature, 0.01),
            top_p=top_p if top_p < 1.0 else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            use_cache=True
        )
        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text_response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        yield generated_text_response

    except Exception as e:
        print(f"Error during local model generation: {e}")
        yield f"Local Model Error: {str(e)}"

# Load dropdown options from file
dropdown_choices = [DEFAULT_DROPDOWN_OPTION] + load_dropdown_options_from_file("dropdown_options.txt")

# Define the Gradio Interface, aligning with the original structure
demo = gr.ChatInterface(
    fn=respond,
    chatbot=gr.Chatbot(
        type="tuples",  # Kept as "tuples" to match original app.py logic and avoid breaking history processing
        label="Unsloth Gemma Customer Support Chatbot", # You can customize this label
        bubble_full_width=False,
        height=600,
        # avatar_images can be set here if desired, e.g., (None, "path_to_bot_avatar.png")
    ),
    additional_inputs=[
        gr.Textbox(value="You are a friendly Customer Support Chatbot expert at helping users.", label="System message", info="Define the role and behavior of the chatbot."),
        gr.Dropdown(
            choices=dropdown_choices,
            value=DEFAULT_DROPDOWN_OPTION,
            label="Predefined User Queries (Optional)",
            info="Select a query. If the main message box is empty, this query will be used."
        ),
        gr.Slider(minimum=1, maximum=MAX_SEQ_LENGTH // 2, value=256, step=1, label="Max new tokens", info="Max new tokens the model will generate."),
        gr.Slider(minimum=0.01, maximum=2.0, value=0.7, step=0.01, label="Temperature", info="Controls randomness. Lower is more deterministic."),
        gr.Slider(
            minimum=0.01,
            maximum=1.0,
            value=0.95,
            step=0.01,
            label="Top-p (nucleus sampling)",
            info="Considers tokens with top_p probability mass. 1.0 considers all."
        ),
    ],
    title="Unsloth Gemma Customer Support Chatbot (Local Model)",
    description=f"Chat with a locally loaded Unsloth model ({model.config._name_or_path if model and hasattr(model, 'config') else 'Nolan-Robbins/unsloth-gemma-3-4B-customer-support'}). How can I help you today?",
    examples=[["i do not know how i could switch tto the standard account"], ["I have to edit order {{Order Number}}, how to do it?"], ["My order #12345 has not arrived."]], 
    cache_examples=False, 
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="neutral"), 
    # Removed problematic arguments: submit_btn, retry_btn, undo_btn, clear_btn
)

if __name__ == "__main__":
    if model is None or tokenizer is None:
        print("Model was not loaded successfully at startup. The UI might show an error or allow retrying.")
    demo.launch()