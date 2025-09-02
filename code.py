!pip install -q gradio_client
!pip install -q gradio
import warnings
warnings.filterwarnings("ignore")
import gradio as gr
from gradio_client import Client, handle_file

def generate_image(prompt, image_url=None, image_file=None):
    # Initialize the client
    client = Client("yanze/PuLID-FLUX")

    # Determine input image
    if image_url:
        id_image = handle_file(image_url)
    elif image_file:
        id_image = handle_file(image_file.name)
    else:
        return "Error: Please provide an image URL or upload an image file."

    # Predict
    try:
        result = client.predict(
            prompt=prompt,
            id_image=id_image,
            start_step=0,
            guidance=10,
            seed="-1",
            true_cfg=1,
            width=1024,
            height=1024,
            num_steps=20,
            id_weight=1,
            neg_prompt="bad quality, worst quality, text, signature, watermark, extra limbs",
            timestep_to_start_cfg=1,
            max_sequence_length=512,
            api_name="/generate_image"
        )

        # Extract the base URL and file path
        base_url = "https://yanze-pulid-flux.hf.space/file="
        file_path = result[0]  # The first element contains the file path of the primary result
        full_url = f"{base_url}{file_path}"

        return full_url
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Image Generation App\nUpload an image or provide an image URL, and enter a prompt to generate a new image.")

        with gr.Row():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt, e.g., portrait, color, cinematic")
            image_url = gr.Textbox(label="Image URL", placeholder="Enter the image URL (optional)")

        image_file = gr.File(label="Upload Image", file_types=["image"])

        with gr.Row():
            submit_button = gr.Button("Generate Image")

        output = gr.Textbox(label="Generated Image URL")
        output_image = gr.Image(label="Generated Image")

        def process(prompt, image_url, image_file):
            result_url = generate_image(prompt, image_url, image_file)
            if result_url.startswith("http"):
                return result_url, result_url
            else:
                return result_url, None

        submit_button.click(
            fn=process,
            inputs=[prompt, image_url, image_file],
            outputs=[output, output_image]
        )

    return demo

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()
