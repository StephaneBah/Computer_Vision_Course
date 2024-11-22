import os
import gradio as gr
from PIL import Image
from model import kmeans, mean_shift

with gr.Blocks() as demo:
    gr.HTML(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&family=Roboto:wght@400&display=swap');

    body { background-color: #f0f0f0; font-family: 'Roboto', sans-serif; }
    h1 { font-family: 'Poppins', sans-serif; color: #4CAF50; text-align: center; }
    h3 { font-family: 'Roboto', sans-serif; color: #333; text-align: center; }
    .segment-btn { 
        background-color: #4CAF50; 
        border-radius: 8px; 
        padding: 10px 20px; 
        color: white; 
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .segment-btn:hover {
        background-color: #45a049; 
        transform: scale(1.05);
    }
    .logo { 
        display: block; 
        margin: 0 auto; 
        width: 150px; 
        height: auto; 
        padding: 20px 0;
    }
    </style>"""
    )
    #logo_path = os.path.join(os.getcwd(), 'assets/img/fashionlook1.png')
    gr.Markdown("""
        <div style="text-align: center">
            <img src='./fashionlook1.png' class='logo' alt='Logo' style='max-width: 300px'>
        </div>
        """)
    gr.Markdown("<h1>FashionLook - Segment Clothes</h1>")
    gr.Markdown("<h3 style='text-align: center;'>" "Upload an image and let our model detect and segment clothes such as shirts, pants, skirts...""</h3>")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<h5>Upload your image</h5>")
            image_input = gr.Image(type='pil', label="Upload Image")
            with gr.Row():
                segment_button = gr.Button("Run Segmentation", elem_id="segment-btn")
        with gr.Column(scale=1):
            gr.Markdown("<h5>Segmented Image with Overlay</h5>")
            segmented_image_output = gr.Image(type="pil", label="Segmented Image", interactive=False)

        # Actions liées aux inputs/outputs
        segment_button.click(
            fn=kmeans, 
            inputs=[image_input], 
            outputs=[segmented_image_output]
        )

# Lancer l'interface
demo.launch(share=False)