import os
import gradio as gr
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io

# Fonctions de traitement d'image
def load_image(image):
    return image

def apply_negative(image):
    img_np = np.array(image)
    negative = 255 - img_np
    return Image.fromarray(negative)

def grayscale(image):
    return image.convert('L')

def binarize_image(image, threshold):
    img_np = np.array(image.convert('L'))
    _, binary = cv2.threshold(img_np, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary)

def resize_image(image, width, height):
    width = int(width)
    height = int(height)
    return image.resize((width, height))

def rotate_image(image, angle):
    return image.rotate(angle)

def show_histogram(image):
    image_gray = image.convert("L")
    # Obtenir les données de l'image en niveaux de gris
    image_array = np.array(image_gray)
    # Calculer l'histogramme
    hist, bins = np.histogram(image_array.flatten(), bins=256, range=[0,256])
    # Créer une figure pour l'affichage de l'histogramme
    fig, ax = plt.subplots()
    ax.plot(hist, color='blue')
    ax.set_xlim([0, 256])
    ax.set_title('Histogram of Image')
    # Enregistrer l'histogramme dans un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # Ouvrir l'image du buffer en utilisant PIL
    hist_image = Image.open(buf)
    return hist_image

def gaussian_filter(image, shape=(3, 3)):
    image = np.array(image)
    filtered = cv2.GaussianBlur(image, shape, 0)
    return Image.fromarray(filtered)

def mean_filter(image, shape=(3, 3)):
    image = np.array(image)
    filtered = cv2.blur(image, shape)
    return Image.fromarray(filtered)

def sobel_edges(image, k=5):
    k = int(k)
    image = np.array(image.convert('L'))
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=k)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=k)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    return Image.fromarray(np.uint8(sobel_combined))

def erosion(image, iterations=3, shape=(5, 5)):
    iterations = int(iterations)
    image = np.array(image.convert("L"))
    kernel = np.ones(shape, np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    return Image.fromarray(eroded_image)

def dilatation(image, iterations=3, shape=(5, 5)):
    iterations = int(iterations)
    image = np.array(image.convert("L"))
    kernel = np.ones(shape, np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return Image.fromarray(dilated_image)


# Interface Gradio
def image_processing(image, operation, modified_image, threshold=128, width=100, height=100, angle=30, k=5, iterations=3):
    current_image = modified_image if modified_image is not None else image
    if operation == "Négatif":
        current_image = apply_negative(image)
    elif operation == "Image en Gris":
        current_image = grayscale(image)
    elif operation == "Binarisation":
        current_image = binarize_image(image, threshold)
    elif operation == "Redimensionner":
        current_image = resize_image(image, width, height)
    elif operation == "Rotation":
        current_image = rotate_image(image, angle)
    elif operation == 'Filtre Gaussien':
        current_image = gaussian_filter(image)
    elif operation == 'Filtre Moyen':
        current_image = mean_filter(image)
    elif operation == 'Sobel Edges Extraction':
        current_image = sobel_edges(image, k)
    elif operation == 'Erosion':
        current_image = erosion(image, iterations)
    elif operation == 'Dilatation':
        current_image = dilatation(image, iterations)

    return current_image, show_histogram(current_image)

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Traitement d'Images")

    with gr.Row():
        operation = gr.Radio(["Négatif", "Image en Gris", "Binarisation", "Redimensionner", "Rotation", 'Filtre Gaussien',
                         'Filtre Moyen', 'Sobel Edges Extraction', 'Erosion', 'Dilatation'], label="Opération", value="Négatif")
    with gr.Row():
        threshold = gr.Slider(0, 255, 128, label="Seuil de binarisation", visible=False)
        width = gr.Number(value=100, label="Largeur", visible=False)
        height = gr.Number(value=100, label="Hauteur", visible=False)
        angle = gr.Slider(0, 360, 30, label="Angle de Rotation", visible=False)
        k = gr.Number(value=5, label="k de Sobel", visible=False)
        iterations = gr.Number(value=3, label="Nombre d'iteration pour les transformations morphologiques", visible=False)

    def update_ui(operation):
        # Mise à jour dynamique de la visibilité des champs
        return {
            threshold: gr.update(visible=operation == "Binarisation"),
            width: gr.update(visible=operation == "Redimensionner"),
            height: gr.update(visible=operation == "Redimensionner"),
            angle: gr.update(visible=operation == "Rotation"),
            k: gr.update(visible=operation == "Sobel Edges Extraction"),
            iterations: gr.update(visible=operation in ["Erosion", "Dilatation"])
        }

    operation.change(update_ui, operation, [threshold, width, height, angle, k, iterations])

    with gr.Row():
        image_input = gr.Image(type="pil", label="Charger Image", scale=2)
        original_hist = gr.Image(label="Histogramme de l'Image Originale", scale=1)
    with gr.Row():
        image_output = gr.Image(type="pil", label="Image Modifiée", interactive=False)
        modified_hist = gr.Image(label="Histogramme de l'Image Modifiée", scale=1)

    # Afficher l'histogramme de l'image d'entrée
    def s_hist(image):
        return show_histogram(image)
    image_input.change(s_hist, inputs=image_input, outputs=original_hist)

    submit_button = gr.Button("Appliquer")
    submit_button.click(image_processing, inputs=[image_input, operation, image_output, threshold, width, height, angle, k, iterations], outputs=[image_output, modified_hist])

# Lancer l'application Gradio
demo.launch()
