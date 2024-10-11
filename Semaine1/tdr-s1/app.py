import os
import gradio as gr
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def resize_image(image, width: int, height: int):
    return image.resize((width, height))

def rotate_image(image, angle):
    return image.rotate(angle)

def show_histogram(image):
    grayscale = image.convert("L")
    plt.hist(grayscale, bins=120)
    #hist_data = grayscale.histogram()
    plt.figure()
    plt.plot(hist_data)
    plt.title("Histogramme des Niveaux de Gris")
    plt.show()

def gaussian_filter(image, shape=(3,3)):
    image = np.array(image)
    filtered = cv2.GaussianBlur(image, shape, 0)
    return Image.fromarray(filtered)

def mean_filter(image, shape=(3,3)):
    image = np.array(image)
    filtered = cv2.blur(image, shape)
    return Image.fromarray(filtered)

def sobel_edges(image, k=5):
    image = np.array(image.convert('L'))
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=k)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=k)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    return Image.fromarray(np.uint8(sobel_combined))

def erosion(image, noyau=(5,5), iterations=3):
    image = np.array(image.convert("L"))
    kernel = np.ones(noyau, np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    return Image.fromarray(eroded_image)

def dilatation(image, noyau=(5,5), iterations=3):
    image = np.array(image.convert("L"))
    kernel = np.ones(noyau, np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return Image.fromarray(dilated_image)


# Ajoutez d'autres fonctions pour l'histogramme, le filtrage, Sobel, etc.

# Interface Gradio
def image_processing(image, operation, threshold=128, width=100, height=100, angle=30, shape=(3,3), noyau=(5,5), k=5, iterations=3):
    if operation == "Négatif":
        return apply_negative(image)
    elif operation == "Image en Gris":
        return grayscale(image)
    elif operation == "Binarisation":
        return binarize_image(image, threshold)
    elif operation == "Redimensionner":
        return resize_image(image, width, height)
    elif operation == "Rotation":
        return rotate_image(image, angle)
    elif operation == 'Histogramme de Gris':
        return show_histogram(image)
    elif operation == 'Filtre Gaussien':
        return gaussian_filter(image, shape)
    elif operation == 'Filtre Moyen':
        return mean_filter(image, shape)
    elif operation == 'Sobel Edges Extraction':
        return sobel_edges(image, k)
    elif operation == 'Erosion':
        return erosion(image, noyau, iterations)
    elif operation == 'Dilatation':
        return dilatation(image)
    # Ajouter d'autres conditions pour les autres opérations
    return image

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Projet de Traitement d'Image")

    with gr.Row():
        operation = gr.Radio(["Négatif", "Image en Gris", "Binarisation", "Redimensionner", "Rotation", 'Histogramme de Gris', 
                            'Filtre Gaussien', 'Filtre Moyen', 'Sobel Edges Extraction', 'Erosion', 'Dilatation'], label="Opération")
        threshold = gr.Slider(0, 255, 128, label="Seuil de binarisation", visible=True)
        width = gr.Number(value=100, label="Largeur", visible=True)
        height = gr.Number(value=100, label="Hauteur", visible=True)
        angle = gr.Slider(0, 360, 30, label="Angle de Rotation", visible=True)
        k = gr.Number(value=5, label="k de Sobel", visible=True)
        iterations = gr.Number(value=3, label="Nombre d'iteration pour les transformations morphologiques", visible=True)
    with gr.Row():
        image_input = gr.Image(type="pil", label="Charger Image")
        image_output = gr.Image(type="pil", label="Image Modifiée")
    submit_button = gr.Button("Appliquer")
    submit_button.click(image_processing, inputs=[image_input, operation, threshold, width, height, angle], outputs=image_output)

# Lancer l'application Gradio
demo.launch()
