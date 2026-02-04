import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from collections import Counter
import re
import yake
import nltk
import seaborn as sns
import argparse
nltk.download('stopwords')
from nltk.corpus import stopwords


def get_model_size_in_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())

def preprocess_text(text):
    words = re.findall(r'\w+', text.lower())
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]

def load_model(device):
    model_name = "Salesforce/blip2-flan-t5-xl"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)

    model_size_bytes = get_model_size_in_bytes(model)
    model_size_gb = model_size_bytes / (1024 ** 3)
    print(f"DEVICE: {device} | MODEL SIZE: {model_size_gb:.2f} GB")

    return processor, model, device

def process_image(image, processor, model, device, prompt=None):
    if prompt:
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    else:
        inputs = processor(images=image, return_tensors="pt").to(device)
    
    output = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(output[0], skip_special_tokens=True)

def set_style():
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("colorblind")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24

def create_histogram(words, frequencies, output_path):
    plt.figure(figsize=(10, 10))
    colors = sns.color_palette("colorblind")
    base_color = colors[0]
    bars = plt.bar(range(len(words)), frequencies, color=base_color)
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.xlabel('Words', fontsize=32, labelpad=20)
    plt.ylabel('Frequency (%)', fontsize=32, labelpad=20)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.margins(x=0.01, y=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close()

def analyze_images(batch_directory, output_directory, prompt=None, device="cuda:0"):
    processor, model, device = load_model(device)
    all_batch_files = sorted([f for f in os.listdir(batch_directory) if f.endswith('.npy')])
    num_classes = len([f for f in all_batch_files if 'class_' in f])
    print(f"Number of classes: {num_classes}")
    class_files = [f"class_{i}" for i in range(num_classes)]

    for class_file in tqdm(class_files, desc="Processing class files"):
        batch_files = [f for f in all_batch_files if class_file in f]
        all_words = []
        all_captions = []

        class_output_dir = os.path.join(output_directory, class_file)
        os.makedirs(class_output_dir, exist_ok=True)
        counter = 1

        for batch_file in tqdm(batch_files, desc="Processing batch files"):
            images = np.load(os.path.join(batch_directory, batch_file))
            for img in images:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = np.clip(img * std + mean, 0, 1) * 255
                img = img.astype(np.uint8)
                image = Image.fromarray(img).convert("RGB")

                captions = process_image(image, processor, model, device, prompt)
                print(f"Generated Keywords/Captions for Image {counter} in {class_file}:\n{captions}\n")
                counter += 1

                all_captions.append(captions)
                words = preprocess_text(captions)
                all_words.extend(words)

        # Word frequency analysis
        word_freq = Counter(all_words)
        
        with open(os.path.join(class_output_dir, f"word_frequencies.txt"), "w") as f:
            for word, freq in word_freq.most_common():
                f.write(f"{word}: {freq}\n")

        # Generate histogram
        set_style()
        top_words = word_freq.most_common(10)
        words, frequencies = zip(*top_words)
        frequencies_percentage = [(f / sum(frequencies)) * 100 for f in frequencies]
        histogram_path = os.path.join(class_output_dir, f"{class_file}_word_histogram.png")
        create_histogram(words, frequencies_percentage, histogram_path)
        print(f"    Saved histogram at {histogram_path}")

        # Captions
        combined_captions = ". ".join(all_captions)
        with open(os.path.join(class_output_dir, f"captions.txt"), "w") as f:
            f.write(combined_captions)
            
        # YAKE keyword extraction
        kw_extractor = yake.KeywordExtractor()
        keywords = kw_extractor.extract_keywords(combined_captions)
        with open(os.path.join(class_output_dir, f"yake_captions_keywords.txt"), "w") as f:
            for kw, score in keywords:
                f.write(f"{kw}: {score}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for captioning and word frequency analysis.")
    parser.add_argument('batch_directory', type=str, help="Directory containing batches of images in .npy format")
    parser.add_argument('--prompt', type=str, default="Describe the image.", help="Optional prompt for image captioning")
    parser.add_argument('--device', type=str, default="cpu", help="Device to run the model")
    args = parser.parse_args()

    batch_directory = args.batch_directory
    output_directory = os.path.join(os.getcwd(), "Captioning")
    prompt = args.prompt
    device = args.device

    os.makedirs(output_directory, exist_ok=True)
    analyze_images(batch_directory, output_directory, prompt, device)


# python captions_generator.py /path/to/batch_directory/withe/images.npy
