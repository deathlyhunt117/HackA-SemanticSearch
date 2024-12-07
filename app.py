import datasets
from transformers import AutoTokenizer, AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import numpy as np
from PIL import Image
import gradio as gr

# Load dataset directly from Hugging Face
try:
    print("[INFO] Loading dataset...")
    ds = datasets.load_dataset('arampacha/rsicd', split='train')
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    exit()

# Load CLIP model and tokenizer
model_name = "openai/clip-vit-base-patch32"
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)

# Initialize Qdrant client
try:
    client = QdrantClient("localhost", port=6333)  # Ensure Qdrant is running on this host and port
    print("[INFO] Client created...")
except Exception as e:
    print(f"[ERROR] Failed to create Qdrant client: {e}")
    exit()

# Create a Qdrant collection for storing satellite images
try:
    print("[INFO] Creating Qdrant data collection...")
    client.create_collection(
        collection_name="satellite_img_db",
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
    )
except Exception as e:
    print(f"[ERROR] Failed to create collection: {e}")
    exit()

# Populate the VectorDB with image embeddings
records = []
print("[INFO] Creating a data collection...")
for idx, sample in tqdm(enumerate(ds), total=len(ds)):
    processed_img = processor(text=None, images=sample['image'], return_tensors="pt")['pixel_values']
    
    # Get image embeddings
    img_embds = model.get_image_features(processed_img).detach().numpy().tolist()[0]
    
    # Prepare pixel data and size for storage
    img_px = list(sample['image'].getdata())
    img_size = sample['image'].size

    records.append(models.Record(id=idx, vector=img_embds, payload={"pixel_lst": img_px, "img_size": img_size, "captions": sample['captions']}))

# Upload records to Qdrant in chunks
print("[INFO] Uploading data records to data collection...")
for i in range(30, len(records), 30):
    try:
        client.upload_records(
            collection_name="satellite_img_db",
            records=records[i-30:i],
        )
        print(f"Finished uploading records up to {i}")
    except Exception as e:
        print(f"[ERROR] Failed to upload records: {e}")

print("[INFO] Successfully uploaded data records to data collection!")

# Function to process text input and retrieve images from Qdrant
def process_text(text):
    inp = tokenizer(text, return_tensors="pt")
    text_embeddings = model.get_text_features(**inp).detach().numpy().tolist()[0]

    hits = client.search(
        collection_name="satellite_img_db",
        query_vector=text_embeddings,
        limit=1,
    )

    if hits:
        for hit in hits:
            img_size = tuple(hit.payload['img_size'])
            pixel_lst = hit.payload['pixel_lst']
            new_image = Image.new("RGB", img_size)
            new_image.putdata(list(map(lambda x: tuple(x), pixel_lst)))
            return new_image

# Create Gradio interface for user input and image output
iface = gr.Interface(
    title="Semantic Search Over Satellite Images Using Qdrant Vector Database",
    description="by Neural Pirates",
    fn=process_text,
    inputs=gr.Textbox(label="Input prompt"),
    outputs=gr.Image(type="pil", label="Satellite Image"),
)

iface.launch()