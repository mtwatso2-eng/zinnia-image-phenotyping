from shiny import App, ui, render, reactive
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os
import time
import pandas as pd
import tensorflow as tf
from scipy import signal
from PIL import Image, ImageOps
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import base64
import asyncio
from io import BytesIO
import utils
import zinnia_image_analysis

app_ui = ui.page_fluid(
    ui.h2("Zinnia Image Phenotyping"),
    ui.tags.script(utils.fileIterator),
    ui.input_action_button("select_dir", "Select Directory", onclick="selectDirectory()"),
    ui.br(),
    ui.output_ui("image_display"),
    ui.output_ui("phenotype_summary"),
    ui.output_ui("completion_message"),
    ui.output_ui("processing_done"),  # Hidden output for JS signaling
    ui.download_button("downloadResults", "Download Phenotypes CSV")
)

def server(input, output, session):
    # Reactive values for storing data
    phenotypes = reactive.value(pd.DataFrame(columns=["image_name", "diameter", "meanColorR", "meanColorG", "meanColorB", "stemLength"]))
    processed_image = reactive.value(None)
    processing_done_counter = reactive.value(0)  # Counter for signaling

    @reactive.effect
    def keep_alive():
        # This will trigger every 5 seconds
        reactive.invalidate_later(5)
        print("Keeping session alive...")
    
    @reactive.effect
    async def process_current_image():
        if not input.current_image():
            processed_image.set(None)
            return
        
        current_index = input.current_index() + 1
        total_images = input.total_images()
        
        with ui.Progress(min=1, max=total_images) as p:
            p.set(current_index, 
                  message=f"Processing image {current_index} of {total_images}",
                  detail="Analyzing image for phenotypes...")
            
            # Convert base64 to image
            print("loading image")
            image_data = base64.b64decode(input.current_image())
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process image with phenotype analyzer
            try:
                img, phenotypes_df = zinnia_image_analysis.countImage(img, input.current_image_name())
                img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
                print(f"Processed image: {input.current_image_name()}, found {len(phenotypes_df)} phenotypes")
            except Exception as e:
                print(f"Error processing image {input.current_image_name()}: {str(e)}")
                img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
                phenotypes_df = pd.DataFrame(columns=["image_name", "diameter", "meanColorR", "meanColorG", "meanColorB", "stemLength"])

            with reactive.isolate():
                if not phenotypes_df.empty:
                    phenotypes.set(pd.concat([phenotypes.get(), phenotypes_df]))
                    print(f"Total phenotypes accumulated: {len(phenotypes.get())}")

            # Create figure for this image
            fig = plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(input.current_image_name())
            
            # Convert figure to base64
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode()
            
            processed_image.set(img_str)
            # Signal to frontend that processing is done
            with reactive.isolate():
                processing_done_counter.set(processing_done_counter.get() + 1)

    @output
    @render.ui
    def image_display():
        if processed_image.get() is None:
            return ui.p("Select a directory with images to begin analysis")
        return ui.img(src=f"data:image/png;base64,{processed_image.get()}")
    
    @output
    @render.ui
    def phenotype_summary():
        phenotypes_df = phenotypes.get()
        if phenotypes_df.empty:
            return ui.p("No phenotypes detected yet. Process images to see results.")
        
        total_phenotypes = len(phenotypes_df)
        flower_count = len(phenotypes_df[phenotypes_df['diameter'].notna()])
        stem_count = len(phenotypes_df[phenotypes_df['stemLength'].notna()])
        
        return ui.div(
            ui.h4("Phenotype Summary"),
            ui.p(f"Total phenotypes detected: {total_phenotypes}"),
            ui.p(f"Flowers: {flower_count}"),
            ui.p(f"Stems: {stem_count}"),
            style="margin: 20px 0; padding: 15px; background-color: #e8f4f8; border-radius: 5px; border-left: 4px solid #2196F3;"
        )

    @output
    @render.ui
    def completion_message():
        if input.show_completion():
            return ui.div(
                ui.h3("Processing Complete!", style="color: green;"),
                ui.p("All images have been processed."),
                style="margin: 20px 0; padding: 20px; background-color: #f0f0f0; border-radius: 5px;"
            )
        return None
    
    @output
    @render.ui
    def processing_done():
        # Hidden div for JS to observe
        return ui.div(str(processing_done_counter.get()), id="processing_done", style="display:none;")

    @render.download(filename="phenotypes.csv")
    async def downloadResults():
        await asyncio.sleep(0.25)
        yield phenotypes.get().to_csv(index=False)

app = App(app_ui, server)
