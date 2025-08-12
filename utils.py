fileIterator = """
        let imageFiles = [];
        let currentImageIndex = 0;
        let autoPlayInterval = null;
        const AUTO_PLAY_DELAY = 1000; // 1 second between images
        
        async function selectDirectory() {
            try {
                const dirHandle = await window.showDirectoryPicker();
                imageFiles = [];
                
                for await (const entry of dirHandle.values()) {
                    if (entry.kind === 'file') {
                        const name = entry.name.toLowerCase();
                        if (name.endsWith('.jpg') || name.endsWith('.jpeg') || 
                            name.endsWith('.png') || name.endsWith('.bmp')) {
                            imageFiles.push(entry);
                        }
                    }
                }
                
                // Sort the image files using natural sorting (handles numbers correctly)
                imageFiles.sort((a, b) => {
                    return a.name.localeCompare(b.name, undefined, {
                        numeric: true,
                        sensitivity: 'base'
                    });
                });
                
                if (imageFiles.length > 0) {
                    Shiny.setInputValue('total_images', imageFiles.length);
                    currentImageIndex = 0;
                    loadCurrentImage();
                    // No autoplay, wait for processing_done signal
                }
            } catch (err) {
                console.error('Error selecting directory:', err);
            }
        }
        
        async function loadCurrentImage() {
            if (imageFiles.length === 0) return;
            
            const file = await imageFiles[currentImageIndex].getFile();
            // 12MB = 12 * 1024 * 1024 bytes
            if (file.size > 12 * 1024 * 1024) {
                // Create a 1024x1024 black PNG as a base64 string
                const canvas = document.createElement('canvas');
                canvas.width = 1024;
                canvas.height = 1024;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = 'black';
                ctx.fillRect(0, 0, 1024, 1024);
                canvas.toBlob(function(blob) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const base64 = e.target.result.split(',')[1];
                        Shiny.setInputValue('current_image', base64);
                        Shiny.setInputValue('current_index', currentImageIndex);
                        Shiny.setInputValue('current_image_name', file.name);
                    };
                    reader.readAsDataURL(blob);
                }, 'image/png');
                return;
            }
            const reader = new FileReader();
            reader.onload = function(e) {
                const base64 = e.target.result.split(',')[1];
                Shiny.setInputValue('current_image', base64);
                Shiny.setInputValue('current_index', currentImageIndex);
                Shiny.setInputValue('current_image_name', file.name);
            };
            reader.readAsDataURL(file);
        }
        
        function nextImage() {
            if (currentImageIndex < imageFiles.length - 1) {
                currentImageIndex++;
                console.log(currentImageIndex);
                loadCurrentImage();
                console.log(currentImageIndex);
            } else {
                Shiny.setInputValue('show_completion', true);
            }
        }
        
        // MutationObserver to watch for processing_done changes
        function setupProcessingDoneObserver() {
            const target = document.getElementById('processing_done');
            if (!target) return;
            let lastValue = target.textContent;
            const observer = new MutationObserver(function(mutationsList) {
                for (const mutation of mutationsList) {
                    if (mutation.type === 'childList') {
                        const newValue = target.textContent;
                        if (newValue !== lastValue) {
                            lastValue = newValue;
                            // Only advance if not at last image
                            if (currentImageIndex < imageFiles.length - 1) {
                                nextImage();
                            }
                        }
                    }
                }
            });
            observer.observe(target, { childList: true });
        }
        
        // Wait for DOMContentLoaded to set up observer
        document.addEventListener('DOMContentLoaded', function() {
            setupProcessingDoneObserver();
        });
    """

def rotate_to_landscape(img):
    height, width = img.shape[:2]
    if height > width:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img