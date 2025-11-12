
// Load images from JSON file
async function loadImagesFromJSON() {
    try {
        const response = await fetch('images.json');
        const images = await response.json();
        return images;
    } catch (error) {
        console.error('Error loading images:', error);
        return [];
    }
}

// Load common object images into grid
async function loadRandomImages() {
    const grid = document.getElementById('imageGrid');
    const commonImages = await loadImagesFromJSON();
    
    // Clear existing images
    grid.innerHTML = '';
    
    commonImages.forEach((image, index) => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.innerHTML = `
            <img src="${image.url}" alt="${image.name}" 
                 title="${image.name}" crossorigin="anonymous">
            <div class="image-info">
                <div class="result" id="result-${index}"></div>
            </div>
            <div class="loading" id="loading-${index}">
                <div class="spinner"></div>
                Analyzing...
            </div>
        `;
        
        card.addEventListener('click', () => classifyImage(card.querySelector('img').src, index));
        grid.appendChild(card);
    });
}

// Convert image to base64
function imageToBase64(imgElement, maxSize = 224) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Set canvas size to 224x224 for ResNet
        canvas.width = maxSize;
        canvas.height = maxSize;
        
        // Draw and resize image
        ctx.drawImage(imgElement, 0, 0, maxSize, maxSize);
        
        // Convert to base64
        const base64 = canvas.toDataURL('image/jpeg', 0.8);
        resolve(base64);
    });
}

// Load image from URL and convert to base64
function loadImageFromUrl(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = async () => {
            try {
                const base64 = await imageToBase64(img);
                resolve(base64);
            } catch (error) {
                reject(error);
            }
        };
        img.onerror = reject;
        img.src = url;
    });
}

// Make API call to classify image
async function callClassificationAPI(base64Image) {
    const apiUrl = "/cnn/search-image";
    const apiKey = document.getElementById('apiKey').value;
    const requestBody = {
        prompt: base64Image
    };

    const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
            'accept': 'application/json',
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error('API error response:', errorText);
        throw new Error(`API call failed: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const result = await response.json();
    
    return result;
}

// Classify image
async function classifyImage(imageUrl, index) {
    const loadingEl = document.getElementById(`loading-${index}`);
    const resultEl = document.getElementById(`result-${index}`);
    if (!loadingEl) {
        console.error(`Loading element not found: loading-${index}`);
        return;
    }
    
    if (!resultEl) {
        console.error(`Result element not found: result-${index}`);
        return;
    }
    
    // Show loading
    loadingEl.classList.add('active');
    resultEl.className = 'result';
    resultEl.style.display = 'none';
    try {
        // Convert image to base64
        const base64Image = await loadImageFromUrl(imageUrl);
        // Make API call
        const result = await callClassificationAPI(base64Image);
        // Check if result has expected structure
        if (!result || !result.image_data) {
            throw new Error('Invalid API response structure');
        }    
        // Display result - updated for exact API response format
        const prediction = result.image_data.top1_class_label;
        const probability = parseFloat(result.image_data.top1_class_probability).toFixed(2);
        resultEl.className = 'result success';
        resultEl.style.display = 'block'; // Force display
        resultEl.innerHTML = `
            <div class="prediction">🎯 ${prediction}</div>
            <div class="probability">Confidence: ${probability}%</div>
        `;
    } catch (error) {
        console.error('Classification error:', error);
        resultEl.className = 'result error';
        resultEl.innerHTML = `
            <div class="error-message">❌ Error: ${error.message}</div>
        `;
    } finally {
        // Hide loading
        loadingEl.classList.remove('active');
    }
}

// Initialize the demo
loadRandomImages();