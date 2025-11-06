
// Generate 12 images of common objects/animals
function generateRandomImages() {
    const commonObjects = [
        {
            url: "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=300&h=200&fit=crop&crop=center",
            name: "Dog"
        },
        {
            url: "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=300&h=200&fit=crop&crop=center",
            name: "Cat"
        },
        {
            url: "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=300&h=200&fit=crop&crop=center",
            name: "Car"
        },
        {
            url: "https://images.unsplash.com/photo-1540962351504-03099e0a754b?w=300&h=200&fit=crop&crop=center",
            name: "Airplane"
        },
        {
            url: "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=300&h=200&fit=crop&crop=center",
            name: "Ball"
        },
        {
            url: "https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=300&h=200&fit=crop&crop=center",
            name: "Bird"
        },
        {
            url: "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=300&h=200&fit=crop&crop=center",
            name: "Bicycle"
        },
        {
            url: "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=300&h=200&fit=crop&crop=center",
            name: "House"
        },
        {
            url: "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=300&h=200&fit=crop&crop=center",
            name: "Apple"
        },
        {
            url: "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=300&h=200&fit=crop&crop=center",
            name: "Coffee Cup"
        },
        {
            url: "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=300&h=200&fit=crop&crop=center",
            name: "Flower"
        },
        {
            url: "https://images.unsplash.com/photo-1541781774459-bb2af2f05b55?w=300&h=200&fit=crop&crop=center",
            name: "Book"
        }
    ];
    return commonObjects;
}

// Load common object images into grid
function loadRandomImages() {
    const grid = document.getElementById('imageGrid');
    const commonImages = generateRandomImages();
    
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
    const apiUrl = document.getElementById('apiUrl').value;
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

// Handle file upload
document.getElementById('fileInput').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const uploadResult = document.getElementById('uploadResult');
    
    // Show uploaded image
    const reader = new FileReader();
    reader.onload = async (e) => {
        const img = new Image();
        img.onload = async () => {
            uploadResult.innerHTML = `
                <img src="${e.target.result}" class="uploaded-image" alt="Uploaded image">
                <div class="loading active" id="upload-loading">
                    <div class="spinner"></div>
                    Analyzing uploaded image...
                </div>
                <div class="result" id="upload-result"></div>
            `;

            try {
                // Convert to base64 with 224x224 size
                const base64Image = await imageToBase64(img);
                
                // Make API call
                const result = await callClassificationAPI(base64Image);
                
                // Display result - updated for exact API response format
                const prediction = result.image_data.top1_class_label;
                const probability = parseFloat(result.image_data.top1_class_probability).toFixed(2);
                
                document.getElementById('upload-loading').classList.remove('active');
                const resultEl = document.getElementById('upload-result');
                resultEl.className = 'result success';
                resultEl.innerHTML = `
                    <div class="prediction">🎯 Prediction: ${prediction}</div>
                    <div class="probability">Confidence: ${probability}%</div>
                `;
                
            } catch (error) {
                console.error('Upload classification error:', error);
                document.getElementById('upload-loading').classList.remove('active');
                const resultEl = document.getElementById('upload-result');
                resultEl.className = 'result error';
                resultEl.innerHTML = `
                    <div class="error-message">❌ Error: ${error.message}</div>
                `;
            }
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
});

// Initialize the demo
loadRandomImages();