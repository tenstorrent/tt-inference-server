// ─── Constants ───────────────────────────────────────────────────────────────
const API_URL = '/v1/images/generations';
const API_KEY = 'sdxl';
const HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`,
};
const $ = (id) => document.getElementById(id);

// ─── State ────────────────────────────────────────────────────────────────────
let isGenerating = false;
let currentGeneration = null; // { base64, prompt, negativePrompt, steps, guidance, seed, elapsed }
let activeObjectURLs = [];
let currentModalId = null;
let db = null;

// ─── XSS Safety ──────────────────────────────────────────────────────────────
function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// ─── Tab Navigation ───────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;

        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        document.querySelectorAll('.tab-content').forEach(s => s.style.display = 'none');
        $(`tab-${tab}`).style.display = 'block';

        if (tab === 'gallery') {
            renderGallery();
        }
    });
});

// ─── Random Seed Toggle ───────────────────────────────────────────────────────
$('random-seed').addEventListener('change', () => {
    $('seed').disabled = $('random-seed').checked;
    if ($('random-seed').checked) {
        $('seed').value = '';
    }
});
// Initialize disabled state
$('seed').disabled = $('random-seed').checked;


// ─── Parameter Collection & Validation ───────────────────────────────────────
function clearErrors() {
    ['prompt-error', 'steps-error', 'guidance-error'].forEach(id => {
        $(id).textContent = '';
    });
}

function collectParams() {
    clearErrors();

    const prompt = $('prompt').value.trim();
    const negativePrompt = $('negative-prompt').value.trim();
    const steps = parseInt($('steps').value, 10);
    const guidance = parseFloat($('guidance').value);
    const useRandomSeed = $('random-seed').checked;
    const seedVal = $('seed').value.trim();
    let valid = true;

    if (!prompt) {
        $('prompt-error').textContent = 'Prompt is required.';
        valid = false;
    }

    if (isNaN(steps) || steps < 12 || steps > 50) {
        $('steps-error').textContent = 'Steps must be between 12 and 50.';
        valid = false;
    }

    if (isNaN(guidance) || guidance < 1.0 || guidance > 20.0) {
        $('guidance-error').textContent = 'Guidance scale must be between 1.0 and 20.0.';
        valid = false;
    }

    if (!valid) return null;

    const body = {
        prompt,
        num_inference_steps: steps,
        guidance_scale: guidance,
    };

    if (negativePrompt) body.negative_prompt = negativePrompt;

    if (!useRandomSeed && seedVal !== '') {
        const seed = parseInt(seedVal, 10);
        if (!isNaN(seed)) body.seed = seed;
    }

    return body;
}

// ─── Generate Image ────────────────────────────────────────────────────────────
$('generate-btn').addEventListener('click', generateImage);

document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const generateTab = $('tab-generate');
        if (generateTab.style.display !== 'none') {
            generateImage();
        }
    }
});

async function generateImage() {
    if (isGenerating) return;

    const params = collectParams();
    if (!params) return;

    isGenerating = true;
    $('generate-btn').disabled = true;
    $('generate-btn').closest('.btn-3d').classList.add('is-disabled');

    // Show loading overlay
    $('image-placeholder').style.display = 'none';
    $('generated-image').style.display = 'none';
    $('loading-overlay').style.display = 'flex';
    $('image-actions').style.display = 'none';
    $('metadata-section').style.display = 'none';

    const startTime = Date.now();
    const timerEl = $('loading-timer');
    timerEl.textContent = '0s';
    const timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        timerEl.textContent = `${elapsed}s`;
    }, 1000);

    try {
        const res = await fetch(API_URL, {
            method: 'POST',
            headers: HEADERS,
            body: JSON.stringify(params),
        });

        clearInterval(timerInterval);
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

        if (!res.ok) {
            let errMsg;
            if (res.status === 401) {
                errMsg = 'Invalid API key.';
            } else if (res.status === 422) {
                const err = await res.json().catch(() => ({}));
                errMsg = `Invalid parameters: ${err.detail || JSON.stringify(err)}`;
            } else if (res.status >= 500) {
                errMsg = 'Server error. Please try again.';
            } else {
                errMsg = `Request failed (HTTP ${res.status}).`;
            }
            showGenerationError(errMsg);
            return;
        }

        const data = await res.json();
        const base64 = data.images[0];

        const imgEl = $('generated-image');
        imgEl.src = `data:image/jpeg;base64,${base64}`;
        imgEl.style.display = 'block';
        $('loading-overlay').style.display = 'none';

        // Store current generation
        currentGeneration = {
            base64,
            prompt: params.prompt,
            negativePrompt: params.negative_prompt || '',
            steps: params.num_inference_steps,
            guidance: params.guidance_scale,
            seed: params.seed != null ? params.seed : 'random',
            resolution: '512×512',
            elapsed: `${elapsed}s`,
        };

        displayMetadata(currentGeneration);
        $('image-actions').style.display = 'flex';
        $('metadata-section').style.display = 'block';

    } catch (err) {
        clearInterval(timerInterval);
        showGenerationError('Could not reach server. Check your connection.');
    } finally {
        isGenerating = false;
        $('generate-btn').disabled = false;
        $('generate-btn').closest('.btn-3d').classList.remove('is-disabled');
    }
}

function showGenerationError(msg) {
    $('loading-overlay').style.display = 'none';
    $('image-placeholder').style.display = 'flex';
    const placeholderText = $('image-placeholder').querySelector('p');
    placeholderText.textContent = msg;
    placeholderText.style.color = '#F54E00';
}

function displayMetadata(gen) {
    $('meta-prompt').textContent = gen.prompt;
    $('meta-steps').textContent = gen.steps;
    $('meta-guidance').textContent = gen.guidance;
    $('meta-seed').textContent = gen.seed;
    $('meta-resolution').textContent = gen.resolution;
    $('meta-time').textContent = gen.elapsed;
}

// ─── Download ─────────────────────────────────────────────────────────────────
$('download-btn').addEventListener('click', () => downloadAsPng($('generated-image'), 'sdxl-output.png'));

function downloadAsPng(imgEl, filename) {
    const canvas = document.createElement('canvas');
    canvas.width = imgEl.naturalWidth || imgEl.width;
    canvas.height = imgEl.naturalHeight || imgEl.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgEl, 0, 0);
    const url = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
}

// ─── Fullscreen ───────────────────────────────────────────────────────────────
$('fullscreen-btn').addEventListener('click', () => {
    const container = $('image-container');
    if (container.requestFullscreen) {
        container.requestFullscreen();
    } else if (container.webkitRequestFullscreen) {
        container.webkitRequestFullscreen();
    }
});

// ─── Save to Gallery ──────────────────────────────────────────────────────────
$('save-gallery-btn').addEventListener('click', async () => {
    if (!currentGeneration) return;
    const btn = $('save-gallery-btn');
    const btnWrap = btn.closest('.btn-3d');
    btn.disabled = true;
    if (btnWrap) btnWrap.classList.add('is-disabled');
    btn.textContent = 'Saving...';
    try {
        await saveToGallery(currentGeneration);
        btn.textContent = 'Saved!';
        setTimeout(() => {
            btn.textContent = 'Save to Gallery';
            btn.disabled = false;
            if (btnWrap) btnWrap.classList.remove('is-disabled');
        }, 2000);
    } catch (e) {
        btn.textContent = 'Save Failed';
        setTimeout(() => {
            btn.textContent = 'Save to Gallery';
            btn.disabled = false;
            if (btnWrap) btnWrap.classList.remove('is-disabled');
        }, 2000);
    }
});

// ─── IndexedDB ────────────────────────────────────────────────────────────────
const DB_NAME = 'sdxl-gallery';
const DB_STORE = 'images';
const DB_VERSION = 1;

function openDB() {
    return new Promise((resolve, reject) => {
        if (db) { resolve(db); return; }
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = (e) => {
            const d = e.target.result;
            if (!d.objectStoreNames.contains(DB_STORE)) {
                const store = d.createObjectStore(DB_STORE, { keyPath: 'id', autoIncrement: true });
                store.createIndex('timestamp', 'timestamp', { unique: false });
            }
        };
        req.onsuccess = (e) => { db = e.target.result; resolve(db); };
        req.onerror = () => reject(req.error);
    });
}

async function saveToGallery(gen) {
    const d = await openDB();
    const blob = base64ToBlob(gen.base64, 'image/jpeg');
    return new Promise((resolve, reject) => {
        const tx = d.transaction(DB_STORE, 'readwrite');
        const store = tx.objectStore(DB_STORE);
        const entry = {
            blob,
            prompt: gen.prompt,
            negative_prompt: gen.negativePrompt,
            steps: gen.steps,
            guidance: gen.guidance,
            seed: gen.seed,
            elapsed: gen.elapsed,
            timestamp: Date.now(),
        };
        const req = store.add(entry);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

function base64ToBlob(base64, mimeType) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Blob([bytes], { type: mimeType });
}

async function getAllGalleryItems() {
    const d = await openDB();
    return new Promise((resolve, reject) => {
        const tx = d.transaction(DB_STORE, 'readonly');
        const store = tx.objectStore(DB_STORE);
        const index = store.index('timestamp');
        const results = [];
        const req = index.openCursor(null, 'prev'); // newest first
        req.onsuccess = (e) => {
            const cursor = e.target.result;
            if (cursor) {
                results.push(cursor.value);
                cursor.continue();
            } else {
                resolve(results);
            }
        };
        req.onerror = () => reject(req.error);
    });
}

async function getGalleryItem(id) {
    const d = await openDB();
    return new Promise((resolve, reject) => {
        const tx = d.transaction(DB_STORE, 'readonly');
        const req = tx.objectStore(DB_STORE).get(id);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function deleteGalleryItem(id) {
    const d = await openDB();
    return new Promise((resolve, reject) => {
        const tx = d.transaction(DB_STORE, 'readwrite');
        const req = tx.objectStore(DB_STORE).delete(id);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
    });
}

// ─── Gallery Rendering ────────────────────────────────────────────────────────
async function renderGallery() {
    // Revoke previous object URLs to prevent memory leaks
    activeObjectURLs.forEach(url => URL.revokeObjectURL(url));
    activeObjectURLs = [];

    const grid = $('gallery-grid');
    const emptyEl = $('gallery-empty');
    grid.innerHTML = '';

    let items;
    try {
        items = await getAllGalleryItems();
    } catch (e) {
        grid.innerHTML = '<p style="color:#F54E00">Failed to load gallery.</p>';
        return;
    }

    if (items.length === 0) {
        emptyEl.style.display = 'flex';
        grid.style.display = 'none';
    } else {
        emptyEl.style.display = 'none';
        grid.style.display = 'grid';

        for (const item of items) {
            const objUrl = URL.createObjectURL(item.blob);
            activeObjectURLs.push(objUrl);

            const card = document.createElement('div');
            card.className = 'gallery-card';
            card.addEventListener('click', () => openModal(item.id));

            const thumb = document.createElement('img');
            thumb.className = 'gallery-card-thumb';
            thumb.src = objUrl;
            thumb.alt = 'Gallery thumbnail';

            const info = document.createElement('div');
            info.className = 'gallery-card-info';

            const promptEl = document.createElement('div');
            promptEl.className = 'gallery-card-prompt';
            promptEl.textContent = item.prompt;

            const badge = document.createElement('div');
            badge.className = 'gallery-card-badge';
            badge.textContent = `${item.steps} steps · cfg ${item.guidance}`;

            const ts = document.createElement('div');
            ts.className = 'gallery-card-timestamp';
            ts.textContent = new Date(item.timestamp).toLocaleString();

            info.appendChild(promptEl);
            info.appendChild(badge);
            info.appendChild(ts);
            card.appendChild(thumb);
            card.appendChild(info);
            grid.appendChild(card);
        }
    }

    // Storage estimate
    if (navigator.storage && navigator.storage.estimate) {
        try {
            const est = await navigator.storage.estimate();
            const used = (est.usage / 1024 / 1024).toFixed(1);
            const quota = (est.quota / 1024 / 1024 / 1024).toFixed(1);
            $('storage-usage').textContent = `Storage: ${used} MB used of ${quota} GB`;
        } catch (_) {}
    }
}

// ─── Gallery Modal ────────────────────────────────────────────────────────────
async function openModal(id) {
    currentModalId = id;
    let item;
    try {
        item = await getGalleryItem(id);
    } catch (e) {
        return;
    }
    if (!item) return;

    const objUrl = URL.createObjectURL(item.blob);
    activeObjectURLs.push(objUrl);

    $('modal-image').src = objUrl;

    const metaEl = $('modal-metadata');
    const rows = [
        ['Prompt', item.prompt],
        ['Negative Prompt', item.negative_prompt || '—'],
        ['Steps', item.steps],
        ['Guidance Scale', item.guidance],
        ['Seed', item.seed],
        ['Generation Time', item.elapsed],
        ['Saved', new Date(item.timestamp).toLocaleString()],
    ];
    metaEl.innerHTML = rows.map(([k, v]) =>
        `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(String(v))}</dd>`
    ).join('');

    $('modal-overlay').style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    $('modal-overlay').style.display = 'none';
    $('modal-image').src = '';
    document.body.style.overflow = '';
    currentModalId = null;
}

$('modal-close').addEventListener('click', closeModal);

$('modal-overlay').addEventListener('click', (e) => {
    if (e.target === $('modal-overlay')) closeModal();
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && $('modal-overlay').style.display !== 'none') {
        closeModal();
    }
});

$('modal-download-btn').addEventListener('click', () => {
    const imgEl = $('modal-image');
    if (imgEl.src) downloadAsPng(imgEl, 'sdxl-gallery.png');
});

$('modal-delete-btn').addEventListener('click', async () => {
    if (currentModalId == null) return;
    if (!confirm('Delete this image from your gallery?')) return;
    try {
        await deleteGalleryItem(currentModalId);
        closeModal();
        renderGallery();
    } catch (e) {
        alert('Failed to delete image.');
    }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
openDB().catch(e => console.warn('IndexedDB unavailable:', e));
