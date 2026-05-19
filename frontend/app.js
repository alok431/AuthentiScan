/**
 * DeepGuard AI - Frontend Application
 * Handles image upload, API communication, and results rendering
 */

const API_BASE = 'https://authentiscan-1-u1vl.onrender.com';
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// DOM Elements
const uploadZone = $('#upload-zone');
const fileInput = $('#file-input');
const previewContainer = $('#preview-container');
const previewImage = $('#preview-image');
const previewInfo = $('#preview-info');
const previewScanning = $('#preview-scanning');
const btnAnalyze = $('#btn-analyze');
const btnClear = $('#btn-clear');
const resultsEmpty = $('#results-empty');
const resultsLoading = $('#results-loading');
const resultsContent = $('#results-content');
const loadingStep = $('#loading-step');
const progressBar = $('#progress-bar');

let currentFile = null;

// ===== Scroll Effects =====
window.addEventListener('scroll', () => {
    const navbar = $('#navbar');
    navbar.classList.toggle('scrolled', window.scrollY > 50);

    // Update active nav link
    const sections = ['hero', 'detector', 'how-it-works', 'stats'];
    const scrollPos = window.scrollY + 200;
    sections.forEach(id => {
        const section = document.getElementById(id);
        if (section) {
            const top = section.offsetTop;
            const height = section.offsetHeight;
            const link = $(`.nav-link[href="#${id}"]`);
            if (link) {
                link.classList.toggle('active', scrollPos >= top && scrollPos < top + height);
            }
        }
    });
});

// ===== Counter Animation =====
const animateCounters = () => {
    $$('.counter').forEach(counter => {
        const target = parseInt(counter.dataset.target);
        let current = 0;
        const step = target / 40;
        const timer = setInterval(() => {
            current += step;
            if (current >= target) {
                counter.textContent = target;
                clearInterval(timer);
            } else {
                counter.textContent = Math.floor(current);
            }
        }, 30);
    });
};

// Intersection Observer for counters
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            animateCounters();
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

const statsSection = $('#stats');
if (statsSection) observer.observe(statsSection);

// ===== File Upload =====
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) {
        const isImageMime = file.type.startsWith('image/');
        const isNoMime = file.type === '';
        const ext = file.name.split('.').pop().toLowerCase();
        const isImageExt = ['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff'].includes(ext);
        
        if (isImageMime || isImageExt || isNoMime) {
            handleFile(file);
        } else {
            alert('Please drop a valid image file (PNG, JPG, BMP, WebP, TIFF).');
        }
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadZone.style.display = 'none';
        previewContainer.style.display = 'block';
        
        // Show file info
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        previewInfo.innerHTML = `
            <strong>${file.name}</strong> &bull; ${sizeMB} MB &bull; ${file.type}
        `;
    };
    reader.readAsDataURL(file);
    
    // Reset results
    showState('empty');
}

btnClear.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    previewContainer.style.display = 'none';
    uploadZone.style.display = 'block';
    showState('empty');
});

// ===== Analysis =====
const LOADING_STEPS = [
    'Initializing forensic analysis...',
    'Running Error Level Analysis...',
    'Analyzing noise patterns...',
    'Computing frequency spectrum...',
    'Checking color consistency...',
    'Detecting edge artifacts...',
    'Analyzing texture patterns...',
    'Examining metadata...',
    'Computing final verdict...'
];

btnAnalyze.addEventListener('click', async () => {
    if (!currentFile) return;
    
    showState('loading');
    previewScanning.classList.add('active');
    btnAnalyze.disabled = true;

    // Animate loading steps
    let stepIndex = 0;
    const stepInterval = setInterval(() => {
        if (stepIndex < LOADING_STEPS.length) {
            loadingStep.textContent = LOADING_STEPS[stepIndex];
            progressBar.style.width = `${((stepIndex + 1) / LOADING_STEPS.length) * 100}%`;
            stepIndex++;
        }
    }, 400);

    try {
        const formData = new FormData();
        formData.append('image', currentFile);

        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        clearInterval(stepInterval);
        progressBar.style.width = '100%';
        loadingStep.textContent = 'Analysis complete!';

        if (data.success) {
            setTimeout(() => renderResults(data), 500);
        } else {
            alert('Analysis failed: ' + (data.error || 'Unknown error'));
            showState('empty');
        }
    } catch (err) {
        clearInterval(stepInterval);
        alert('Connection error. Make sure the backend server is running on port 5000.');
        showState('empty');
    } finally {
        previewScanning.classList.remove('active');
        btnAnalyze.disabled = false;
    }
});

function showState(state) {
    resultsEmpty.style.display = state === 'empty' ? 'flex' : 'none';
    resultsLoading.style.display = state === 'loading' ? 'flex' : 'none';
    resultsContent.style.display = state === 'results' ? 'block' : 'none';
}

// ===== Render Results =====
function renderResults(data) {
    showState('results');

    const { verdict, analysis, image_info } = data;
    const verdictCard = $('#verdict-card');
    const scorePercent = Math.round(verdict.score * 100);

    // Determine verdict class
    let vClass = 'uncertain';
    let vIcon = '⚠️';
    if (verdict.verdict === 'REAL' || verdict.verdict === 'LIKELY REAL') {
        vClass = 'real';
        vIcon = '✅';
    } else if (verdict.verdict === 'FAKE' || verdict.verdict === 'LIKELY FAKE') {
        vClass = 'fake';
        vIcon = '🚨';
    }

    verdictCard.className = `verdict-card ${vClass}`;
    $('#verdict-icon').textContent = vIcon;
    $('#verdict-label').textContent = verdict.verdict;
    $('#verdict-sub').textContent = `Confidence: ${Math.round(verdict.confidence)}% | Risk Level: ${verdict.risk_level}`;

    // Animate meter
    setTimeout(() => {
        $('#meter-fill').style.width = `${scorePercent}%`;
        $('#meter-indicator').style.left = `calc(${scorePercent}% - 8px)`;
        $('#score-value').textContent = `${scorePercent}%`;
    }, 100);

    // Analysis breakdown
    const grid = $('#analysis-grid');
    grid.innerHTML = '';
    
    const analysisOrder = ['ela', 'noise', 'frequency', 'color', 'edges', 'texture', 'metadata'];
    analysisOrder.forEach(key => {
        if (!analysis[key]) return;
        const item = analysis[key];
        const pct = Math.round(item.score * 100);
        const level = pct < 40 ? 'low' : pct < 65 ? 'medium' : 'high';
        
        const el = document.createElement('div');
        el.className = 'analysis-item';
        el.innerHTML = `
            <span class="analysis-name">${item.name}</span>
            <div class="analysis-bar-wrap">
                <div class="analysis-bar ${level}" style="width: 0%"></div>
            </div>
            <span class="analysis-score" style="color: var(--${level === 'low' ? 'green' : level === 'medium' ? 'orange' : 'red'})">${pct}%</span>
        `;
        grid.appendChild(el);
        
        // Animate bar
        setTimeout(() => {
            el.querySelector('.analysis-bar').style.width = `${pct}%`;
        }, 200);
    });

    // Visualizations
    const visGrid = $('#vis-grid');
    visGrid.innerHTML = '';
    
    if (analysis.ela && analysis.ela.visualization) {
        visGrid.innerHTML += `
            <div class="vis-card">
                <img src="${analysis.ela.visualization}" alt="ELA Visualization">
                <div class="vis-card-label">Error Level Analysis</div>
            </div>`;
    }
    if (analysis.frequency && analysis.frequency.visualization) {
        visGrid.innerHTML += `
            <div class="vis-card">
                <img src="${analysis.frequency.visualization}" alt="Frequency Spectrum">
                <div class="vis-card-label">Frequency Spectrum</div>
            </div>`;
    }

    // Image Details
    const detailsGrid = $('#details-grid');
    detailsGrid.innerHTML = '';
    
    const details = [
        ['Filename', image_info.filename],
        ['Dimensions', image_info.size],
        ['Format', image_info.format],
        ['File Size', formatBytes(image_info.file_size)],
        ['Color Mode', image_info.mode],
        ['Has EXIF', analysis.metadata?.details?.has_exif ? 'Yes' : 'No'],
        ['Has Camera', analysis.metadata?.details?.has_camera ? 'Yes' : 'No'],
        ['Timestamp', new Date(data.timestamp).toLocaleString()]
    ];

    details.forEach(([label, value]) => {
        detailsGrid.innerHTML += `
            <div class="detail-item">
                <span class="detail-label">${label}</span>
                <span class="detail-value">${value}</span>
            </div>`;
    });
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(2) + ' MB';
}

// ===== Smooth Scroll for nav links =====
$$('.nav-link, .btn[href^="#"]').forEach(link => {
    link.addEventListener('click', (e) => {
        const href = link.getAttribute('href');
        if (href && href.startsWith('#')) {
            e.preventDefault();
            const target = document.getElementById(href.slice(1));
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// Start scan button
const btnStartScan = $('#btn-start-scan');
if (btnStartScan) {
    btnStartScan.addEventListener('click', (e) => {
        e.preventDefault();
        document.getElementById('detector').scrollIntoView({ behavior: 'smooth' });
    });
}

console.log('🛡️ DeepGuard AI initialized');
