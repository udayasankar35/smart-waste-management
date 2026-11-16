/* ═══════════════════════════════════════════════════════════════════════════════
   EcoSort — Smart Recycling & Waste-Segregation Assistant
   Frontend Application Logic · Multimodal Interactions & Motion System
   Bharat AI-SoC 2026–27 · PS2
   ═══════════════════════════════════════════════════════════════════════════════ */

// ─── Application State ────────────────────────────────────────────────────────
let currentMode          = 'gallery';
let webcamActive         = false;
let isAnalyzing          = false;
let currentImageB64      = null;
let speechRecognizer     = null;
let isRecordingVoice     = false;
let lastInputWasVoice    = false;   // Track if last input was via microphone
let currentUtterance     = null;    // Track current TTS utterance for stop control
let loadingIntervalId    = null;
let currentDetectionData = null; // Cache latest detection result for detailed fullscreen inspection

// Lightbox modal zoom & pan state
let lightboxZoom         = 1.0;
let lightboxPanX         = 0;
let lightboxPanY         = 0;
let isLightboxPanning    = false;
let lightboxStartX       = 0;
let lightboxStartY       = 0;
let didLightboxDrag      = false;
let lightboxMouseDownX   = 0;
let lightboxMouseDownY   = 0;

// Waste visual mappings
const WASTE_ICONS = {
    plastic:     '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a4 4 0 0 0-4 4v2H7a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V10a2 2 0 0 0-2-2h-1V6a4 4 0 0 0-4-4z"></path></svg>',
    recyclable:  '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"></polyline><polyline points="23 20 23 14 17 14"></polyline><path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"></path></svg>',
    metal:       '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>',
    dry_waste:   '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>',
    wet_waste:   '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a9 9 0 0 1 9 9c0 5-4 9-9 9s-9-4-9-9a9 9 0 0 1 9-9z"></path></svg>',
    hazardous:   '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
    glass:       '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 2h8l2 9H6l2-9z"></path><path d="M6 11v8a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3v-8"></path></svg>',
    paper:       '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path></svg>',
    cardboard:   '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>',
    organic:     '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a9 9 0 0 1 9 9c0 5-4 9-9 9s-9-4-9-9a9 9 0 0 1 9-9z"></path></svg>',
    e_waste:     '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="5" y="2" width="14" height="20" rx="2" ry="2"></rect><line x1="12" y1="18" x2="12.01" y2="18"></line></svg>',
    medical:     '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 6v12m-6-6h12"></path><rect x="3" y="3" width="18" height="18" rx="2"></rect></svg>',
    textile:     '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.38 3.46L16 2a4 4 0 0 1-8 0L3.62 3.46a2 2 0 0 0-1.34 2.23l.58 3.47a1 1 0 0 0 .99.84H6v10a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V10h2.15a1 1 0 0 0 .99-.84l.58-3.47a2 2 0 0 0-1.34-2.23z"></path></svg>',
    default:     '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>'
};

const WASTE_COLORS = {
    plastic:     '#10b981',
    recyclable:  '#34d399',
    metal:       '#94a3b8',
    dry_waste:   '#a78bfa',
    wet_waste:   '#38bdf8',
    hazardous:   '#f43f5e',
    glass:       '#67e8f9',
    paper:       '#fbbf24',
    cardboard:   '#d97706',
    organic:     '#4ade80',
    e_waste:     '#818cf8',
    medical:     '#fb7185',
    textile:     '#c084fc',
    default:     '#64748b',
};

// Rich Category Knowledge for Modal Drawer
const CATEGORY_DETAILS = {
    plastic: {
        title: "Plastic & Polymer Recycling",
        badge: "Blue Bin · Dry Recyclable",
        status: "Widely Recyclable (Codes #1, #2, #5)",
        decomposition: "450 – 1,000 years",
        description: "Plastics are petroleum-based polymers classified into 7 Resin Identification Codes (RIC). PET and HDPE have high circular economy demand.",
        steps: [
            "Empty completely — pour out liquids and wipe away heavy sauces.",
            "Rinse thoroughly with a small splash of water to prevent contamination.",
            "Crush bottles vertically to reduce volume in transport bins.",
            "Leave screw caps attached — modern optical sorters process them together."
        ],
        resinCodes: [
            { code: "#1 PET", name: "Polyethylene Terephthalate", items: "Water bottles, soda bottles", recyclable: "Yes" },
            { code: "#2 HDPE", name: "High-Density Polyethylene", items: "Milk jugs, shampoo bottles", recyclable: "Yes" },
            { code: "#3 PVC", name: "Polyvinyl Chloride", items: "Pipes, blister packaging", recyclable: "No (Toxic)" },
            { code: "#4 LDPE", name: "Low-Density Polyethylene", items: "Grocery bags, cling wraps", recyclable: "Special Drop-off" },
            { code: "#5 PP", name: "Polypropylene", items: "Yogurt tubs, medicine bottles", recyclable: "Yes" },
            { code: "#6 PS", name: "Polystyrene (Styrofoam)", items: "Takeout containers, cups", recyclable: "Rarely" },
            { code: "#7 OTHER", name: "Multi-layer / Polycarbonate", items: "Mixed polymers, electronics", recyclable: "Facility Dependent" }
        ],
        impact: "Recycling 1 ton of plastic saves approximately 5,774 kWh of energy, 16.3 barrels of oil, and 22 cubic meters of landfill space."
    },
    recyclable: {
        title: "Paper & Cardboard Recycling",
        badge: "Blue Bin · Clean Fibers",
        status: "Recyclable (Up to 5–7 Times)",
        decomposition: "2 – 6 weeks",
        description: "Paper and cardboard consist of natural cellulose fibers. Clean, dry paper is pulped and reformed into recycled cartons, newsprint, and paperboard.",
        steps: [
            "Keep paper strictly clean and dry — wet fibers degrade and clog sorting screens.",
            "Flatten all corrugated cardboard boxes to maximize collection capacity.",
            "Remove packing tape, plastic bubble wraps, and Styrofoam inserts.",
            "Compost soiled paper (pizza boxes with heavy grease) rather than recycling."
        ],
        impact: "Recycling 1 ton of cardboard saves 17 mature trees, 26,000 liters of water, and 4,000 kWh of electricity."
    },
    metal: {
        title: "Metal & Aluminum Cans",
        badge: "Blue Bin · Infinitely Circular",
        status: "100% Recyclable Forever",
        decomposition: "50 – 200 years",
        description: "Aluminum and steel cans are infinitely recyclable without any degradation in structural integrity. A recycled aluminum can returns to shelves in 60 days.",
        steps: [
            "Rinse out food cans and beverage cans to eliminate odors and insects.",
            "Do not crush cans completely if your local municipality uses optical sorting.",
            "Ball clean aluminum foil into a fist-sized sphere before placing in the bin.",
            "Never place pressurized aerosol cans unless completely discharged."
        ],
        impact: "Recycling aluminum requires 95% less energy and produces 95% fewer greenhouse gas emissions than extracting bauxite ore."
    },
    glass: {
        title: "Glass Bottles & Jars",
        badge: "Blue Bin or Glass Bank",
        status: "Infinitely Recyclable",
        decomposition: "1,000,000+ years",
        description: "Glass made from silica sand can be melted and remanufactured repeatedly with zero loss in quality, purity, or clarity.",
        steps: [
            "Rinse glass jars and wine/beer bottles to remove all sticky residue.",
            "Separate metal caps and lids (place metal caps in the metal bin).",
            "Do not mix ceramics, Pyrex, or window glass into bottle banks as they have higher melting points."
        ],
        impact: "Every ton of glass recycled saves 1.2 tons of raw materials and reduces carbon dioxide emissions by 315 kg."
    },
    wet_waste: {
        title: "Wet & Organic Biodegradable Waste",
        badge: "Green Bin · Compostable",
        status: "Compostable & Anaerobic Digestion",
        decomposition: "1 – 6 months",
        description: "Food scraps, fruit peels, coffee grounds, and garden clippings decompose naturally into nutrient-rich organic compost or renewable biogas.",
        steps: [
            "Drain excess liquid before disposal to prevent bin odor and leachate.",
            "Never mix plastic wraps, sticker labels, or rubber bands with food scraps.",
            "Wrap food scraps in newspaper or certified compostable bags.",
            "Home Bokashi or vermicomposting can convert 100% of kitchen waste into organic fertilizer."
        ],
        impact: "Composting organic waste prevents methane generation in anaerobic landfills, a greenhouse gas 28x more potent than CO2."
    },
    hazardous: {
        title: "Hazardous & Electronic Waste (E-Waste)",
        badge: "Special Hazardous Depot",
        status: "Toxic Heavy Metals — Dedicated Handling",
        decomposition: "Non-biodegradable (Toxic)",
        description: "Batteries, lithium cells, circuit boards, fluorescent bulbs, and chemical containers contain lead, mercury, cadmium, and lithium.",
        steps: [
            "NEVER throw batteries or electronics into municipal dry or wet bins.",
            "Tape the terminals of lithium batteries with clear tape to prevent short-circuit fires.",
            "Drop off at certified municipal e-waste collection drives or retail battery kiosks.",
            "Keep expired paints, pesticides, and medicines in original labeled containers."
        ],
        impact: "Proper e-waste recycling recovers precious rare earth minerals (gold, silver, copper) and shields groundwater aquifers from toxic heavy metal poisoning."
    }
};

// ─── Initialization ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    checkServerStatus();
    setInterval(checkServerStatus, 15000);

    // Initialize UI and Motion Systems
    initScrollReveal();
    initTiltEffect();
    initHeroScannerSpotlight();
    initFloatingImagesParallax();
    initCountUpNumbers();
    initNavbarScroll();
    initVoiceRecognition();
    initLightboxEvents();
    initMagneticButtons();

    // Check Llama status on startup
    checkLlamaStatus();

    // Make dropzone clickable
    const dz = document.getElementById('dropZone');
    if (dz) {
        dz.addEventListener('click', () => {
            const fi = document.getElementById('fileInput');
            if (fi) fi.click();
        });
    }
});

// ─── Interactive Floating Sample Image Loader ──────────────────────────────
async function loadSampleImage(imagePath, sampleName) {
    try {
        setMode('gallery');
        scrollToSection('scanner-section');
        setScanningActive(true, `Analyzing sample: ${sampleName}...`);
        startPhasedLoading(`Loading sample image (${sampleName})...`);

        const res = await fetch(imagePath);
        const blob = await res.blob();
        const reader = new FileReader();
        reader.onload = () => {
            currentImageB64 = reader.result;
            const preview = document.getElementById('previewImage');
            if (preview) preview.src = currentImageB64;
            const prevArea = document.getElementById('previewArea');
            if (prevArea) prevArea.classList.remove('hidden');
            analyzeImage();
        };
        reader.readAsDataURL(blob);
    } catch (err) {
        console.error('Error loading sample image:', err);
        setScanningActive(false);
        hideLoading();
    }
}
window.loadSampleImage = loadSampleImage;

// ─── Floating 3D Parallax Mouse Tracking in Hero ───────────────────────────
function initFloatingImagesParallax() {
    const hero = document.getElementById('hero');
    const floatItems = document.querySelectorAll('.floating-interactive-item');
    if (!hero || floatItems.length === 0) return;

    hero.addEventListener('mousemove', (e) => {
        const rect = hero.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width - 0.5;
        const y = (e.clientY - rect.top) / rect.height - 0.5;

        floatItems.forEach((item, index) => {
            const depth = (index + 1) * 16;
            const moveX = x * depth;
            const moveY = y * depth;
            item.style.transform = `translate(${moveX}px, ${moveY}px)`;
        });
    });

    hero.addEventListener('mouseleave', () => {
        floatItems.forEach((item) => {
            item.style.transform = '';
        });
    });
}

// ─── 2D Scanning HUD & Laser Line Controller ─────────────────────────────────
function setScanningActive(isActive, customStatus = 'AI Spectral Analysis In Progress...') {
    const webcamContainer = document.getElementById('webcamContainer');
    const previewContainer = document.getElementById('previewImgContainer');
    const webcamStatus = document.getElementById('webcamScanStatus');
    const previewStatus = document.getElementById('previewScanStatus');

    if (isActive) {
        if (webcamContainer) webcamContainer.classList.add('scanning-active');
        if (previewContainer) previewContainer.classList.add('scanning-active');
        if (webcamStatus) webcamStatus.textContent = customStatus;
        if (previewStatus) previewStatus.textContent = customStatus;
    } else {
        if (webcamContainer) webcamContainer.classList.remove('scanning-active');
        if (previewContainer) previewContainer.classList.remove('scanning-active');
    }
}

// ─── Interactive Magnetic Hover & Radial Spotlight ───────────────────────────
function initMagneticButtons() {
    const buttons = document.querySelectorAll('.magnetic-btn, .btn-primary, .btn-secondary');
    buttons.forEach(btn => {
        btn.addEventListener('mousemove', (e) => {
            const rect = btn.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Set coordinates for CSS radial highlight
            btn.style.setProperty('--mouse-x', `${x}px`);
            btn.style.setProperty('--mouse-y', `${y}px`);

            // Subtle magnetic displacement
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const deltaX = (x - centerX) * 0.15;
            const deltaY = (y - centerY) * 0.15;
            btn.style.transform = `translate(${deltaX}px, ${deltaY}px)`;
        });

        btn.addEventListener('mouseleave', () => {
            btn.style.transform = '';
            btn.style.removeProperty('--mouse-x');
            btn.style.removeProperty('--mouse-y');
        });
    });
}

// ─── Server Status Check ─────────────────────────────────────────────────────
async function checkServerStatus() {
    try {
        const res = await fetch('/api/webcam/status', { signal: AbortSignal.timeout(5000) });
        if (res.ok) {
            setStatus('online', 'Arm Native');
        } else {
            setStatus('offline', 'Offline Safe');
        }
    } catch {
        setStatus('offline', 'Offline Safe');
    }
}

function setStatus(state, text) {
    const dot  = document.getElementById('statusDot');
    const span = document.getElementById('statusText');
    if (dot) dot.className = 'status-dot ' + (state === 'online' ? 'online' : '');
    if (span) span.textContent = text;
}

async function checkLlamaStatus() {
    try {
        const res = await fetch('/api/llama/status');
        const data = await res.json();
        const badge = document.getElementById('engineBadgeInfo');
        if (badge && data.success) {
            const backend = data.active_backend || 'Llama 1B';
            badge.textContent = `Llama 1B (${backend})`;
        }
    } catch {
        // Silent catch
    }
}

// ─── Navigation & Tab Switching ──────────────────────────────────────────────
function switchTab(tabName) {
    // Nav tab buttons
    document.querySelectorAll('.nav-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    // Tab panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === 'panel-' + tabName);
    });

    // Lazy-load data when switching
    if (tabName === 'dashboard') refreshDashboard();
    if (tabName === 'records')   loadRecords();
}

function scrollToSection(sectionId) {
    const el = document.getElementById(sectionId);
    if (el) {
        const yOffset = -80; // Offset for sticky navbar
        const y = el.getBoundingClientRect().top + window.pageYOffset + yOffset;
        window.scrollTo({ top: y, behavior: 'smooth' });
    }
}

function toggleMobileMenu() {
    const drawer = document.getElementById('mobileDrawer');
    const btn = document.getElementById('hamburgerBtn');
    if (drawer && btn) {
        drawer.classList.toggle('open');
        btn.classList.toggle('open');
    }
}

// ─── Mode Switch (Gallery / Webcam) ──────────────────────────────────────────
function setMode(mode) {
    currentMode = mode;
    const btnG = document.getElementById('btnGallery');
    const btnW = document.getElementById('btnWebcam');
    const galM = document.getElementById('galleryMode');
    const webM = document.getElementById('webcamMode');
    const prevA = document.getElementById('previewArea');

    if (btnG) btnG.classList.toggle('active', mode === 'gallery');
    if (btnW) btnW.classList.toggle('active', mode === 'webcam');
    if (galM) galM.classList.toggle('hidden', mode !== 'gallery');
    if (webM) webM.classList.toggle('hidden', mode !== 'webcam');
    if (prevA) prevA.classList.add('hidden');
}

// ─── Gallery: File Handling ──────────────────────────────────────────────────
function handleDrop(e) {
    e.preventDefault();
    const dropTarget = e.target.closest('.drop-zone');
    if (dropTarget) dropTarget.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) loadImageFile(file);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) loadImageFile(file);
}

function loadImageFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImageB64 = e.target.result;
        const preview = document.getElementById('previewImage');
        if (preview) preview.src = currentImageB64;
        const prevArea = document.getElementById('previewArea');
        if (prevArea) prevArea.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

// ─── Gallery: Analyze Image ──────────────────────────────────────────────────
async function analyzeImage() {
    if (!currentImageB64 || isAnalyzing) return;
    isAnalyzing = true;

    setScanningActive(true, 'Analyzing Material & Spectral Signatures...');
    startPhasedLoading();
    const btn = document.getElementById('btnAnalyze');
    if (btn) btn.disabled = true;

    try {
        const res = await fetch('/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: currentImageB64 }),
        });

        const data = await res.json();
        if (data.success) {
            renderResults(data);
        } else {
            showError('Detection failed: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        showError('Server connection error: Python backend is not running. Make sure `python web_app.py` is running in your terminal.');
    } finally {
        isAnalyzing = false;
        setScanningActive(false);
        if (btn) btn.disabled = false;
        hideLoading();
    }
}

// ─── Webcam Stream & Capture ─────────────────────────────────────────────────
function startWebcam() {
    const feed    = document.getElementById('webcamFeed');
    const overlay = document.getElementById('webcamOverlay');

    if (feed) feed.src = '/api/webcam/stream?' + Date.now();
    if (overlay) overlay.classList.add('hidden');
    webcamActive = true;

    const bStart = document.getElementById('btnStartCam');
    const bCap   = document.getElementById('btnCapture');
    const bStop  = document.getElementById('btnStopCam');
    if (bStart) bStart.disabled = true;
    if (bCap)   bCap.disabled   = false;
    if (bStop)  bStop.disabled  = false;
}

function stopWebcam() {
    const feed    = document.getElementById('webcamFeed');
    const overlay = document.getElementById('webcamOverlay');

    if (feed) feed.src = '';
    if (overlay) overlay.classList.remove('hidden');
    webcamActive = false;

    const bStart = document.getElementById('btnStartCam');
    const bCap   = document.getElementById('btnCapture');
    const bStop  = document.getElementById('btnStopCam');
    if (bStart) bStart.disabled = false;
    if (bCap)   bCap.disabled   = true;
    if (bStop)  bStop.disabled  = true;
}

async function captureFrame() {
    if (isAnalyzing) return;
    isAnalyzing = true;
    setScanningActive(true, 'Capturing Frame & Running YOLOv8 Detection...');
    startPhasedLoading('Capturing frame & running YOLOv8 detection...');
    const bCap = document.getElementById('btnCapture');
    if (bCap) bCap.disabled = true;

    try {
        const res = await fetch('/api/webcam/capture', { method: 'POST' });
        const data = await res.json();

        if (data.success) {
            renderResults(data);
        } else {
            showError('Capture failed: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        showError('Server connection error: Python backend is not running. Make sure `python web_app.py` is running in your terminal.');
    } finally {
        isAnalyzing = false;
        setScanningActive(false);
        if (webcamActive && bCap) bCap.disabled = false;
        hideLoading();
    }
}

// ─── Render Detection Results & AI Cards ─────────────────────────────────────
function renderResults(data) {
    currentDetectionData = data;
    const area = document.getElementById('resultsArea');
    const badge = document.getElementById('inferenceBadge');

    if (badge && data.inference_ms) {
        badge.textContent = `${data.inference_ms} ms (YOLO)`;
    }

    // Store last detection for context-aware queries
    if (data.detections && data.detections.length > 0) {
        window.lastDetection = {
            waste_type: data.detections[0].class_name,
            confidence: data.detections[0].confidence,
            bbox: data.detections[0].bbox,
            object_guess: data.object_guess,
            plastic_type: data.plastic_type,
            recycling_steps: data.recycling_steps,
            household_reuse: data.household_reuse,
            decomposition_time: data.decomposition_time,
            num_detections: data.detections.length,
            all_detections: data.detections.map(d => ({
                object: d.detected_object || d.class_name,
                detected_object: d.detected_object || d.class_name,
                class_name: d.class_name,
                confidence: d.confidence,
                bbox: d.bbox
            })),
            detections_summary: `${data.detections.length} object(s) captured in picture: ` +
                data.detections.map(d => `${d.detected_object || d.class_name} (${(d.confidence*100).toFixed(0)}%)`).join(', ')
        };
    }

    // Failure / Zero detections state
    if (!data.detections || data.num_detections === 0) {
        window.lastDetection = null;
        if (area) {
            area.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                    </div>
                    <p><strong>Couldn't identify this item</strong></p>
                    <p class="empty-sub">Try a clearer photo with the object centered and well-lit.</p>
                    <button class="btn btn-primary btn-small" style="margin-top:1rem;" onclick="scrollToSection('scanner-section')">
                        <svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
                        Try Again
                    </button>
                </div>`;
        }
        const statusPill = document.getElementById('detectChatStatus');
        if (statusPill) {
            statusPill.textContent = 'Context: No object detected';
            statusPill.className = 'chat-status-pill';
        }
        const guessCard = document.getElementById('objectGuessCard');
        if (guessCard) guessCard.classList.add('hidden');
        appendDetectMessage('bot',
            '⚠️ **No waste item identified with confidence in this photo.**\n\n' +
            '• Please ensure the object is centered against a clean background.\n' +
            '• Or ask me below: *"How do I dispose of this plastic item?"* or *"Can plastic folders / containers be recycled?"*'
        );
        return;
    }

    let html = '';

    // Render detections with elevated UI styling & confidence tiers
    data.detections.forEach((det, idx) => {
        const name    = det.class_name || 'unknown';
        const confNum = Math.round(det.confidence * 100);
        const conf    = (det.confidence * 100).toFixed(1);
        const icon    = WASTE_ICONS[name] || WASTE_ICONS.default;
        const color   = WASTE_COLORS[name] || WASTE_COLORS.default;

        const objName = det.detected_object || name.replace(/_/g, ' ');
        const catTitle = name.replace(/_/g, ' ');

        let tierClass = 'conf-tier-low';
        let tierLabel = 'Low Confidence';
        if (confNum >= 75) {
            tierClass = 'conf-tier-high';
            tierLabel = 'Verified High Match';
        } else if (confNum >= 50) {
            tierClass = 'conf-tier-med';
            tierLabel = 'Probable Match';
        }

        html += `
            <div class="detection-item" style="--card-accent:${color}; animation-delay:${idx * 0.08}s;">
                <div class="detection-icon" style="color:${color}; background:${color}18; border-color:${color}44;">${icon}</div>
                <div class="detection-info">
                    <div class="detection-header-line">
                        <span class="detection-name">${escapeHtml(objName)}</span>
                        <span class="detection-category-pill" style="color:${color}; border-color:${color}44; background:${color}12;">${escapeHtml(catTitle)}</span>
                    </div>
                    <div class="detection-meta-row">
                        <span class="detection-score-value">${conf}%</span>
                        <span class="detection-conf-sub">YOLOv8 Real-time Inference</span>
                        <span class="detection-tier-chip ${tierClass}">${tierLabel}</span>
                    </div>
                    <div class="detection-bar">
                        <div class="detection-bar-fill" style="width:${conf}%; background:linear-gradient(90deg, ${color}, #10b981);"></div>
                    </div>
                </div>
            </div>`;
    });

    // Annotated bounding box image with Full Screen Mode trigger
    if (data.annotated_image) {
        html += `
            <div class="annotated-image-wrapper">
                <div class="annotated-image-header">
                    <span class="annotated-title">
                        <svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                        YOLO Bounding Box Visualization
                    </span>
                    <button type="button" class="btn-fullscreen-trigger" onclick="openImageLightbox('${escapeHtml(data.annotated_image)}')">
                        <svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path></svg>
                        Full Screen Mode
                    </button>
                </div>
                <div class="annotated-image-container clickable-preview" onclick="openImageLightbox('${escapeHtml(data.annotated_image)}')">
                    <img src="${data.annotated_image}" alt="YOLO Annotated detection result">
                    <div class="image-expand-hint">
                        <svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path></svg>
                        Click for Detailed Full Screen Inspection
                    </div>
                </div>
            </div>`;
    }

    if (area) area.innerHTML = html;

    // AI Object Estimation Card & Llama Reasoning
    const guessCard = document.getElementById('objectGuessCard');
    if (guessCard && data.object_guess) {
        const objNameEl = document.getElementById('guessObjectName');
        const matNameEl = document.getElementById('guessMaterialName');
        const badgeEl   = document.getElementById('guessResinBadge');
        const recycleList = document.getElementById('guessRecycleList');
        const reuseList   = document.getElementById('guessReuseList');
        const llamaBadge  = document.getElementById('llamaPoweredBadge');

        if (objNameEl) objNameEl.textContent = data.object_guess;
        if (matNameEl) matNameEl.textContent = data.plastic_type || 'Material Analyzed';
        if (badgeEl)   badgeEl.textContent   = (data.plastic_type ? data.plastic_type.split('(')[0] : 'Identified').trim();

        if (recycleList) {
            recycleList.innerHTML = (data.recycling_steps || []).map((step, idx) => `
                <li class="interactive-step-card">
                    <span class="step-badge-num">0${idx+1}</span>
                    <div class="step-card-content">
                        <span class="step-card-title">Protocol Phase ${idx+1}</span>
                        <span class="step-card-desc">${escapeHtml(step)}</span>
                    </div>
                </li>
            `).join('');
        }

        if (reuseList) {
            reuseList.innerHTML = (data.household_reuse || []).map((item, idx) => `
                <li class="interactive-reuse-card">
                    <div class="reuse-card-icon">
                        <svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"></path></svg>
                    </div>
                    <div class="reuse-card-body">
                        <span class="reuse-card-label">Circular Reuse #${idx+1}</span>
                        <span class="reuse-card-text">${escapeHtml(item)}</span>
                    </div>
                </li>
            `).join('');
        }

        // Show Llama badge
        if (llamaBadge) llamaBadge.style.display = 'flex';

        guessCard.classList.remove('hidden');

        // Update chatbot context pill in Detect Tab
        const statusPill = document.getElementById('detectChatStatus');
        if (statusPill) {
            statusPill.textContent = 'Context: ' + data.object_guess.split('(')[0].trim();
            statusPill.className = 'chat-status-pill active';
        }

        // Post notification into embedded chatbot
        if (data.detections.length > 1) {
            const detListStr = data.detections.map((d, i) =>
                `  ${i+1}. **${d.detected_object || d.class_name}** (${(d.confidence * 100).toFixed(0)}%)`
            ).join('\n');
            appendDetectMessage('bot',
                `📸 **Picture Analyzed — ${data.detections.length} Objects Detected:**\n${detListStr}\n\n` +
                `• **Primary Item:** ${data.object_guess}\n` +
                `• **Resin / Material:** ${data.plastic_type}\n\n` +
                `Ask me below: *"What are the objects in the picture?"*, *"How can I reuse these at home?"*, or *"Can these be recycled?"*`
            );
        } else {
            const confPct = (data.detections[0].confidence * 100).toFixed(0);
            appendDetectMessage('bot',
                `**Identified: ${data.object_guess}** (${confPct}% confidence)\n` +
                `• **Resin / Material:** ${data.plastic_type}\n\n` +
                `Ask me below how to **reuse this object at home** or check the recycling recommendations above!`
            );
        }
    } else if (guessCard) {
        guessCard.classList.add('hidden');
    }
}

// ─── Dashboard Tab ───────────────────────────────────────────────────────────
async function refreshDashboard() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        if (!data.success) return;

        // Stat cards
        animateNumber('totalRecords', data.total_records || 0);

        // Today count from daily_stats
        let todayCount = 0;
        if (data.daily_stats && data.daily_stats.length > 0) {
            const today = new Date().toISOString().split('T')[0];
            const todayStat = data.daily_stats.find(d => d.date === today);
            todayCount = todayStat ? todayStat.count : 0;
        }
        animateNumber('todayRecords', todayCount);

        // Alerts count
        const alertTotal = data.recent_alerts ? data.recent_alerts.length : 0;
        animateNumber('totalAlerts', alertTotal);

        // Hazardous count from distribution
        let hazCount = 0;
        if (data.waste_distribution) {
            const haz = data.waste_distribution.find(d => d.waste_type === 'hazardous');
            hazCount = haz ? haz.count : 0;
        }
        animateNumber('hazardousCount', hazCount);

        // Distribution & alerts
        renderDistribution(data.waste_distribution || []);
        renderAlerts(data.recent_alerts || []);
    } catch (err) {
        console.error('Dashboard refresh error:', err);
    }
}

function renderDistribution(dist) {
    const container = document.getElementById('distributionChart');
    if (!container) return;

    if (!dist.length) {
        container.innerHTML = '<div class="empty-state small"><p>No data yet — run detections first!</p></div>';
        return;
    }

    const maxCount = Math.max(...dist.map(d => d.count), 1);
    let html = '';

    dist.forEach(d => {
        const pct   = Math.round((d.count / maxCount) * 100);
        const color = WASTE_COLORS[d.waste_type] || WASTE_COLORS.default;
        const icon  = WASTE_ICONS[d.waste_type] || WASTE_ICONS.default;

        html += `
            <div class="dist-bar-group">
                <div class="dist-label">${icon} ${(d.waste_type || '').replace(/_/g, ' ')}</div>
                <div class="dist-track">
                    <div class="dist-fill" style="width:${pct}%; background:${color}"></div>
                </div>
                <div class="dist-count">${d.count}</div>
            </div>`;
    });

    container.innerHTML = html;
}

function renderAlerts(alerts) {
    const container = document.getElementById('alertsList');
    if (!container) return;

    if (!alerts.length) {
        container.innerHTML = '<div class="empty-state small"><p>All systems nominal — no recent alerts</p></div>';
        return;
    }

    let html = '';
    alerts.slice(0, 10).forEach(a => {
        const time = a.timestamp ? new Date(a.timestamp).toLocaleTimeString() : '';
        html += `
            <div class="alert-item">
                <div class="alert-icon">
                    <svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                </div>
                <div class="alert-text">${escapeHtml(a.message || a.alert_type || 'Alert')}</div>
                <div class="alert-time">${time}</div>
            </div>`;
    });

    container.innerHTML = html;
}

// ─── Records Tab ─────────────────────────────────────────────────────────────
async function loadRecords() {
    try {
        const filterEl = document.getElementById('typeFilter');
        const typeFilter = filterEl ? filterEl.value : '';
        let url = '/api/records?limit=50';
        if (typeFilter) url += '&type=' + encodeURIComponent(typeFilter);

        const res  = await fetch(url);
        const data = await res.json();
        const tbody = document.getElementById('recordsBody');
        if (!tbody) return;

        if (!data.success || !data.records || data.records.length === 0) {
            tbody.innerHTML = '<tr><td colspan="9" class="empty-cell">No records found</td></tr>';
            return;
        }

        let html = '';
        data.records.forEach(r => {
            const icon  = WASTE_ICONS[r.waste_type] || WASTE_ICONS.default;
            const color = WASTE_COLORS[r.waste_type] || WASTE_COLORS.default;

            const conf     = r.ai_confidence != null ? (r.ai_confidence * 100).toFixed(1) + '%' : '—';
            const humidity = r.humidity != null && r.humidity >= 0 ? r.humidity.toFixed(1) + '%' : '—';
            const gas      = r.gas_level != null && r.gas_level > 0 ? r.gas_level : '—';
            const fill     = r.bin_fill_level != null && r.bin_fill_level >= 0 ? r.bin_fill_level.toFixed(0) + '%' : '—';
            const isHaz    = r.waste_type === 'hazardous' || (r.gas_level && r.gas_level > 300);
            const source   = r.vehicle_id || '—';
            const timeStr  = r.timestamp ? new Date(r.timestamp).toLocaleString() : '—';

            html += `
                <tr>
                    <td>${r.id || '—'}</td>
                    <td>${timeStr}</td>
                    <td><span class="type-badge" style="background:${color}22; color:${color}">${icon} ${(r.waste_type || '').replace(/_/g, ' ')}</span></td>
                    <td>${conf}</td>
                    <td>${humidity}</td>
                    <td>${gas}</td>
                    <td>${fill}</td>
                    <td>${isHaz ? '<span class="badge badge-red">⚠️ Yes</span>' : '<span class="badge badge-green">✅ No</span>'}</td>
                    <td>${source}</td>
                </tr>`;
        });

        tbody.innerHTML = html;
    } catch (err) {
        console.error('Records loading error:', err);
        const tbody = document.getElementById('recordsBody');
        if (tbody) tbody.innerHTML = '<tr><td colspan="9" class="empty-cell">Error loading records from database</td></tr>';
    }
}

// ─── Chatbot Handlers ────────────────────────────────────────────────────────
function sendQuickMessage(text) {
    const input = document.getElementById('chatInput');
    if (input) input.value = text;
    sendMessage();
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    if (!input) return;
    const message = input.value.trim();
    if (!message) return;

    input.value = '';
    appendMessage('user', message);
    const typingId = showTyping();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                detection_context: window.lastDetection || null
            }),
        });

        const data = await res.json();
        removeTyping(typingId);

        if (data.success) {
            appendMessage('bot', data.response);
            appendDetectMessage('bot', data.response);
        } else {
            const err = data.error || 'AI recommendation unavailable – please try again.';
            appendMessage('bot', err);
            appendDetectMessage('bot', err);
        }
    } catch (err) {
        removeTyping(typingId);
        const msg = 'AI recommendation unavailable – the waste was detected, but the recommendation couldn\'t be generated right now. Make sure `python web_app.py` is active.';
        appendMessage('bot', msg);
        appendDetectMessage('bot', msg);
    }
}

function sendDetectQuickMessage(text) {
    const input = document.getElementById('detectChatInput');
    if (input) input.value = text;
    sendDetectMessage();
}

async function sendDetectMessage() {
    const input = document.getElementById('detectChatInput');
    if (!input) return;
    const message = input.value.trim();
    if (!message) return;

    input.value = '';
    appendDetectMessage('user', message);
    appendMessage('user', message);

    const typingId = showDetectTyping();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                detection_context: window.lastDetection || null
            }),
        });

        const data = await res.json();
        removeDetectTyping(typingId);

        if (data.success) {
            appendDetectMessage('bot', data.response);
            appendMessage('bot', data.response);
        } else {
            const err = data.error || 'Something went wrong';
            appendDetectMessage('bot', err);
            appendMessage('bot', err);
        }
    } catch (err) {
        removeDetectTyping(typingId);
        const msg = 'AI recommendation unavailable – the waste was detected, but the recommendation couldn\'t be generated right now.';
        appendDetectMessage('bot', msg);
        appendMessage('bot', msg);
    }
}

const SVG_USER_AVATAR = `<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`;
const SVG_BOT_AVATAR = `<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8.01" y2="16"></line><line x1="16" y1="16" x2="16.01" y2="16"></line></svg>`;

function appendMessage(type, text) {
    const container = document.getElementById('chatMessages');
    if (!container) return;
    const avatar = type === 'user' ? SVG_USER_AVATAR : SVG_BOT_AVATAR;

    let formatted = escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');

    const div = document.createElement('div');
    div.className = 'chat-message ' + type;

    // Add TTS speak button for bot messages
    let speakBtn = '';
    if (type === 'bot') {
        speakBtn = '<div class="message-actions"><button class="tts-btn" onclick="speakBotMessage(this)" title="Listen to response" aria-label="Listen to response"><svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg></button></div>';
    }

    div.innerHTML = '<div class="message-avatar">' + avatar + '</div><div class="message-bubble"><p>' + formatted + '</p>' + speakBtn + '</div>';

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;

    // Auto-speak bot response if last input was voice
    if (type === 'bot' && lastInputWasVoice) {
        lastInputWasVoice = false;
        speakText(text);
    }
}

function appendDetectMessage(type, text) {
    const container = document.getElementById('detectChatMessages');
    if (!container) return;
    const avatar = type === 'user' ? SVG_USER_AVATAR : SVG_BOT_AVATAR;

    let formatted = escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');

    const div = document.createElement('div');
    div.className = 'chat-message ' + type;

    // Add TTS speak button for bot messages
    let speakBtn = '';
    if (type === 'bot') {
        speakBtn = '<div class="message-actions"><button class="tts-btn" onclick="speakBotMessage(this)" title="Listen to response" aria-label="Listen to response"><svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg></button></div>';
    }

    div.innerHTML = '<div class="message-avatar">' + avatar + '</div><div class="message-bubble"><p>' + formatted + '</p>' + speakBtn + '</div>';

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function showTyping() {
    const container = document.getElementById('chatMessages');
    if (!container) return null;
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.className = 'chat-message bot';
    div.id = id;
    div.innerHTML = `
        <div class="message-avatar">${SVG_BOT_AVATAR}</div>
        <div class="message-bubble">
            <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeTyping(id) {
    if (!id) return;
    const el = document.getElementById(id);
    if (el) el.remove();
}

function showDetectTyping() {
    const container = document.getElementById('detectChatMessages');
    if (!container) return null;
    const id = 'detect-typing-' + Date.now();
    const div = document.createElement('div');
    div.className = 'chat-message bot';
    div.id = id;
    div.innerHTML = `
        <div class="message-avatar">${SVG_BOT_AVATAR}</div>
        <div class="message-bubble">
            <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeDetectTyping(id) {
    if (!id) return;
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ─── Voice Recognition & Speech-to-Text Pipeline ───────────────────────────
let voiceAccumulatedTranscript = '';
let voiceManualStopInitiated   = false;
let voiceSilenceTimer          = null;

function showVoiceHelp(title, htmlContent) {
    let banner = document.getElementById('voiceHelpBanner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'voiceHelpBanner';
        banner.className = 'voice-help-banner';
        const chatBar = document.querySelector('.chat-input-bar');
        if (chatBar && chatBar.parentNode) {
            chatBar.parentNode.insertBefore(banner, chatBar);
        }
    }
    banner.innerHTML = `
        <div class="voice-help-header">
            <span class="voice-help-title">${title}</span>
            <button class="voice-help-close" onclick="closeVoiceHelp()" aria-label="Close">&times;</button>
        </div>
        <div class="voice-help-content">${htmlContent}</div>
    `;
    banner.classList.remove('hidden');
}

function closeVoiceHelp() {
    const banner = document.getElementById('voiceHelpBanner');
    if (banner) banner.classList.add('hidden');
}
window.closeVoiceHelp = closeVoiceHelp;

function updateVoiceStatus(text, isRecording = true) {
    const statusBar = document.getElementById('voiceStatusBar');
    if (statusBar) {
        if (isRecording) statusBar.classList.remove('hidden');
        const statusText = statusBar.querySelector('.voice-status-text');
        if (statusText) statusText.textContent = text;
    }
}

function initVoiceRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn('[EcoSort] Web Speech API not supported in this browser.');
        return false;
    }

    try {
        if (speechRecognizer) {
            try { speechRecognizer.abort(); } catch (e) {}
        }
        speechRecognizer = new SpeechRecognition();
        speechRecognizer.continuous = true;
        speechRecognizer.interimResults = true;
        speechRecognizer.lang = 'en-US';
        speechRecognizer.maxAlternatives = 1;

        speechRecognizer.onstart = () => {
            console.log('[EcoSort] 🎤 Speech recognition started');
            isRecordingVoice = true;
            voiceManualStopInitiated = false;
            voiceAccumulatedTranscript = '';
            
            const micBtn = document.getElementById('voiceMicBtn');
            if (micBtn) {
                micBtn.classList.add('recording');
                micBtn.setAttribute('title', 'Click to finish speaking and send');
            }
            updateVoiceStatus('🎤 Listening... Speak now', true);
            closeVoiceHelp();
        };

        speechRecognizer.onresult = (event) => {
            let finalPart = '';
            let interimPart = '';

            for (let i = 0; i < event.results.length; i++) {
                const res = event.results[i];
                if (res.isFinal) {
                    finalPart += res[0].transcript + ' ';
                } else {
                    interimPart += res[0].transcript;
                }
            }

            const combined = (finalPart + interimPart).trim();
            if (combined) {
                voiceAccumulatedTranscript = combined;
                const input = document.getElementById('chatInput');
                if (input) {
                    input.value = combined;
                }
                const preview = combined.length > 40 ? combined.substring(0, 38) + '...' : combined;
                updateVoiceStatus(`🗣️ "${preview}"`, true);

                // Auto-finish after 2.2 seconds of silence once speech has been recognized
                if (voiceSilenceTimer) clearTimeout(voiceSilenceTimer);
                voiceSilenceTimer = setTimeout(() => {
                    if (isRecordingVoice && voiceAccumulatedTranscript.length > 0) {
                        console.log('[EcoSort] Silence detected, finishing recording and sending...');
                        finishVoiceRecordingAndSend();
                    }
                }, 2200);
            }
        };

        speechRecognizer.onerror = (event) => {
            console.error('[EcoSort] Speech recognition error:', event.error);
            if (voiceSilenceTimer) clearTimeout(voiceSilenceTimer);
            
            if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                const isBrave = (navigator.brave && typeof navigator.brave.isBrave === 'function') ||
                                (navigator.userAgent.includes('Brave'));
                let helpHtml = `
                    <p><strong>Microphone permission is blocked in your browser.</strong></p>
                    <ol>
                        <li>Look at your browser's address bar at the top (next to <code>127.0.0.1:5000</code>).</li>
                        <li>Click the <strong>Microphone icon (🎤) with an exclamation mark</strong> or the tune/lock icon.</li>
                        <li>Change <strong>Microphone</strong> from <em>Block</em> to <strong>Allow</strong>.</li>
                        <li>Click <em>"Try Microphone Again"</em> below.</li>
                    </ol>
                `;
                if (isBrave) {
                    helpHtml += `
                        <div class="brave-note">
                            <strong>🦁 Brave Browser Tip:</strong> Brave blocks Google Speech services by default.
                            If speech-to-text still doesn't type after allowing the mic, open <code>brave://settings/privacy</code> and toggle ON <strong>"Use Google services for speech recognition"</strong>, or open EcoSort in <strong>Google Chrome</strong> or <strong>Microsoft Edge</strong>.
                        </div>
                    `;
                }
                helpHtml += `<button class="btn btn-sm btn-accent" style="margin-top:0.6rem;padding:0.4rem 0.9rem;cursor:pointer;" onclick="toggleVoiceRecording()">🔄 Try Microphone Again</button>`;
                showVoiceHelp('⚠️ Microphone Access Blocked', helpHtml);
                updateVoiceStatus('❌ Microphone blocked. Check instructions above.', true);
                setTimeout(() => stopVoiceRecording(), 4000);
            } else if (event.error === 'network') {
                showVoiceHelp('🌐 Speech Recognition Network Error', `
                    <p>The browser could not connect to speech recognition services.</p>
                    <p>If you are using <strong>Brave Browser</strong>, please open <code>brave://settings/privacy</code> and turn ON <strong>"Use Google services for speech recognition"</strong>, or use <strong>Google Chrome</strong> / <strong>Edge</strong>.</p>
                `);
                stopVoiceRecording();
            } else if (event.error === 'no-speech') {
                updateVoiceStatus('🔇 No speech detected. Please speak closer to mic.', true);
                setTimeout(() => stopVoiceRecording(), 2500);
            } else if (event.error !== 'aborted') {
                updateVoiceStatus(`⚠️ Error: ${event.error}`, true);
                setTimeout(() => stopVoiceRecording(), 2500);
            } else {
                stopVoiceRecording();
            }
        };

        speechRecognizer.onend = () => {
            console.log('[EcoSort] Speech recognition onend triggered');
            if (voiceSilenceTimer) clearTimeout(voiceSilenceTimer);
            
            const input = document.getElementById('chatInput');
            const captured = ((input ? input.value : '') || voiceAccumulatedTranscript || '').trim();

            stopVoiceRecording();

            if (!voiceManualStopInitiated && captured && captured.length > 0) {
                console.log('[EcoSort] Auto-sending recognized speech:', captured);
                if (input) input.value = captured;
                lastInputWasVoice = true;
                sendMessage();
            }
        };

        return true;
    } catch (err) {
        console.error('[EcoSort] Error setting up SpeechRecognition:', err);
        return false;
    }
}

async function toggleVoiceRecording() {
    // If already recording, user clicked the mic button to finish and submit
    if (isRecordingVoice) {
        console.log('[EcoSort] Finishing voice recording upon user click...');
        finishVoiceRecordingAndSend();
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        showVoiceHelp('Microphone Speech-to-Text Not Supported', `
            <p>Your browser does not support the Web Speech Recognition API.</p>
            <p>Please open EcoSort in <strong>Google Chrome</strong> or <strong>Microsoft Edge</strong> for voice interaction.</p>
        `);
        return;
    }

    // Explicitly test/prompt microphone permission via getUserMedia
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            updateVoiceStatus('Checking microphone permission...', true);
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
        } catch (permErr) {
            console.warn('[EcoSort] getUserMedia permission denied:', permErr);
            const isBrave = (navigator.brave && typeof navigator.brave.isBrave === 'function') ||
                            (navigator.userAgent.includes('Brave'));
            let helpHtml = `
                <p>The browser blocked microphone access for <code>127.0.0.1:5000</code>.</p>
                <ol>
                    <li>Click the <strong>Microphone icon (🎤) with exclamation/slash</strong> in the address bar at the top.</li>
                    <li>Set <strong>Microphone</strong> to <strong>Allow</strong>.</li>
                    <li>Click <em>"Try Microphone Again"</em> below.</li>
                </ol>
            `;
            if (isBrave) {
                helpHtml += `
                    <div class="brave-note">
                        <strong>🦁 Brave Browser Tip:</strong> In addition to allowing the microphone, open <code>brave://settings/privacy</code> and toggle ON <strong>"Use Google services for speech recognition"</strong>, or use Google Chrome / Edge.
                    </div>
                `;
            }
            helpHtml += `<button class="btn btn-sm btn-accent" style="margin-top:0.6rem;padding:0.4rem 0.9rem;cursor:pointer;" onclick="toggleVoiceRecording()">🔄 Try Microphone Again</button>`;
            showVoiceHelp('⚠️ Microphone Access Blocked', helpHtml);
            stopVoiceRecording();
            return;
        }
    }

    if (!speechRecognizer) {
        initVoiceRecognition();
    }

    if (!speechRecognizer) {
        showVoiceHelp('Error', '<p>Failed to initialize speech recognizer. Please reload the page.</p>');
        return;
    }

    try {
        const input = document.getElementById('chatInput');
        if (input) {
            input.value = '';
            input.placeholder = 'Listening... Speak your question';
        }
        voiceAccumulatedTranscript = '';
        voiceManualStopInitiated = false;
        speechRecognizer.start();
    } catch (e) {
        console.warn('[EcoSort] speechRecognizer.start() error:', e);
        if (e.message && e.message.includes('already started')) {
            try {
                speechRecognizer.abort();
                setTimeout(() => {
                    try { speechRecognizer.start(); } catch (err2) { stopVoiceRecording(); }
                }, 150);
            } catch (err) {
                stopVoiceRecording();
            }
        } else {
            stopVoiceRecording();
        }
    }
}

function finishVoiceRecordingAndSend() {
    voiceManualStopInitiated = true;
    if (voiceSilenceTimer) clearTimeout(voiceSilenceTimer);
    updateVoiceStatus('⏳ Processing speech...', true);

    if (speechRecognizer) {
        try { speechRecognizer.stop(); } catch (e) {}
    }

    const input = document.getElementById('chatInput');
    const textToSend = ((input ? input.value : '') || voiceAccumulatedTranscript || '').trim();

    stopVoiceRecording();

    if (textToSend && textToSend.length > 0) {
        console.log('[EcoSort] Submitting user speech:', textToSend);
        if (input) input.value = textToSend;
        lastInputWasVoice = true;
        sendMessage();
    } else {
        console.log('[EcoSort] No speech text to submit');
    }
}
window.finishVoiceRecordingAndSend = finishVoiceRecordingAndSend;

function cancelVoiceRecording() {
    voiceManualStopInitiated = true;
    if (voiceSilenceTimer) clearTimeout(voiceSilenceTimer);
    voiceAccumulatedTranscript = '';
    const input = document.getElementById('chatInput');
    if (input) input.value = '';
    if (speechRecognizer) {
        try { speechRecognizer.abort(); } catch (e) {}
    }
    stopVoiceRecording();
}
window.cancelVoiceRecording = cancelVoiceRecording;

function stopVoiceRecording() {
    isRecordingVoice = false;
    if (voiceSilenceTimer) clearTimeout(voiceSilenceTimer);
    
    const micBtn = document.getElementById('voiceMicBtn');
    if (micBtn) {
        micBtn.classList.remove('recording');
        micBtn.setAttribute('title', 'Speak to EcoSort - click to talk');
    }
    const statusBar = document.getElementById('voiceStatusBar');
    if (statusBar) statusBar.classList.add('hidden');

    const input = document.getElementById('chatInput');
    if (input && input.placeholder.includes('Listening')) {
        input.placeholder = 'Ask about recycling, resin codes, or household reuse...';
    }
}

function triggerVoicePrompt() {
    const input = document.getElementById('chatInput');
    if (input) input.focus();
    toggleVoiceRecording();
}

// ─── Text-to-Speech (TTS) Engine ─────────────────────────────────────────────
function speakText(text) {
    if (!('speechSynthesis' in window)) {
        console.warn('[EcoSort] Text-to-Speech is not supported in this browser.');
        return;
    }

    stopSpeaking();

    // Clean text for speech: strip markdown, URLs, emoji, bullet points
    let cleanText = text
        .replace(/\*\*(.*?)\*\*/g, '$1')
        .replace(/\*(.*?)\*/g, '$1')
        .replace(/#{1,6}\s/g, '')
        .replace(/[\[\]()]/g, ' ')
        .replace(/https?:\/\/\S+/g, '')
        .replace(/[•●▪-]/g, '')
        .replace(/\n{2,}/g, '. ')
        .replace(/\n/g, '. ')
        .replace(/\s{2,}/g, ' ')
        .trim();

    if (!cleanText) return;

    currentUtterance = new SpeechSynthesisUtterance(cleanText);
    currentUtterance.rate = 0.98;
    currentUtterance.pitch = 1.0;
    currentUtterance.volume = 1.0;
    currentUtterance.lang = 'en-US';

    const voices = speechSynthesis.getVoices();
    const preferredVoice = voices.find(v =>
        v.name.includes('Google') && v.lang.startsWith('en')
    ) || voices.find(v =>
        v.name.includes('Natural') && v.lang.startsWith('en')
    ) || voices.find(v =>
        v.lang.startsWith('en') && v.localService === false
    ) || voices.find(v =>
        v.lang.startsWith('en')
    );
    if (preferredVoice) currentUtterance.voice = preferredVoice;

    currentUtterance.onend = () => {
        currentUtterance = null;
        updateAllTtsButtons(false);
    };
    currentUtterance.onerror = () => {
        currentUtterance = null;
        updateAllTtsButtons(false);
    };

    speechSynthesis.speak(currentUtterance);
}

function stopSpeaking() {
    if ('speechSynthesis' in window) {
        if (speechSynthesis.speaking || speechSynthesis.pending) {
            speechSynthesis.cancel();
        }
    }
    currentUtterance = null;
    updateAllTtsButtons(false);
}

function speakBotMessage(btn) {
    if (btn.classList.contains('speaking')) {
        stopSpeaking();
        return;
    }

    stopSpeaking();

    const bubble = btn.closest('.message-bubble');
    if (!bubble) return;
    const pEl = bubble.querySelector('p');
    if (!pEl) return;
    const text = pEl.textContent || pEl.innerText;

    updateAllTtsButtons(false);
    btn.classList.add('speaking');
    btn.innerHTML = '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>';
    btn.title = 'Stop listening';

    const origOnEnd = () => {
        btn.classList.remove('speaking');
        btn.innerHTML = '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>';
        btn.title = 'Listen to response';
    };

    speakText(text);

    if (currentUtterance) {
        currentUtterance.onend = origOnEnd;
        currentUtterance.onerror = origOnEnd;
    }
}

function updateAllTtsButtons(isSpeaking) {
    document.querySelectorAll('.tts-btn.speaking').forEach(btn => {
        if (!isSpeaking) {
            btn.classList.remove('speaking');
            btn.innerHTML = '<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>';
            btn.title = 'Listen to response';
        }
    });
}

if ('speechSynthesis' in window) {
    speechSynthesis.getVoices();
    speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
}


// ─── Phased Progressive Loading Overlay ──────────────────────────────────────
function startPhasedLoading(initialText) {
    const steps = [
        "Analyzing your image…",
        "Identifying the material…",
        "Consulting the recycling assistant…",
        "Preparing your recommendation…"
    ];
    let stepIndex = 0;
    showLoading(initialText || steps[0]);

    if (loadingIntervalId) clearInterval(loadingIntervalId);
    loadingIntervalId = setInterval(() => {
        stepIndex++;
        if (stepIndex < steps.length) {
            const label = document.getElementById('loadingText');
            if (label) label.textContent = steps[stepIndex];
        }
    }, 700);
}

function showLoading(text) {
    const label = document.getElementById('loadingText');
    const overlay = document.getElementById('loadingOverlay');
    if (label) label.textContent = text || 'Analyzing waste...';
    if (overlay) overlay.classList.remove('hidden');
}

function hideLoading() {
    if (loadingIntervalId) {
        clearInterval(loadingIntervalId);
        loadingIntervalId = null;
    }
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.add('hidden');
}

function showError(msg) {
    const area = document.getElementById('resultsArea');
    if (area) {
        area.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>
                </div>
                <p><strong>Processing Notice</strong></p>
                <p class="empty-sub">${escapeHtml(msg)}</p>
                <button class="btn btn-primary btn-small" style="margin-top:1rem;" onclick="scrollToSection('scanner-section')">
                    <svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
                    Retry
                </button>
            </div>`;
    }
}

// ─── Category Filter & Modal System ──────────────────────────────────────────
function filterCategories(category) {
    // Update chip buttons
    document.querySelectorAll('.filter-chip').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === category);
    });

    // Filter cards
    document.querySelectorAll('.category-card').forEach(card => {
        const cardCat = card.dataset.category;
        if (category === 'all' || cardCat === category) {
            card.style.display = 'flex';
        } else {
            card.style.display = 'none';
        }
    });
}

function openCategoryModal(categoryKey) {
    const data = CATEGORY_DETAILS[categoryKey];
    if (!data) return;

    const modal = document.getElementById('categoryModal');
    const content = document.getElementById('categoryModalContent');
    if (!modal || !content) return;

    let resinHtml = '';
    if (data.resinCodes) {
        resinHtml = `
            <div style="margin-top:1.5rem;">
                <h4 style="font-family:var(--font-heading); font-size:1.05rem; margin-bottom:0.8rem; color:var(--text-primary);">
                    Resin Identification Codes (RIC #1 – #7):
                </h4>
                <div style="display:flex; flex-direction:column; gap:0.5rem;">
                    ${data.resinCodes.map(r => `
                        <div style="background:rgba(6,17,11,0.5); border:1px solid var(--border); padding:0.6rem 0.85rem; border-radius:var(--radius-sm); display:flex; justify-content:space-between; align-items:center; font-size:0.85rem;">
                            <div>
                                <strong style="color:var(--accent-mint);">${r.code}</strong> · ${r.name}
                                <div style="font-size:0.75rem; color:var(--text-muted);">${r.items}</div>
                            </div>
                            <span style="font-size:0.75rem; padding:0.2rem 0.6rem; border-radius:999px; background:rgba(16,185,129,0.15); color:var(--accent-mint);">
                                ${r.recyclable}
                            </span>
                        </div>
                    `).join('')}
                </div>
            </div>`;
    }

    content.innerHTML = `
        <div style="display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:1rem;">
            <div>
                <span style="font-family:var(--font-mono); font-size:0.75rem; color:var(--accent-mint); text-transform:uppercase;">
                    ${escapeHtml(data.badge)}
                </span>
                <h2 style="font-family:var(--font-heading); font-size:1.8rem; font-weight:800; color:var(--text-primary); margin:0.3rem 0;">
                    ${escapeHtml(data.title)}
                </h2>
                <span style="font-size:0.85rem; color:var(--text-secondary);">
                    Decomposition Time: <strong>${escapeHtml(data.decomposition)}</strong>
                </span>
            </div>
        </div>

        <p style="font-size:0.95rem; color:var(--text-secondary); line-height:1.6; margin-bottom:1.5rem;">
            ${escapeHtml(data.description)}
        </p>

        <h4 style="font-family:var(--font-heading); font-size:1.05rem; margin-bottom:0.8rem; color:var(--text-primary);">
            Preparation & Recycling Steps:
        </h4>
        <ul style="list-style:none; display:flex; flex-direction:column; gap:0.6rem; margin-bottom:1.5rem;">
            ${data.steps.map((s, i) => `
                <li style="background:rgba(6,17,11,0.4); border:1px solid rgba(110,231,183,0.1); padding:0.65rem 0.9rem; border-radius:var(--radius-sm); font-size:0.88rem; color:var(--text-secondary); line-height:1.5;">
                    <span style="display:inline-block; width:20px; height:20px; text-align:center; line-height:20px; border-radius:50%; background:var(--accent-dim); color:var(--accent-mint); font-size:0.72rem; font-weight:bold; margin-right:0.4rem;">${i+1}</span>
                    ${escapeHtml(s)}
                </li>
            `).join('')}
        </ul>

        ${resinHtml}

        <div style="margin-top:1.5rem; padding:1rem 1.2rem; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25); border-radius:var(--radius-md); font-size:0.86rem; color:var(--text-secondary); line-height:1.6; display:flex; align-items:flex-start; gap:0.5rem;">
            <svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--accent-mint); margin-top:0.2rem;"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
            <div><strong>Environmental Impact:</strong> ${escapeHtml(data.impact)}</div>
        </div>
    `;

    modal.classList.remove('hidden');
}

function closeCategoryModal(e) {
    if (e && e.target !== e.currentTarget && !e.target.classList.contains('modal-close-btn')) return;
    const modal = document.getElementById('categoryModal');
    if (modal) modal.classList.add('hidden');
}

// ─── Interactive Motion & Effects ────────────────────────────────────────────
function initScrollReveal() {
    // Assign incremental stagger indices to cards in grid collections
    const staggerContainers = document.querySelectorAll('.categories-grid, .stats-row, .pipeline-flow, .input-features-grid');
    staggerContainers.forEach(container => {
        Array.from(container.children).forEach((child, idx) => {
            child.style.setProperty('--stagger', idx);
        });
    });

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                // Trigger count-up if element is inside
                const counters = entry.target.querySelectorAll('.count-up');
                counters.forEach(c => runCountUp(c));
            }
        });
    }, { threshold: 0.12 });

    document.querySelectorAll('.reveal-on-scroll, .reveal-child').forEach(el => {
        observer.observe(el);
    });
}

function initTiltEffect() {
    // 3D Card Tilt on Mouse Move
    const cards = document.querySelectorAll('.tilt-card');
    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -7;
            const rotateY = ((x - centerX) / centerX) * 7;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-8px)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });
    });
}

function initHeroScannerSpotlight() {
    const spotlightEl = document.getElementById('heroScannerSpotlight');
    if (!spotlightEl) return;

    spotlightEl.addEventListener('mousemove', (e) => {
        const rect = spotlightEl.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        spotlightEl.style.boxShadow = `
            0 20px 60px rgba(0,0,0,0.7),
            ${(x - rect.width/2)/10}px ${(y - rect.height/2)/10}px 35px rgba(16, 185, 129, 0.22)
        `;
    });

    spotlightEl.addEventListener('mouseleave', () => {
        spotlightEl.style.boxShadow = '';
    });
}

function initNavbarScroll() {
    const header = document.getElementById('mainHeader');
    if (!header) return;
    window.addEventListener('scroll', () => {
        if (window.scrollY > 40) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    }, { passive: true });
}

function initCountUpNumbers() {
    document.querySelectorAll('.count-up').forEach(el => {
        // Only run for elements that don't need intersection
        if (el.dataset.target) {
            runCountUp(el);
        }
    });
}

function runCountUp(el) {
    if (el.dataset.counted === 'true') return;
    const target = parseInt(el.dataset.target || el.textContent, 10);
    if (isNaN(target)) return;

    el.dataset.counted = 'true';
    let current = 0;
    const step = Math.max(1, Math.floor(target / 25));
    const timer = setInterval(() => {
        current += step;
        if (current >= target) {
            el.textContent = target;
            clearInterval(timer);
        } else {
            el.textContent = current;
        }
    }, 40);
}

function animateNumber(elementId, targetNum) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const target = parseInt(targetNum, 10) || 0;
    let current = parseInt(el.textContent, 10) || 0;
    if (current === target) return;

    const diff = target - current;
    const step = diff > 0 ? Math.ceil(diff / 15) : Math.floor(diff / 15);
    const timer = setInterval(() => {
        current += step;
        if ((step > 0 && current >= target) || (step < 0 && current <= target)) {
            el.textContent = target;
            clearInterval(timer);
        } else {
            el.textContent = current;
        }
    }, 30);
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ─── Theme Management (Light & Dark Mode) ────────────────────────────────────
function initTheme() {
    let savedTheme = 'dark';
    try {
        savedTheme = localStorage.getItem('ecosort_theme') || 'dark';
    } catch (e) {
        savedTheme = 'dark';
    }
    applyTheme(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(newTheme);
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try {
        localStorage.setItem('ecosort_theme', theme);
    } catch (e) {}

    const themeLabel = document.getElementById('themeLabelText');
    if (themeLabel) {
        themeLabel.textContent = theme === 'light' ? 'Light' : 'Dark';
    }

    const mobileLabel = document.getElementById('mobileThemeLabelText');
    if (mobileLabel) {
        mobileLabel.textContent = theme === 'light' ? 'Light Mode' : 'Dark Mode';
    }

    const themeBtn = document.getElementById('themeToggleBtn');
    if (themeBtn) {
        themeBtn.setAttribute('title', theme === 'light' ? 'Switch to Dark Mode' : 'Switch to Light Mode');
    }
}

// ─── Fullscreen Lightbox & Detailed Image Inspection ─────────────────────────
function openImageLightbox(src) {
    if (!src) return;
    const modal = document.getElementById('imageLightboxModal');
    const img = document.getElementById('lightboxImage');
    const downloadBtn = document.getElementById('lightboxDownloadBtn');
    const titleEl = document.getElementById('lightboxTitle');
    const badgesContainer = document.getElementById('lightboxBadges');

    if (!modal || !img) return;

    img.src = src;
    if (downloadBtn) {
        downloadBtn.href = src;
    }

    if (titleEl) {
        if (currentDetectionData && currentDetectionData.object_guess) {
            titleEl.textContent = `YOLO Inspection — ${currentDetectionData.object_guess}`;
        } else {
            titleEl.textContent = 'YOLO Detection Analysis — Detailed Inspection';
        }
    }

    if (badgesContainer) {
        if (currentDetectionData && currentDetectionData.detections && currentDetectionData.detections.length > 0) {
            badgesContainer.innerHTML = currentDetectionData.detections.map(det => {
                const name = det.detected_object || det.class_name || 'Item';
                const conf = (det.confidence * 100).toFixed(0);
                const color = WASTE_COLORS[det.class_name] || '#10b981';
                return `<span class="lightbox-det-badge" style="background:${color}22; color:${color}; border:1px solid ${color}55;"><span style="width:7px; height:7px; border-radius:50%; background:${color}; display:inline-block; margin-right:4px;"></span>${escapeHtml(name)} · ${conf}%</span>`;
            }).join('');
        } else {
            badgesContainer.innerHTML = '<span class="lightbox-det-badge" style="background:var(--accent-dim); color:var(--accent-mint); border:1px solid var(--border);">100% High Resolution Result</span>';
        }
    }

    resetLightboxZoom();
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeImageLightbox(event) {
    if (event) {
        // If clicking directly on the image itself, don't close
        if (event.target && event.target.id === 'lightboxImage') {
            return;
        }
        // If user was actively dragging/panning the image, don't close on release
        if (didLightboxDrag) {
            didLightboxDrag = false;
            return;
        }
    }
    const modal = document.getElementById('imageLightboxModal');
    if (modal) {
        modal.classList.add('hidden');
        document.body.style.overflow = '';
        resetLightboxZoom();
    }
}

function zoomLightbox(delta) {
    const newZoom = Math.min(Math.max(0.5, lightboxZoom + delta), 4.0);
    lightboxZoom = Math.round(newZoom * 100) / 100;
    if (lightboxZoom <= 1.0) {
        lightboxPanX = 0;
        lightboxPanY = 0;
    }
    updateLightboxTransform();
}

function resetLightboxZoom() {
    lightboxZoom = 1.0;
    lightboxPanX = 0;
    lightboxPanY = 0;
    updateLightboxTransform();
}

function updateLightboxTransform() {
    const canvas = document.getElementById('lightboxCanvas');
    const zoomLevelEl = document.getElementById('lightboxZoomLevel');
    const viewport = document.getElementById('lightboxViewport');

    if (canvas) {
        canvas.style.transform = `translate(${lightboxPanX}px, ${lightboxPanY}px) scale(${lightboxZoom})`;
    }
    if (viewport) {
        viewport.style.cursor = lightboxZoom > 1.0 ? (isLightboxPanning ? 'grabbing' : 'grab') : 'default';
    }
    if (zoomLevelEl) {
        zoomLevelEl.textContent = `${Math.round(lightboxZoom * 100)}%`;
    }
}

function initLightboxEvents() {
    const viewport = document.getElementById('lightboxViewport');
    if (!viewport) return;

    // Mouse wheel zoom
    viewport.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY < 0 ? 0.25 : -0.25;
        zoomLightbox(delta);
    }, { passive: false });

    // Drag / Pan mouse events
    viewport.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        lightboxMouseDownX = e.clientX;
        lightboxMouseDownY = e.clientY;
        didLightboxDrag = false;
        if (lightboxZoom <= 1.0) return;
        isLightboxPanning = true;
        lightboxStartX = e.clientX - lightboxPanX;
        lightboxStartY = e.clientY - lightboxPanY;
        viewport.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
        if (!isLightboxPanning) return;
        if (Math.abs(e.clientX - lightboxMouseDownX) > 6 || Math.abs(e.clientY - lightboxMouseDownY) > 6) {
            didLightboxDrag = true;
        }
        lightboxPanX = e.clientX - lightboxStartX;
        lightboxPanY = e.clientY - lightboxStartY;
        updateLightboxTransform();
    });

    window.addEventListener('mouseup', () => {
        if (isLightboxPanning) {
            isLightboxPanning = false;
            updateLightboxTransform();
        }
    });

    // Touch support for mobile devices
    let touchStartX = 0;
    let touchStartY = 0;
    viewport.addEventListener('touchstart', (e) => {
        if (e.touches.length === 1 && lightboxZoom > 1.0) {
            isLightboxPanning = true;
            touchStartX = e.touches[0].clientX - lightboxPanX;
            touchStartY = e.touches[0].clientY - lightboxPanY;
        }
    }, { passive: true });

    viewport.addEventListener('touchmove', (e) => {
        if (isLightboxPanning && e.touches.length === 1) {
            lightboxPanX = e.touches[0].clientX - touchStartX;
            lightboxPanY = e.touches[0].clientY - touchStartY;
            updateLightboxTransform();
        }
    }, { passive: true });

    viewport.addEventListener('touchend', () => {
        isLightboxPanning = false;
    }, { passive: true });

    // Keyboard ESC to close
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const modal = document.getElementById('imageLightboxModal');
            if (modal && !modal.classList.contains('hidden')) {
                closeImageLightbox();
            }
        }
    });
}

// ─── Interactive Floating Sample Image Loader ──────────────────────────────────
async function loadSampleImage(imageUrl, label) {
    try {
        scrollToSection('scanner-section');
        setMode('gallery');
        setScanningActive(true, `Loading sample: ${label || 'Waste Item'}...`);

        // Fetch image as Blob and convert to Base64 Data URL
        const resp = await fetch(imageUrl);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const blob = await resp.blob();

        const reader = new FileReader();
        reader.onload = function(e) {
            currentImageB64 = e.target.result;
            const preview = document.getElementById('previewImage');
            if (preview) preview.src = currentImageB64;
            const prevArea = document.getElementById('previewArea');
            if (prevArea) prevArea.classList.remove('hidden');

            // Scroll scanner spotlight into clear view
            const galArea = document.getElementById('galleryMode');
            if (galArea) {
                galArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }

            // Automatically trigger the real AI detection pipeline!
            analyzeImage();
        };
        reader.readAsDataURL(blob);
    } catch (err) {
        console.error('Failed to load sample image:', err);
        setScanningActive(false);
        if (typeof showError === 'function') {
            showError(`Could not load sample: ${err.message}. Please select an image manually.`);
        }
    }
}
window.loadSampleImage = loadSampleImage;

// ─── 3D Card Tilt & Parallax Physics ─────────────────────────────────────────
function initCardTiltPhysics() {
    const tiltCards = document.querySelectorAll('.floating-interactive-item, .tilt-card, .scanner-card');
    tiltCards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -8;
            const rotateY = ((x - centerX) / centerX) * 8;
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.03, 1.03, 1.03)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });
    });
}

// ─── Regional Recycling Rules Selector ───────────────────────────────────────
const REGIONAL_RULES = {
    global: {
        flag: "🌐",
        name: "🌐 Global ISO",
        fullName: "Global Standard (ISO 15270)",
        rules: "Standard 3-bin stream: Recyclables, Organics, Residual Landfill. Empty & dry containers."
    },
    munich: {
        flag: "🇩🇪",
        name: "🇩🇪 Munich (Pfand)",
        fullName: "Munich, Germany (Pfand & Dual System)",
        rules: "0.25€ deposit on PET and aluminum cans. Return to Pfandautomat. Gelber Sack for light packaging."
    },
    california: {
        flag: "🇺🇸",
        name: "🇺🇸 California (SB 54)",
        fullName: "California, USA (SB 54 / CRV)",
        rules: "CRV eligible containers return 5¢ - 10¢. Clean plastics #1, #2, #5 in blue bin. Food-soiled paper in green bin."
    },
    tokyo: {
        flag: "🇯🇵",
        name: "🇯🇵 Tokyo (3-Stream)",
        fullName: "Tokyo, Japan (Burnable / Non-Burnable / Resources)",
        rules: "Strict separation: Remove bottle caps & labels into plastic. Wash PET body and crush flat for Resource day."
    },
    india: {
        flag: "🇮🇳",
        name: "🇮🇳 India (SWM 2016)",
        fullName: "India (SWM Rules 2016)",
        rules: "Segregate at source into Dry (Blue), Wet (Green), and Sanitary/Hazardous (Red). Rinse multi-layer pouches."
    }
};

function selectRegionalRules(regionKey) {
    const region = REGIONAL_RULES[regionKey] || REGIONAL_RULES.global;
    const badge = document.getElementById('currentRegionBadge');
    if (badge) badge.textContent = region.name;
    
    const sel = document.getElementById('regionSelect');
    if (sel && sel.value !== regionKey) sel.value = regionKey;
    const mobileSel = document.getElementById('mobileRegionSelect');
    if (mobileSel && mobileSel.value !== regionKey) mobileSel.value = regionKey;
    
    // Store in localStorage
    try {
        localStorage.setItem('ecosort_region', regionKey);
    } catch (e) {}
}
window.selectRegionalRules = selectRegionalRules;

// ─── Scroll-Triggered Fade-In Animations (IntersectionObserver) ──────────────
function initScrollAnimations() {
    const revealElements = document.querySelectorAll('.reveal-on-scroll');
    if (!revealElements || !revealElements.length) return;

    if ('IntersectionObserver' in window) {
        const observerOptions = {
            root: null,
            rootMargin: '0px 0px -40px 0px',
            threshold: 0.1
        };

        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                    // Unobserve to save CPU cycles once revealed
                    obs.unobserve(entry.target);
                }
            });
        }, observerOptions);

        revealElements.forEach(el => {
            const rect = el.getBoundingClientRect();
            // If already within or above the viewport, reveal immediately
            if (rect.top < window.innerHeight) {
                el.classList.add('is-visible');
            } else {
                observer.observe(el);
            }
        });
    } else {
        // Fallback for older browsers
        revealElements.forEach(el => el.classList.add('is-visible'));
    }
}
window.initScrollAnimations = initScrollAnimations;

// ─── Student & Pro Jargon Guide Accordion Toggle ─────────────────────────────
function toggleJargonAccordion() {
    const drawer = document.getElementById('jargonContentDrawer');
    const btn = document.getElementById('jargonToggleBtn');
    if (!drawer) return;

    const isHidden = drawer.classList.contains('hidden');
    if (isHidden) {
        drawer.classList.remove('hidden');
        if (btn) {
            btn.classList.add('is-open');
            const txt = btn.querySelector('.toggle-icon-text');
            if (txt) txt.textContent = 'Collapse Guide';
        }
    } else {
        drawer.classList.add('hidden');
        if (btn) {
            btn.classList.remove('is-open');
            const txt = btn.querySelector('.toggle-icon-text');
            if (txt) txt.textContent = 'Expand Guide';
        }
    }
}
window.toggleJargonAccordion = toggleJargonAccordion;

// Auto-run on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initCardTiltPhysics();
        initScrollAnimations();
        const savedRegion = localStorage.getItem('ecosort_region') || 'munich';
        selectRegionalRules(savedRegion);
    });
} else {
    initCardTiltPhysics();
    initScrollAnimations();
    const savedRegion = localStorage.getItem('ecosort_region') || 'munich';
    selectRegionalRules(savedRegion);
}


// ═══════════════════════════════════════════════════════════════════════════════
// RESIN IDENTIFICATION CODE (RIC #1-#7) EXPLORER INTERACTION
// ═══════════════════════════════════════════════════════════════════════════════
const RESIN_DATA = {
    1: {
        emblem: '♳',
        badge: 'RIC #1 · ASTM D7611 · THERMOPLASTIC POLYESTER',
        fullName: 'Polyethylene Terephthalate (PET / PETE)',
        verdictClass: 'badge-high-value',
        verdictText: 'Highest Value · Universally Curbside Accepted',
        products: 'Single-use water bottles, carbonated beverage bottles, cooking oil bottles, peanut butter containers, salad dressing jars, clamshell berry boxes.',
        circularMarket: 'Mechanically washed and pelletized into food-grade rPET bottles, polar fleece outerwear, geo-textiles, industrial strapping, and synthetic carpets.',
        prepTip: 'Empty all liquids and rinse with a splash of water. Leave caps screwed tightly on—loose caps slip through MRF optical sorting screens and get landfilled.',
        contaminationWarning: 'Unrinsed sugar syrups degrade remelted optical clarity. Even 0.1% PVC contamination ruins an entire 10-ton PET reprocessing batch with corrosive acid.',
        economicStat: '<strong>Commodity Value:</strong> $450–$620 per metric ton sorted clear flake. Highest recycling rate of all post-consumer packaging polymers.'
    },
    2: {
        emblem: '♴',
        badge: 'RIC #2 · ASTM D7611 · HIGH-DENSITY POLYETHYLENE',
        fullName: 'High-Density Polyethylene (HDPE)',
        verdictClass: 'badge-high-value',
        verdictText: 'Highest Value · Universally Curbside Accepted',
        products: 'Translucent gallon milk jugs, household detergent bottles, shampoo and conditioner bottles, motor oil jugs, bleach containers, vitamin bottles.',
        circularMarket: 'Remelted into agricultural drainage piping, outdoor composite plastic lumber, recycling bins, crates, non-food detergent jugs, and picnic tables.',
        prepTip: 'Rinse thoroughly to remove viscous surfactants. Unpigmented (natural milk jug) HDPE commands premium market pricing over colored detergent bottles.',
        contaminationWarning: 'Motor oil residue or chemical weed-killer remnants render HDPE hazardous waste and contaminate clean curbside washing tanks.',
        economicStat: '<strong>Commodity Value:</strong> $500–$750 per metric ton unpigmented natural flake. In extremely high demand for industrial pipe manufacturing.'
    },
    3: {
        emblem: '♵',
        badge: 'RIC #3 · ASTM D7611 · CHLORINATED POLYMER',
        fullName: 'Polyvinyl Chloride (PVC / Vinyl)',
        verdictClass: 'badge-hazard',
        verdictText: 'Rarely Accepted Curbside · Hazardous Contaminant',
        products: 'Rigid plumbing pipes, electrical conduit, blister packaging, clear medical tubing, vinyl flooring tiles, credit cards, faux leather upholstery.',
        circularMarket: 'Extremely limited curbside reprocessing. Recycled primarily via dedicated construction drop-offs into traffic speed bumps, mudflaps, and cable sheathing.',
        prepTip: 'Do NOT toss into household curbside recycling bins. Take rigid PVC construction off-cuts to specialized architectural salvage yards.',
        contaminationWarning: 'Contains chlorine additives. When melted with other plastics, it releases corrosive hydrochloric acid and toxic dioxins, destroying recycling extruders.',
        economicStat: '<strong>Commodity Value:</strong> Negative economic value in municipal MRFs; classified as a major hazardous contaminant in PET sorting bales.'
    },
    4: {
        emblem: '♶',
        badge: 'RIC #4 · ASTM D7611 · LOW-DENSITY POLYETHYLENE',
        fullName: 'Low-Density Polyethylene (LDPE)',
        verdictClass: 'badge-high-value',
        verdictText: 'Store Drop-Off Only · Clean Stream Recyclable',
        products: 'Grocery shopping bags, bread bags, dry cleaning bags, bubble wrap, stretch film wrap, flexible squeeze condiment bottles, 6-pack beverage rings.',
        circularMarket: 'Processed in dedicated clean plastic film streams into trash can liners, shipping envelopes, composite decking (e.g. Trex), and agricultural mulch film.',
        prepTip: 'Never place loose in curbside bins! Thin films wrap around rotating sorting disks, shutting down entire recovery facilities. Drop at supermarket film bins.',
        contaminationWarning: 'Food grease and crumbs glued to thin film cause mold and bacterial decay, destroying entire film bales.',
        economicStat: '<strong>Commodity Value:</strong> $220–$380 per ton when bundled into clean, dry post-commercial film bales.'
    },
    5: {
        emblem: '♷',
        badge: 'RIC #5 · ASTM D7611 · POLYPROPYLENE',
        fullName: 'Polypropylene (PP)',
        verdictClass: 'badge-high-value',
        verdictText: 'Widely Accepted · High Heat Resistance',
        products: 'Yogurt containers, sour cream tubs, margarine tubs, prescription medicine bottles, reusable lunchboxes, syrup bottles, automotive battery casings, plastic bottle caps.',
        circularMarket: 'Remelted into automotive battery cases, signal lights, broom bristles, landscape borders, ice scrapers, storage bins, and bicycle racks.',
        prepTip: 'Wipe and rinse residual dairy fats. Check municipal guidelines—over 60% of US and European municipalities now actively sort PP.',
        contaminationWarning: 'Food oils embedded in porous PP walls cause odor degradation in post-consumer pellets if not sanitized with caustic wash.',
        economicStat: '<strong>Commodity Value:</strong> $320–$510 per ton. One of the fastest-growing recycled plastics markets globally.'
    },
    6: {
        emblem: '♸',
        badge: 'RIC #6 · ASTM D7611 · EXPANDED POLYSTYRENE',
        fullName: 'Polystyrene (PS / Styrofoam)',
        verdictClass: 'badge-hazard',
        verdictText: 'Rarely Accepted Curbside · Cost-Prohibitive',
        products: 'Expanded foam takeout clamshells, disposable coffee cups, meat and poultry trays, packing peanuts, CD jewel cases, disposable plastic cutlery, foam insulation.',
        circularMarket: 'Rarely viable curbside because it is 95% air and cost-prohibitive to transport. Specialized facilities densify it into architectural moldings and picture frames.',
        prepTip: 'Reuse packing peanuts for shipping packages. Check for specialized architectural drop-offs with cold-densifier or thermal-compaction equipment.',
        contaminationWarning: 'Brittle structure fractures into microscopic plastic beads that contaminate waterways and leaches residual styrene monomer under heat.',
        economicStat: '<strong>Commodity Value:</strong> Extremely low / negative municipal value due to high transportation volume and low bulk density.'
    },
    7: {
        emblem: '♹',
        badge: 'RIC #7 · ASTM D7611 · COMPOSITE / OTHER',
        fullName: 'Other / Multi-Layer Plastics & Biopolymers',
        verdictClass: 'badge-hazard',
        verdictText: 'Specialized Facility Only · Non-Standard',
        products: 'Large 5-gallon water dispenser carboys, sunglasses lenses, bulletproof polycarbonate panels, multi-layer metallized chip bags, PLA compostable plastics, nylon.',
        circularMarket: 'Mechanical recycling is generally not possible for mixed composites. Polycarbonate bottles can be specialized-reprocessed; PLA requires commercial composting.',
        prepTip: 'Do not mix PLA (corn-based bioplastics) into curbside plastics—PLA has a lower melting point and ruins regular PET and HDPE extrusion batches.',
        contaminationWarning: 'Polycarbonate may contain Bisphenol-A (BPA). Multi-layer polymer foil pouches cannot be separated with standard thermal float-sink tanks.',
        economicStat: '<strong>Commodity Value:</strong> Highly variable; specialized recovery only for pure unmixed polycarbonate or nylon industrial scrap.'
    }
};

function selectResinCode(code) {
    const data = RESIN_DATA[code];
    if (!data) return;

    // Update active state on dial buttons
    const buttons = document.querySelectorAll('.resin-dial-btn');
    buttons.forEach(btn => {
        const ric = parseInt(btn.getAttribute('data-ric'), 10);
        if (ric === code) {
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
        } else {
            btn.classList.remove('active');
            btn.setAttribute('aria-selected', 'false');
        }
    });

    // Animate details card crossfade
    const card = document.getElementById('resinDisplayPanel');
    if (!card) return;

    card.style.opacity = '0.5';
    card.style.transform = 'translateY(4px)';

    setTimeout(() => {
        const emblem = document.getElementById('resinEmblem');
        const badge = document.getElementById('resinCodeBadge');
        const fullName = document.getElementById('resinFullName');
        const verdictBadge = document.getElementById('resinVerdictBadge');
        const verdictText = document.getElementById('resinVerdictText');
        const products = document.getElementById('resinProducts');
        const circularMarket = document.getElementById('resinCircularMarket');
        const prepTip = document.getElementById('resinPrepTip');
        const warning = document.getElementById('resinContaminationWarning');
        const economicStat = document.getElementById('resinEconomicStat');

        if (emblem) emblem.textContent = data.emblem;
        if (badge) badge.textContent = data.badge;
        if (fullName) fullName.textContent = data.fullName;
        if (verdictBadge) {
            verdictBadge.className = 'resin-verdict-badge ' + data.verdictClass;
        }
        if (verdictText) verdictText.textContent = data.verdictText;
        if (products) products.textContent = data.products;
        if (circularMarket) circularMarket.textContent = data.circularMarket;
        if (prepTip) prepTip.textContent = data.prepTip;
        if (warning) warning.textContent = data.contaminationWarning;
        if (economicStat) economicStat.innerHTML = data.economicStat;

        card.style.opacity = '1';
        card.style.transform = 'translateY(0)';
    }, 120);
}
window.selectResinCode = selectResinCode;


// ═══════════════════════════════════════════════════════════════════════════════
// ANIMATED IMPACT COUNTERS (COUNT-UP INTERPOLATION)
// ═══════════════════════════════════════════════════════════════════════════════
function initImpactCounters() {
    const counterElements = document.querySelectorAll('.impact-number[data-target]');
    if (!counterElements.length) return;

    const animateCounter = (el) => {
        const target = parseFloat(el.getAttribute('data-target'));
        const decimals = parseInt(el.getAttribute('data-decimals') || '0', 10);
        const duration = 1800; // ms
        const startTime = performance.now();

        const updateValue = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out cubic
            const easeOutProgress = 1 - Math.pow(1 - progress, 3);
            const currentVal = target * easeOutProgress;

            if (decimals > 0) {
                el.textContent = currentVal.toFixed(decimals);
            } else {
                el.textContent = Math.round(currentVal).toLocaleString();
            }

            if (progress < 1) {
                requestAnimationFrame(updateValue);
            } else {
                if (decimals > 0) {
                    el.textContent = target.toFixed(decimals);
                } else {
                    el.textContent = Math.round(target).toLocaleString();
                }
            }
        };

        requestAnimationFrame(updateValue);
    };

    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateCounter(entry.target);
                    obs.unobserve(entry.target);
                }
            });
        }, { threshold: 0.25 });

        counterElements.forEach(el => observer.observe(el));
    } else {
        counterElements.forEach(el => animateCounter(el));
    }
}
window.initImpactCounters = initImpactCounters;

// Initialize on DOM Ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initImpactCounters();
    });
} else {
    initImpactCounters();
}


// ═══════════════════════════════════════════════════════════════════════════════
// STUDENT & PROFESSIONAL EDUCATIONAL GUIDE INTERACTIONS
// ═══════════════════════════════════════════════════════════════════════════════
function filterEducationalTrack(track) {
    const buttons = document.querySelectorAll('.guide-filter-btn');
    buttons.forEach(btn => {
        const matches = btn.getAttribute('data-filter') === track;
        btn.classList.toggle('active', matches);
        btn.setAttribute('aria-selected', matches ? 'true' : 'false');
    });

    const cards = document.querySelectorAll('.educational-card');
    cards.forEach(card => {
        const cardTrack = card.getAttribute('data-track');
        if (track === 'all' || cardTrack === track) {
            card.style.display = 'flex';
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 30);
        } else {
            card.style.opacity = '0';
            card.style.transform = 'translateY(6px)';
            setTimeout(() => {
                card.style.display = 'none';
            }, 200);
        }
    });
}
window.filterEducationalTrack = filterEducationalTrack;

function toggleGuideDrawer(drawerNum, btnEl) {
    const drawer = document.getElementById('specDrawer' + drawerNum);
    if (!drawer) return;

    const isHidden = drawer.classList.contains('hidden');
    if (isHidden) {
        drawer.classList.remove('hidden');
        if (btnEl) {
            btnEl.classList.add('is-open');
            btnEl.setAttribute('aria-expanded', 'true');
            const textEl = btnEl.querySelector('.btn-text');
            const iconEl = btnEl.querySelector('.btn-icon');
            if (textEl) textEl.textContent = 'Hide Specification';
            if (iconEl) iconEl.innerHTML = '&minus;';
        }
    } else {
        drawer.classList.add('hidden');
        if (btnEl) {
            btnEl.classList.remove('is-open');
            btnEl.setAttribute('aria-expanded', 'false');
            const textEl = btnEl.querySelector('.btn-text');
            const iconEl = btnEl.querySelector('.btn-icon');
            if (textEl) textEl.textContent = 'Technical Specification';
            if (iconEl) iconEl.innerHTML = '&plus;';
        }
    }
}
window.toggleGuideDrawer = toggleGuideDrawer;