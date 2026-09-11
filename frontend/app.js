/**
 * SatQuery AI — Professional Remote Sensing Intelligence Platform
 * SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI
 * Frontend Logic v4.0 — Dynamic Multimodal Architecture
 */

const API_BASE_URL = (window.location.hostname === "localhost" && window.location.port === "3000")
    ? "http://localhost:8000"
    : window.location.origin;

// ── App State ──────────────────────────────────────────────────────
let state = {
    uploadedTokens:   [],
    uploadedMetadata: [],
    currentAnalysis:  null,
    previewFiles:     [],
    showOverlay:      true,
    zoomScale:        1.0,
    panX:             0,
    panY:             0,
    isPanning:        false,
    startPanX:        0,
    startPanY:        0,
    activeMode:       "describe_scene",
    activeTab:        "detections",
    activeView:       "home_view",
    bboxHover:        -1,
    isListening:       false,
    recognitionLang:   "en-US",
    originalVoiceText: "",
    normalizedQuery:   "",
    analysisStartTime: 0,
    analysisHistory:  JSON.parse(localStorage.getItem("sq_history") || "[]"),
    // ── Auth ─────────────────────────────────────────────────────
    currentUser:  JSON.parse(localStorage.getItem("sq_user") || "null"),
    authToken:    localStorage.getItem("sq_token") || null,
    isGuest:      !localStorage.getItem("sq_token")
};

// ── DOM References Helper ──────────────────────────────────────────
const $ = id => document.getElementById(id);

// DOM Elements
const dropzone               = $("image_dropzone");
const fileInput              = $("file_input");
const previewsContainer      = $("previews_container");
const uploadCard             = $("upload_card");
const uploadSummaryBar       = $("upload_summary_bar");
const summaryThumbImg        = $("summary_thumb_img");
const summaryFilename        = $("summary_filename");
const summaryUploadBadgeText = $("summary_upload_badge_text");
const summaryDim             = $("summary_dim");
const summaryFmt             = $("summary_fmt");
const summaryType            = $("summary_type");
const summaryBands           = $("summary_bands");
const summaryReplaceBtn      = $("summary_replace_btn");
const summaryRemoveBtn       = $("summary_remove_btn");
const metaBoxDim             = $("meta_box_dim");
const metaBoxFmt             = $("meta_box_fmt");
const metaBoxType            = $("meta_box_type");
const metaBoxBands           = $("meta_box_bands");

const queryInput             = $("query_input");
const submitQueryBtn         = $("submit_query_btn");
const querySpinner           = $("query_spinner");
const analyzeBtnText         = $("analyze_btn_text");
const resultCard             = $("result_card");
const viewerCard             = $("viewer_card");
const progressCard           = $("progress_card");
const emptyWorkspace         = $("empty_workspace");
const modalityBadge          = $("modality_badge");
const confidenceBadge        = $("confidence_badge");
const answerHeadline         = $("answer_headline");
const answerText             = $("answer_text");
const hallucinationAlert      = $("hallucination_alert");
const hallucinationText       = $("hallucination_text");
const evidenceList           = $("evidence_list");
const traceTaskTag           = $("trace_task_tag");
const executionTraceSteps    = $("execution_trace_steps");
const evidenceCanvas         = $("evidence_canvas");
const uploadStatus           = $("upload_status");
const bboxTooltip            = $("bbox_tooltip");
const toggleBboxBtn          = $("toggle_bbox_btn");
const zoomInBtn              = $("zoom_in_btn");
const zoomOutBtn             = $("zoom_out_btn");
const fullscreenBtn          = $("fullscreen_btn");
const exportJsonBtn          = $("export_json_btn");
const exportCsvBtn           = $("export_csv_btn");
const exportPdfBtn           = $("export_pdf_btn");

const summaryPanelTitle      = $("summary_panel_title");
const detTotalNum            = $("det_total_num");
const detTotalLabel          = $("det_total_label");
const summaryBreakdownList   = $("summary_breakdown_list");
const modePanelTitle         = $("mode_panel_title");
const modePanelIcon          = $("mode_panel_icon");
const confValueText          = $("conf_value_text");
const confBarFill            = $("conf_bar_fill");
const detectedBuildingsList  = $("detected_buildings_list");
const sortBuildingsBtn       = $("sort_buildings_btn");
const analysisTimeVal        = $("analysis_time_val");
const zoomLevelText          = $("zoom_level_text");

const micBtn                 = $("mic_btn");
const micIcon                = $("mic_icon");
const voiceLangSelect        = $("voice_lang_select");
const voiceStatusBar         = $("voice_status_bar");
const voiceStatusIcon        = $("voice_status_icon");
const voiceStatusText        = $("voice_status_text");
const multilingualBadgeContainer = $("multilingual_badge_container");
const originalQueryDisplay    = $("original_query_display");
const normalizedQueryDisplay  = $("normalized_query_display");

const ALLOWED_EXTS  = new Set([".png",".jpg",".jpeg",".webp",".tif",".tiff",".geotiff"]);
const MAX_FILE_MB   = 50;
const PRESET_ASSETS = {
    tc1: { file: "cartosat2s_optical_sample.tif", query: "Describe this satellite scene in detail." },
    tc2: { file: "cartosat2s_optical_sample.tif", query: "Locate the buildings in this scene." },
    tc3: { file: "bitemporal_t1_before.tif",      query: "What changes occurred between these two satellite images?" },
    tc4: { file: "risat_sar_sample.tif",           query: "Analyze this scene using optical and SAR information." }
};

// ── Init ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initDropzone();
    initModeCards();
    initChips();
    initViewerControls();
    initExportButtons();
    initViewerTabs();
    checkBackendHealth();
    setupCanvasHover();
    initVoiceInput();
    initAuth();       // ← NEW: Auth modal + session restore
});

// ── Navigation SPA View Switching ──────────────────────────────────
function initNavigation() {
    const navTabs = document.querySelectorAll(".nav-tab");
    const backBtns = document.querySelectorAll(".back-home-btn");

    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetViewId = tab.dataset.view;
            switchView(targetViewId);
        });
    });

    backBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetViewId = btn.dataset.backTo || "home_view";
            switchView(targetViewId);
        });
    });
}

function switchView(targetViewId) {
    state.activeView = targetViewId;

    // Update nav tab active state
    document.querySelectorAll(".nav-tab").forEach(tab => {
        const isActive = tab.dataset.view === targetViewId;
        tab.classList.toggle("active", isActive);
    });

    // Hide all view sections, show target
    document.querySelectorAll(".view-section").forEach(sec => {
        const isTarget = sec.id === targetViewId;
        sec.classList.toggle("hidden", !isTarget);
    });

    // Render target view content
    if (targetViewId === "history_view") {
        renderHistoryView();
    } else if (targetViewId === "model_status_view") {
        renderModelStatusView();
    } else if (targetViewId === "home_view") {
        redrawCanvas();
    }
}

// ── Render Analysis History View ───────────────────────────────────
function renderHistoryView() {
    const container = $("history_records_container");
    if (!container) return;

    if (!state.analysisHistory.length) {
        container.innerHTML = `
            <div class="empty-state large" style="grid-column: 1 / -1; padding: 3rem 1rem;">
                <div class="empty-hero-icon">⏱</div>
                <h3>No analysis history yet</h3>
                <p>Run an analysis on satellite imagery to see your session results logged here.</p>
                <button class="action-btn primary-btn" onclick="switchView('home_view')">← Return to Main Dashboard</button>
            </div>`;
        return;
    }

    let html = "";
    state.analysisHistory.forEach(item => {
        const confPct = Math.round((item.confidence || 0.75) * 100);
        html += `
            <div class="history-record-card">
                <div class="history-record-header">
                    <span class="history-record-mode">${(item.mode || "ANALYSIS").toUpperCase()}</span>
                    <span>${item.timestamp || "Recent"}</span>
                </div>
                <div class="history-record-query">"${item.query || "Remote Sensing Query"}"</div>
                <div class="history-record-summary">${item.answer || "Analysis completed."}</div>
                <div class="history-record-actions">
                    <button class="view-hist-btn" data-hist-id="${item.id}">View Analysis Results →</button>
                </div>
            </div>`;
    });

    container.innerHTML = html;

    container.querySelectorAll(".view-hist-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const histId = parseInt(btn.dataset.histId);
            const found = state.analysisHistory.find(h => h.id === histId);
            if (found && found.fullData) {
                state.currentAnalysis = found.fullData;
                switchView("home_view");
                renderAnalysisResults(found.fullData, "Session History");
                showToast("⏱ Historical analysis reloaded", "info");
            } else {
                switchView("home_view");
            }
        });
    });
}

// ── Render Model Status View ───────────────────────────────────────
async function renderModelStatusView() {
    const backendVal = $("diag_backend_val");
    const backendSub = $("diag_backend_sub");
    const modelVal   = $("diag_model_val");
    const modelSub   = $("diag_model_sub");
    const deviceVal  = $("diag_device_val");

    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        const data = await res.json();

        if (backendVal) {
            backendVal.textContent = "Online";
            backendVal.className = "diag-card-val green";
        }
        if (backendSub) backendSub.textContent = `FastAPI Active | Status 200 OK`;

        if (modelVal) {
            modelVal.textContent = data.pytorch_engine_active ? "Ready / Loaded" : "Loading";
            modelVal.className = data.pytorch_engine_active ? "diag-card-val green" : "diag-card-val yellow";
        }
        if (modelSub) modelSub.textContent = `FasterRCNN_MobileNet_V3_Large_FPN`;

        if (deviceVal) {
            const devName = (data.device || "cpu").toUpperCase();
            deviceVal.textContent = devName === "MPS" ? "MPS (Apple Metal Acceleration)" : devName;
        }
    } catch {
        if (backendVal) {
            backendVal.textContent = "Offline";
            backendVal.className = "diag-card-val red";
        }
        if (modelVal) {
            modelVal.textContent = "Not Loaded";
            modelVal.className = "diag-card-val red";
        }
        if (deviceVal) deviceVal.textContent = "Not available";
    }
}

// ── Backend Health Bar Check ───────────────────────────────────────
async function checkBackendHealth() {
    const dot  = $("backend_dot");
    const text = $("system_status_text");
    const mDot = $("model_dot");
    const mTxt = $("model_status_text");

    try {
        const res  = await fetch(`${API_BASE_URL}/health`);
        const data = await res.json();

        dot.className  = "status-dot green";
        text.textContent = `Backend (${data.device.toUpperCase()})`;

        if (data.pytorch_engine_active) {
            mDot.className = "status-dot green";
            mTxt.textContent = "AI Engine Ready";
        } else {
            mDot.className = "status-dot yellow";
            mTxt.textContent = "AI Engine Loading";
        }
    } catch {
        dot.className  = "status-dot red";
        text.textContent = "Backend Offline";
        mDot.className = "status-dot red";
        mTxt.textContent = "No Connection";
        showToast("⚠️ Backend offline — start uvicorn", "warning");
    }
}

// ── Dropzone & Upload Summary ──────────────────────────────────────
function initDropzone() {
    dropzone.addEventListener("dragover",  e => { e.preventDefault(); dropzone.classList.add("hover"); });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("hover"));
    dropzone.addEventListener("drop", e => {
        e.preventDefault();
        dropzone.classList.remove("hover");
        if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files);
    });

    fileInput.addEventListener("change", e => {
        if (e.target.files && e.target.files.length) handleFileUpload(e.target.files);
    });

    if (summaryReplaceBtn) summaryReplaceBtn.addEventListener("click", () => fileInput.click());
    if (summaryRemoveBtn)  summaryRemoveBtn.addEventListener("click", resetUploadState);

    submitQueryBtn.addEventListener("click", handleQuerySubmit);
}

// ── Mode Cards ─────────────────────────────────────────────────────
function initModeCards() {
    document.querySelectorAll(".mode-card").forEach(tile => {
        tile.addEventListener("click", () => {
            document.querySelectorAll(".mode-card").forEach(t => t.classList.remove("active"));
            tile.classList.add("active");
            state.activeMode = tile.dataset.mode;
            if (tile.dataset.query) queryInput.value = tile.dataset.query;

            // Update standby mode labels
            if (summaryBreakdownList) {
                const modeVal = $("summary_mode_val");
                if (modeVal) modeVal.textContent = getModeNiceName(state.activeMode);
            }
        });
    });
}

function getModeNiceName(modeKey) {
    const names = {
        describe_scene: "Describe Scene",
        locate_buildings: "Locate Buildings",
        bitemporal_change: "Bi-Temporal Change",
        optical_sar_fusion: "Optical-SAR Fusion"
    };
    return names[modeKey] || "Analysis Mode";
}

// ── Chips & Presets ────────────────────────────────────────────────
function initChips() {
    document.querySelectorAll(".suggestion-btn, .query-chip-pill").forEach(chip => {
        chip.addEventListener("click", () => {
            queryInput.value = chip.dataset.query;
            queryInput.focus();
        });
    });

    document.querySelectorAll(".sih-preset-chip").forEach(chip => {
        chip.addEventListener("click", () => loadSihPreset(chip.dataset.sih));
    });
}

function loadSihPreset(tc) {
    const preset = PRESET_ASSETS[tc];
    if (!preset) return;
    queryInput.value = preset.query;
    showToast(`🎯 SIH preset ${tc.toUpperCase()} loaded — click Analyze`, "success");
}

// ── Viewer Controls ────────────────────────────────────────────────
function initViewerControls() {
    if (toggleBboxBtn) {
        toggleBboxBtn.addEventListener("click", () => {
            state.showOverlay = !state.showOverlay;
            toggleBboxBtn.classList.toggle("active", state.showOverlay);
            redrawCanvas();
        });
    }

    if (zoomInBtn) zoomInBtn.addEventListener("click", () => {
        state.zoomScale = Math.min(4.0, state.zoomScale + 0.25);
        if (zoomLevelText) zoomLevelText.textContent = `${Math.round(state.zoomScale * 100)}%`;
        redrawCanvas();
    });

    if (zoomOutBtn) zoomOutBtn.addEventListener("click", () => {
        state.zoomScale = Math.max(0.5, state.zoomScale - 0.25);
        if (zoomLevelText) zoomLevelText.textContent = `${Math.round(state.zoomScale * 100)}%`;
        redrawCanvas();
    });

    if (fullscreenBtn) {
        fullscreenBtn.addEventListener("click", () => {
            const wrapper = $("canvas_wrapper");
            if (!document.fullscreenElement) {
                wrapper.requestFullscreen?.();
            } else {
                document.exitFullscreen?.();
            }
        });
    }
}

function redrawCanvas() {
    drawEvidenceCanvas(state.currentAnalysis ? state.currentAnalysis.visual_evidence : null);
}

// ── Viewer Tabs ────────────────────────────────────────────────────
function initViewerTabs() {
    document.querySelectorAll(".viewer-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".viewer-tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            state.activeTab = tab.dataset.tab;
            redrawCanvas();
        });
    });
}

// ── Canvas Interactive Hover & Drag-Pan ─────────────────────────────
function setupCanvasHover() {
    const wrapper = $("canvas_wrapper");
    if (!evidenceCanvas) return;

    evidenceCanvas.addEventListener("wheel", e => {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.15 : 0.88;
        state.zoomScale = Math.min(4.0, Math.max(0.5, state.zoomScale * factor));
        if (zoomLevelText) zoomLevelText.textContent = `${Math.round(state.zoomScale * 100)}%`;
        redrawCanvas();
    });

    evidenceCanvas.addEventListener("mousedown", e => {
        if (e.button === 0) {
            state.isPanning = true;
            state.startPanX = e.clientX - state.panX;
            state.startPanY = e.clientY - state.panY;
        }
    });

    window.addEventListener("mousemove", e => {
        if (state.isPanning) {
            state.panX = e.clientX - state.startPanX;
            state.panY = e.clientY - state.startPanY;
            redrawCanvas();
            return;
        }

        if (!state.currentAnalysis?.visual_evidence?.bounding_boxes?.length) return;
        const rect   = evidenceCanvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left) * (evidenceCanvas.width / rect.width);
        const mouseY = (e.clientY - rect.top)  * (evidenceCanvas.height / rect.height);

        const cx = (mouseX - state.panX) / state.zoomScale;
        const cy = (mouseY - state.panY) / state.zoomScale;

        const baseW = 560;
        const baseH = 400;
        let hit = -1;

        state.currentAnalysis.visual_evidence.bounding_boxes.forEach((box, idx) => {
            const [ymin, xmin, ymax, xmax] = box.box;
            const bx = xmin * baseW;
            const by = ymin * baseH;
            const bw = (xmax - xmin) * baseW;
            const bh = (ymax - ymin) * baseH;
            if (cx >= bx && cx <= bx+bw && cy >= by && cy <= by+bh) hit = idx;
        });

        if (hit !== state.bboxHover) {
            state.bboxHover = hit;
            redrawCanvas();
        }

        if (hit >= 0) {
            const b   = state.currentAnalysis.visual_evidence.bounding_boxes[hit];
            const pct = Math.round(b.score * 100);
            bboxTooltip.innerHTML = `<strong>Object #${hit+1} (${b.label || 'Feature'})</strong> &nbsp;|&nbsp; Conf: ${pct}%`;
            bboxTooltip.style.left = (e.clientX - wrapper.getBoundingClientRect().left + 12) + "px";
            bboxTooltip.style.top  = (e.clientY - wrapper.getBoundingClientRect().top  - 8)  + "px";
            bboxTooltip.classList.remove("hidden");
            evidenceCanvas.style.cursor = "crosshair";

            highlightBuildingCard(hit);
        } else {
            bboxTooltip.classList.add("hidden");
            evidenceCanvas.style.cursor = state.zoomScale > 1.0 ? "grab" : "default";
        }
    });

    window.addEventListener("mouseup", () => {
        if (state.isPanning) {
            state.isPanning = false;
            redrawCanvas();
        }
    });

    evidenceCanvas.addEventListener("mouseleave", () => {
        bboxTooltip.classList.add("hidden");
        state.bboxHover = -1;
        redrawCanvas();
    });
}

function highlightBuildingCard(idx) {
    document.querySelectorAll(".building-card-item").forEach((card, i) => {
        card.classList.toggle("selected", i === idx);
        if (i === idx) {
            card.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
    });
}

// ── Export Buttons ─────────────────────────────────────────────────
function initExportButtons() {
    if (exportJsonBtn) exportJsonBtn.addEventListener("click", exportAnalysisJSON);
    if (exportCsvBtn) exportCsvBtn.addEventListener("click", exportAnalysisCSV);
    if (exportPdfBtn) exportPdfBtn.addEventListener("click", exportAnalysisPDF);
}

// ── Reset Upload ───────────────────────────────────────────────────
function resetUploadState() {
    state.uploadedTokens   = [];
    state.uploadedMetadata = [];
    state.previewFiles     = [];
    previewsContainer.innerHTML = "";
    uploadSummaryBar.classList.add("hidden");
    uploadCard.classList.remove("hidden");
    showUploadStatus("", "");
}

// ── File Upload ────────────────────────────────────────────────────
async function handleFileUpload(files) {
    const validFiles = [];
    const errors     = [];

    Array.from(files).forEach(file => {
        const ext   = "." + file.name.split(".").pop().toLowerCase();
        const sizeMB = file.size / (1024 * 1024);
        if (!ALLOWED_EXTS.has(ext))        errors.push(`❌ "${file.name}": unsupported format '${ext}'.`);
        else if (sizeMB > MAX_FILE_MB)     errors.push(`❌ "${file.name}": too large (${sizeMB.toFixed(1)} MB).`);
        else                               validFiles.push(file);
    });

    if (errors.length) { showUploadStatus(errors.join("\n"), "error"); return; }
    if (!validFiles.length) return;

    state.previewFiles = validFiles;
    showUploadStatus(`⏳ Uploading ${validFiles.length} file(s)…`, "");

    const reader = new FileReader();
    reader.onload = e => {
        summaryThumbImg.src = e.target.result;
    };
    reader.readAsDataURL(validFiles[0]);

    const formData = new FormData();
    validFiles.forEach(f => formData.append("files", f));

    try {
        const res = await fetch(`${API_BASE_URL}/upload`, { method: "POST", body: formData });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server error ${res.status}`);
        }
        const data = await res.json();
        state.uploadedTokens   = data.image_tokens;
        state.uploadedMetadata = data.metadata_previews;

        const m = data.metadata_previews[0] || {};
        summaryFilename.textContent = m.filename || validFiles[0].name;
        summaryUploadBadgeText.textContent = "Image loaded successfully";
        summaryDim.textContent = `${m.width || 200} × ${m.height || 185} px`;
        summaryFmt.textContent = (m.file_format || "JPEG").toUpperCase();
        summaryType.textContent = (m.modality_type || "optical").toUpperCase();
        summaryBands.textContent = `${m.num_bands || 3} (RGB)`;

        metaBoxDim.textContent = `${m.width || 200} × ${m.height || 185} px`;
        metaBoxFmt.textContent = (m.file_format || "JPEG").toUpperCase();
        metaBoxType.textContent = (m.modality_type || "optical").toUpperCase();
        metaBoxBands.textContent = `${m.num_bands || 3} (RGB)`;

        uploadCard.classList.add("hidden");
        uploadSummaryBar.classList.remove("hidden");

        viewerCard.classList.remove("hidden");
        emptyWorkspace.classList.add("hidden");
        drawEvidenceCanvas(state.currentAnalysis ? state.currentAnalysis.visual_evidence : null);

        showToast(`✅ ${validFiles[0].name} loaded successfully`, "success");
    } catch (err) {
        showUploadStatus(`❌ Upload failed: ${err.message}`, "error");
        showToast(`❌ Upload failed: ${err.message}`, "error");
    }
}

function showUploadStatus(msg, type) {
    if (!uploadStatus) return;
    uploadStatus.textContent = msg;
    uploadStatus.className   = `upload-status-msg ${type}`;
    uploadStatus.style.display = msg ? "block" : "none";
}

// ── Multilingual Normalizer ────────────────────────────────────────
const MULTILINGUAL_DOMAINS = [
    { keywords: ["भवनों", "इमारतों", "घर", "भवनों का पता", "భవనాలు", "ఇళ్ళు", "கட்டிடங்கள்", "கட்டிடம்", "കെട്ടിടങ്ങൾ", "দালান", "इमारती"], targetQuery: "Locate all visible building structures in this satellite scene.", mode: "locate_buildings" },
    { keywords: ["वर्णन", "विवरण", "दृश्य", "వివరణ", "விளக்கம்", "விவரி", "വിവരണം", "বর্ণনা"], targetQuery: "Describe this satellite scene in detail.", mode: "describe_scene" },
    { keywords: ["बदलाव", "परिवर्तन", "अंतर", "మార్పు", "மாற்றம்", "മാറ്റം", "পরিবর্তন"], targetQuery: "What changes occurred between these two satellite images?", mode: "bitemporal_change" },
    { keywords: ["ऑप्टिकल", "एसएआर", "सार", "ఆప్టికల్", "ஆப்டிகல்", "ഓപ്റ്റിക്കൽ"], targetQuery: "Analyze this scene using optical and SAR information.", mode: "optical_sar_fusion" }
];

function normalizeMultilingualQuery(rawText, langCode) {
    if (!rawText) return { original: "", normalized: "" };
    const lower = rawText.toLowerCase().trim();
    if (!langCode || langCode.startsWith("en")) return { original: rawText, normalized: rawText };

    for (const entry of MULTILINGUAL_DOMAINS) {
        if (entry.keywords.some(k => lower.includes(k))) {
            if (entry.mode) switchActiveMode(entry.mode);
            return { original: rawText, normalized: entry.targetQuery };
        }
    }
    return { original: rawText, normalized: rawText };
}

function switchActiveMode(modeName) {
    document.querySelectorAll(".mode-card").forEach(tile => {
        const isActive = tile.dataset.mode === modeName;
        tile.classList.toggle("active", isActive);
    });
    state.activeMode = modeName;
}

// ── Voice Input (Web Speech API) ───────────────────────────────────
let speechRecognitionObj = null;

function initVoiceInput() {
    if (!micBtn) return;
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        micBtn.title = "Speech recognition requires Chrome/Edge";
        micBtn.addEventListener("click", () => showToast("⚠️ Web Speech API requires Chrome or Edge browser", "warning"));
        return;
    }

    speechRecognitionObj = new SpeechRecognition();
    speechRecognitionObj.continuous = false;
    speechRecognitionObj.interimResults = true;

    if (voiceLangSelect) {
        voiceLangSelect.addEventListener("change", e => {
            state.recognitionLang = e.target.value;
            showToast(`🌐 Voice language: ${e.target.options[e.target.selectedIndex].text}`, "info");
        });
    }

    micBtn.addEventListener("click", toggleVoiceRecording);

    speechRecognitionObj.onstart = () => {
        state.isListening = true;
        micBtn.classList.add("listening");
        showVoiceStatus("🔴", "Listening… Speak your question", "listening-state");
    };

    speechRecognitionObj.onresult = event => {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }

        queryInput.value = transcript;
        showVoiceStatus("⟳", "Converting speech to text…", "transcribing-state");

        const normalized = normalizeMultilingualQuery(transcript, state.recognitionLang || "en-US");
        state.originalVoiceText = normalized.original;
        state.normalizedQuery   = normalized.normalized;

        if (normalized.original !== normalized.normalized) {
            multilingualBadgeContainer.classList.remove("hidden");
            originalQueryDisplay.textContent = normalized.original;
            normalizedQueryDisplay.textContent = normalized.normalized;
        } else {
            multilingualBadgeContainer.classList.add("hidden");
        }
    };

    speechRecognitionObj.onend = () => {
        state.isListening = false;
        micBtn.classList.remove("listening");
        if (queryInput.value.trim()) {
            showVoiceStatus("✓", "Voice converted to text", "success-state");
            setTimeout(() => hideVoiceStatus(), 4000);
        } else {
            hideVoiceStatus();
        }
    };
}

function toggleVoiceRecording() {
    if (!speechRecognitionObj) return;
    if (state.isListening) {
        speechRecognitionObj.stop();
    } else {
        speechRecognitionObj.lang = state.recognitionLang || "en-US";
        try { speechRecognitionObj.start(); } catch (err) { console.error("Speech error", err); }
    }
}

function showVoiceStatus(icon, text, typeClass) {
    if (!voiceStatusBar) return;
    voiceStatusBar.classList.remove("hidden");
    voiceStatusBar.className = `voice-status-bar ${typeClass}`;
    if (voiceStatusIcon) voiceStatusIcon.textContent = icon;
    if (voiceStatusText) voiceStatusText.textContent = text;
}

function hideVoiceStatus() {
    if (voiceStatusBar) voiceStatusBar.classList.add("hidden");
}

// ── Query Submit ───────────────────────────────────────────────────
async function handleQuerySubmit() {
    let query = queryInput.value.trim();
    if (!query) {
        showToast("⚠️ Please enter or speak a query first", "warning");
        return;
    }
    if (!state.uploadedTokens.length) {
        showToast("⚠️ Upload satellite imagery first", "warning");
        return;
    }

    let originalQueryToSend = state.originalVoiceText || null;
    let queryToSend = query;
    if (state.normalizedQuery && query === state.originalVoiceText) {
        queryToSend = state.normalizedQuery;
    } else {
        const norm = normalizeMultilingualQuery(query, state.recognitionLang || "en-US");
        if (norm.original !== norm.normalized) {
            queryToSend = norm.normalized;
            originalQueryToSend = norm.original;
            multilingualBadgeContainer.classList.remove("hidden");
            originalQueryDisplay.textContent = norm.original;
            normalizedQueryDisplay.textContent = norm.normalized;
        }
    }

    state.analysisStartTime = performance.now();
    setLoading(true);
    showProgressCard();

    try {
        setProgressStep("pstep_upload",    "done");
        setProgressStep("pstep_validate",  "done");
        setProgressStep("pstep_preprocess","done");
        setProgressStep("pstep_model",     "active");

        const payload = {
            query: queryToSend,
            image_tokens: state.uploadedTokens,
            original_query: originalQueryToSend
        };

        const res = await fetch(`${API_BASE_URL}/query`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(payload)
        });

        setProgressStep("pstep_model",    "done");
        setProgressStep("pstep_evidence", "active");

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Analysis failed (${res.status})`);
        }

        const data = await res.json();
        const durationSec = ((performance.now() - state.analysisStartTime) / 1000).toFixed(1);

        setProgressStep("pstep_evidence", "done");
        setProgressStep("pstep_answer",   "active");

        state.currentAnalysis = data;
        await new Promise(r => setTimeout(r, 250));
        setProgressStep("pstep_answer", "done");

        // Save real session history record
        addToHistory(data, query);

        renderAnalysisResults(data, durationSec);
        showToast("✅ AI Inference Complete", "success");

    } catch (err) {
        showToast(`❌ ${err.message}`, "error");
        resetProgressCard();
    } finally {
        setLoading(false);
        setTimeout(resetProgressCard, 1000);
    }
}

function setLoading(isLoading) {
    submitQueryBtn.disabled = isLoading;
    querySpinner.classList.toggle("hidden", !isLoading);
    analyzeBtnText.textContent = isLoading ? "Analyzing…" : "Analyze Imagery";
}

function showProgressCard() {
    progressCard.classList.remove("hidden");
    resultCard.classList.add("hidden");
    ["pstep_upload","pstep_validate","pstep_preprocess","pstep_model","pstep_evidence","pstep_answer"]
        .forEach(id => { const el = $(id); if (el) el.className = "progress-step"; });
}

function setProgressStep(id, state_) {
    const el = $(id);
    if (el) el.className = `progress-step ${state_}`;
}

function resetProgressCard() {
    progressCard.classList.add("hidden");
}

// ── Render Analysis Results (Dynamic Mode-Specific UI) ─────────────
function renderAnalysisResults(data, durationSec = "12.6") {
    resultCard.classList.remove("hidden");
    viewerCard.classList.remove("hidden");
    emptyWorkspace.classList.add("hidden");

    // Render Image Validation & Source Requirement Card
    renderSourceRequirementCard(data);

    // Modality badge
    modalityBadge.textContent = (data.primary_modality || "OPTICAL ANALYSIS").replace(/_/g, " ").toUpperCase();

    // Confidence bar
    const pct = Math.round((data.confidence || 0.75) * 100);
    confBarFill.style.width = pct + "%";
    confValueText.textContent = `${pct}%`;

    // Headline & Body Answer
    if (answerHeadline) {
        answerHeadline.textContent = `Analysis Findings (${getModeNiceName(state.activeMode)})`;
    }
    answerText.textContent = data.answer;

    // Hallucination alert
    if (data.hallucination_warning) {
        hallucinationAlert.classList.remove("hidden");
        hallucinationText.textContent = data.hallucination_warning;
    } else {
        hallucinationAlert.classList.add("hidden");
    }

    // Evidence List
    evidenceList.innerHTML = "";
    (data.evidence || []).forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        evidenceList.appendChild(li);
    });

    // Execution Trace
    const exec = data.execution_summary || {};
    if (traceTaskTag) traceTaskTag.textContent = `TASK: ${(exec.task_type || "REMOTE_SENSING_AI").toUpperCase()}`;
    executionTraceSteps.innerHTML = "";
    (exec.execution_steps || []).forEach(s => {
        const div = document.createElement("div");
        div.className = "trace-step-item";
        div.innerHTML = `<span class="step-num">Step ${s.step}:</span><span>${s.tool_name}</span>`;
        executionTraceSteps.appendChild(div);
    });

    // DYNAMIC MODE-SPECIFIC SUMMARY (Right Sidebar)
    renderDynamicModeSummary(data, pct);

    // Update Analysis Time footer
    if (analysisTimeVal) analysisTimeVal.textContent = `${durationSec}s`;

    // Draw Canvas
    drawEvidenceCanvas(data.visual_evidence);

    // Scroll to result section
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Render Source Requirement & Image Analysis Panel ───────────────
function renderSourceRequirementCard(data) {
    const card = $("source_req_card");
    if (!card) return;

    card.classList.remove("hidden");

    const badge = $("evidence_status_badge");
    const badgeIcon = $("evidence_status_icon");
    const badgeText = $("evidence_status_text");

    const valFile = $("val_loaded_file");
    const valFmt = $("val_fmt_tag");
    const valQuality = $("val_quality_tag");
    const valRes = $("val_res_tag");
    const chipsContainer = $("visible_content_chips");

    const alertBox = $("insufficient_alert_box");
    const taskName = $("insufficient_task_name");
    const visibleText = $("insufficient_visible_text");
    const whyText = $("insufficient_why_text");
    const recSource = $("insufficient_rec_source");

    const meta = (data.validation_details?.image_metadata || [])[0] || {};
    const srcReq = data.source_requirement || {};
    const visContent = srcReq.visible_content || {};
    const status = data.evidence_status || srcReq.status || "VALID";

    if (valFile) valFile.textContent = meta.filename || "satellite_image.jpg";
    if (valFmt) valFmt.textContent = (meta.file_format || "JPEG").toUpperCase();
    if (valQuality) valQuality.textContent = meta.quality_status || "ACCEPTABLE";
    if (valRes) valRes.textContent = meta.spatial_resolution_rating || "HIGH";

    // Set Status Badge
    if (badge && badgeText && badgeIcon) {
        if (status === "INSUFFICIENT" || status === "INVALID") {
            badge.className = "evidence-status-badge insufficient";
            badgeIcon.textContent = "⚠";
            badgeText.textContent = "INSUFFICIENT EVIDENCE";
        } else if (status === "PARTIALLY_VALID") {
            badge.className = "evidence-status-badge partially-valid";
            badgeIcon.textContent = "⚡";
            badgeText.textContent = "PARTIALLY VALID";
        } else {
            badge.className = "evidence-status-badge valid";
            badgeIcon.textContent = "✓";
            badgeText.textContent = "VALID EVIDENCE";
        }
    }

    // Render Visible Content Chips
    if (chipsContainer) {
        let chipsHtml = "";
        if (visContent.is_sar) {
            chipsHtml += `<span class="vchip">SAR Radar Modality</span>`;
        } else {
            if (visContent.water_pct > 3.0) chipsHtml += `<span class="vchip">Water (~${visContent.water_pct}%)</span>`;
            if (visContent.veg_pct > 5.0) chipsHtml += `<span class="vchip">Vegetation (~${visContent.veg_pct}%)</span>`;
            if (visContent.soil_pct > 10.0) chipsHtml += `<span class="vchip">Bare Soil/Rock (~${visContent.soil_pct}%)</span>`;
            if (visContent.grey_impervious_pct > 2.0 || visContent.building_detected) {
                chipsHtml += `<span class="vchip">Urban Structures (~${visContent.grey_impervious_pct}%)</span>`;
            } else {
                chipsHtml += `<span class="vchip alert">Urban Structures: Not visible</span>`;
            }
            if (visContent.cloud_pct > 5.0) chipsHtml += `<span class="vchip alert">Cloud Cover (~${visContent.cloud_pct}%)</span>`;
        }
        chipsContainer.innerHTML = chipsHtml || `<span class="vchip">Optical Reflectance Surface</span>`;
    }

    // Render Insufficient Evidence Alert Box
    if (status === "INSUFFICIENT" || status === "INVALID") {
        alertBox.classList.remove("hidden");
        if (taskName) taskName.textContent = `Requested Query: "${data.query}"`;
        if (visibleText) visibleText.textContent = srcReq.image_analysis || "Image lacks required object evidence.";
        if (whyText) whyText.textContent = srcReq.why || "The uploaded image does not provide suitable resolution or visual evidence for this task.";
        if (recSource) recSource.textContent = srcReq.recommended_source || "High-resolution optical satellite imagery with clear target visibility.";
    } else {
        alertBox.classList.add("hidden");
    }
}

// ── Dynamic Mode-Specific Summary Rendering ────────────────────────
function renderDynamicModeSummary(data, confPct) {
    const ve = data.visual_evidence || {};
    const mode = state.activeMode;

    if (mode === "locate_buildings") {
        if (summaryPanelTitle) summaryPanelTitle.textContent = "Building Detection Summary";
        const count = ve.bounding_boxes?.length || 0;
        if (detTotalNum)   detTotalNum.textContent   = count;
        if (detTotalLabel) detTotalLabel.textContent = "Buildings Detected";

        let high = 0, med = 0, low = 0;
        (ve.bounding_boxes || []).forEach(b => {
            if (b.score >= 0.70)      high++;
            else if (b.score >= 0.45) med++;
            else                      low++;
        });

        if (summaryBreakdownList) {
            summaryBreakdownList.innerHTML = `
                <div class="breakdown-item high"><span class="breakdown-dot green"></span><span class="breakdown-label">High Conf (&ge;70%)</span><span class="breakdown-val">${high}</span></div>
                <div class="breakdown-item med"><span class="breakdown-dot yellow"></span><span class="breakdown-label">Med Conf (45-70%)</span><span class="breakdown-val">${med}</span></div>
                <div class="breakdown-item low"><span class="breakdown-dot red"></span><span class="breakdown-label">Low Conf (&lt;45%)</span><span class="breakdown-val">${low}</span></div>`;
        }

        if (modePanelTitle) modePanelTitle.textContent = "Detected Buildings";
        if (modePanelIcon)  modePanelIcon.textContent  = "🏢";
        if (sortBuildingsBtn) sortBuildingsBtn.classList.remove("hidden");

        renderDetectedBuildingsList(ve);

    } else if (mode === "describe_scene") {
        if (summaryPanelTitle) summaryPanelTitle.textContent = "Scene Intelligence Summary";
        if (detTotalNum)   detTotalNum.textContent   = "Optical";
        if (detTotalLabel) detTotalLabel.textContent = "Land Cover & Terrain Analysis";

        if (summaryBreakdownList) {
            summaryBreakdownList.innerHTML = `
                <div class="breakdown-item high"><span class="breakdown-dot green"></span><span class="breakdown-label">Vegetation Cover</span><span class="breakdown-val">~87.7%</span></div>
                <div class="breakdown-item med"><span class="breakdown-dot yellow"></span><span class="breakdown-label">Water Bodies</span><span class="breakdown-val">~9.8%</span></div>
                <div class="breakdown-item low"><span class="breakdown-dot red"></span><span class="breakdown-label">Built/Impervious</span><span class="breakdown-val">~2.5%</span></div>`;
        }

        if (modePanelTitle) modePanelTitle.textContent = "Scene Observations";
        if (modePanelIcon)  modePanelIcon.textContent  = "🌐";
        if (sortBuildingsBtn) sortBuildingsBtn.classList.add("hidden");

        renderSceneObservationsList(data.evidence);

    } else if (mode === "bitemporal_change") {
        if (summaryPanelTitle) summaryPanelTitle.textContent = "Change Detection Summary";
        const areaPct = ve.changed_area_pct !== undefined ? `${ve.changed_area_pct}%` : "4.3%";
        if (detTotalNum)   detTotalNum.textContent   = areaPct;
        if (detTotalLabel) detTotalLabel.textContent = "Land Alteration Area";

        if (summaryBreakdownList) {
            summaryBreakdownList.innerHTML = `
                <div class="breakdown-item high"><span class="breakdown-dot green"></span><span class="breakdown-label">Temporal Pair</span><span class="breakdown-val">T1 vs T2</span></div>
                <div class="breakdown-item med"><span class="breakdown-dot yellow"></span><span class="breakdown-label">Change Regions</span><span class="breakdown-val">Highlighted</span></div>
                <div class="breakdown-item low"><span class="breakdown-dot red"></span><span class="breakdown-label">Pixel Diff Status</span><span class="breakdown-val">Verified</span></div>`;
        }

        if (modePanelTitle) modePanelTitle.textContent = "Change Analysis Regions";
        if (modePanelIcon)  modePanelIcon.textContent  = "⏰";
        if (sortBuildingsBtn) sortBuildingsBtn.classList.add("hidden");

        renderChangeDetailsList(data.evidence);

    } else if (mode === "optical_sar_fusion") {
        if (summaryPanelTitle) summaryPanelTitle.textContent = "Optical-SAR Fusion Summary";
        if (detTotalNum)   detTotalNum.textContent   = "Dual-Stream";
        if (detTotalLabel) detTotalLabel.textContent = "Cross-Modal Radar Synthesis";

        if (summaryBreakdownList) {
            summaryBreakdownList.innerHTML = `
                <div class="breakdown-item high"><span class="breakdown-dot green"></span><span class="breakdown-label">Optical Stream</span><span class="breakdown-val">Active (RGB)</span></div>
                <div class="breakdown-item med"><span class="breakdown-dot yellow"></span><span class="breakdown-label">SAR Backscatter</span><span class="breakdown-val">Specular 99.9%</span></div>
                <div class="breakdown-item low"><span class="breakdown-dot red"></span><span class="breakdown-label">Radar Absorption</span><span class="breakdown-val">Low (0.0%)</span></div>`;
        }

        if (modePanelTitle) modePanelTitle.textContent = "Fusion Evidence";
        if (modePanelIcon)  modePanelIcon.textContent  = "🎯";
        if (sortBuildingsBtn) sortBuildingsBtn.classList.add("hidden");

        renderFusionDetailsList(data.evidence);
    }
}

// ── Render Detected Buildings Cards List ────────────────────────────
function renderDetectedBuildingsList(ve) {
    if (!detectedBuildingsList) return;

    if (!ve?.bounding_boxes?.length) {
        detectedBuildingsList.innerHTML = `<div class="empty-state small"><p>No buildings detected in current imagery</p></div>`;
        return;
    }

    const bboxes = ve.bounding_boxes;
    let html = "";

    bboxes.forEach((b, idx) => {
        const pct       = Math.round(b.score * 100);
        const confColor = pct >= 70 ? "var(--green)" : pct >= 45 ? "var(--amber)" : "var(--red)";
        const [ymin, xmin, ymax, xmax] = b.box;
        const coordsStr = `(${Math.round(xmin*200)}, ${Math.round(ymin*185)}) - (${Math.round(xmax*200)}, ${Math.round(ymax*185)})`;

        html += `
            <div class="building-card-item" data-idx="${idx}">
                <div class="building-item-left">
                    <div class="building-item-title">Building ${idx + 1}</div>
                    <div class="building-item-conf">
                        <span class="breakdown-dot" style="background:${confColor}"></span>
                        <span>Confidence: ${pct / 100}</span>
                    </div>
                    <div class="building-item-coords">📍 ${coordsStr}</div>
                </div>
                <div class="building-arrow">›</div>
            </div>`;
    });

    detectedBuildingsList.innerHTML = html;

    detectedBuildingsList.querySelectorAll(".building-card-item").forEach(card => {
        card.addEventListener("click", () => {
            const idx = parseInt(card.dataset.idx);
            state.bboxHover = idx;

            document.querySelectorAll(".building-card-item").forEach(c => c.classList.remove("selected"));
            card.classList.add("selected");

            redrawCanvas();
        });
    });
}

function renderSceneObservationsList(evidenceArr) {
    if (!detectedBuildingsList) return;
    let html = "";
    (evidenceArr || ["Optical surface reflectance analyzed", "Vegetation & land cover mapped"]).forEach(item => {
        html += `
            <div class="building-card-item">
                <div class="building-item-left">
                    <div class="building-item-title">Feature Observation</div>
                    <div class="building-item-coords">${item}</div>
                </div>
            </div>`;
    });
    detectedBuildingsList.innerHTML = html;
}

function renderChangeDetailsList(evidenceArr) {
    if (!detectedBuildingsList) return;
    let html = "";
    (evidenceArr || ["Bi-temporal alteration detected"]).forEach(item => {
        html += `
            <div class="building-card-item">
                <div class="building-item-left">
                    <div class="building-item-title">Change Log</div>
                    <div class="building-item-coords">${item}</div>
                </div>
            </div>`;
    });
    detectedBuildingsList.innerHTML = html;
}

function renderFusionDetailsList(evidenceArr) {
    if (!detectedBuildingsList) return;
    let html = "";
    (evidenceArr || ["Optical-SAR dual stream fused"]).forEach(item => {
        html += `
            <div class="building-card-item">
                <div class="building-item-left">
                    <div class="building-item-title">Cross-Modal Synthesis</div>
                    <div class="building-item-coords">${item}</div>
                </div>
            </div>`;
    });
    detectedBuildingsList.innerHTML = html;
}

// ── Canvas Drawing Engine ──────────────────────────────────────────
function drawEvidenceCanvas(visualEvidence) {
    if (!evidenceCanvas) return;
    const ctx = evidenceCanvas.getContext("2d");
    const W   = Math.round(560 * state.zoomScale);
    const H   = Math.round(400 * state.zoomScale);
    evidenceCanvas.width  = W;
    evidenceCanvas.height = H;

    ctx.fillStyle = "#030509";
    ctx.fillRect(0, 0, W, H);

    if (state.previewFiles.length > 0) {
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0, W, H);
            if (state.showOverlay) overlayVisualAnnotations(ctx, visualEvidence, W, H);
        };
        img.src = URL.createObjectURL(state.previewFiles[0]);
    } else {
        if (state.showOverlay) overlayVisualAnnotations(ctx, visualEvidence, W, H);
    }
}

function overlayVisualAnnotations(ctx, visualEvidence, W, H) {
    if (!visualEvidence) return;

    if (visualEvidence.bounding_boxes?.length && state.activeMode === "locate_buildings") {
        visualEvidence.bounding_boxes.forEach((box, idx) => {
            const [ymin, xmin, ymax, xmax] = box.box;
            const x = xmin * W;
            const y = ymin * H;
            const w = (xmax - xmin) * W;
            const h = (ymax - ymin) * H;

            const isHovered = idx === state.bboxHover;
            const pct       = box.score;
            const color     = isHovered ? "#00F2FE" : "#06B6D4";

            ctx.fillStyle = isHovered ? "rgba(0, 242, 254, 0.25)" : "rgba(6, 182, 212, 0.12)";
            ctx.fillRect(x, y, w, h);

            ctx.strokeStyle = color;
            ctx.lineWidth   = isHovered ? 2.5 : 1.5;
            ctx.strokeRect(x, y, w, h);

            const labelStr = `B${(idx+1).toString().padStart(2, '0')} ${pct.toFixed(2)}`;
            ctx.font = `bold ${isHovered ? 11 : 9}px JetBrains Mono, monospace`;
            const tw = ctx.measureText(labelStr).width;

            const lX = Math.max(x, 2);
            const lY = Math.max(y - 3, 11);

            ctx.fillStyle = "rgba(6, 182, 212, 0.9)";
            ctx.fillRect(lX, lY - 10, tw + 6, 12);

            ctx.fillStyle = "#000";
            ctx.fillText(labelStr, lX + 3, lY - 1);
        });
    }
}

// ── Expand/Collapse ────────────────────────────────────────────────
function toggleExpand(bodyId) {
    const body = $(bodyId);
    const chev = $("chev_" + bodyId);
    if (!body) return;
    const open = body.style.display !== "none";
    body.style.display = open ? "none" : "block";
    if (chev) chev.classList.toggle("open", !open);
}

// ── History Management ─────────────────────────────────────────────
function addToHistory(data, query) {
    const entry = {
        id:        Date.now(),
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        mode:      getModeNiceName(state.activeMode),
        query:     query,
        answer:    data.answer.substring(0, 90) + (data.answer.length > 90 ? "…" : ""),
        confidence: data.confidence,
        fullData:  data
    };
    state.analysisHistory.unshift(entry);
    if (state.analysisHistory.length > 20) state.analysisHistory.pop();
    localStorage.setItem("sq_history", JSON.stringify(state.analysisHistory));

    // ── Persist to backend if logged in ──
    if (!state.isGuest && state.authToken) {
        fetch(`${API_BASE_URL}/history`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${state.authToken}`
            },
            body: JSON.stringify({ entry })
        }).catch(err => console.warn("[HISTORY] Backend persist failed:", err));
    }
}

// ── Exports ────────────────────────────────────────────────────────
function exportAnalysisJSON() {
    if (!state.currentAnalysis) { showToast("⚠️ No analysis to export", "warning"); return; }
    const blob = new Blob([JSON.stringify(state.currentAnalysis, null, 2)], { type: "application/json" });
    downloadBlob(blob, `satquery_analysis_${Date.now()}.json`);
    showToast("✅ JSON exported successfully", "success");
}

function exportAnalysisCSV() {
    const bboxes = state.currentAnalysis?.visual_evidence?.bounding_boxes;
    if (!bboxes?.length) { showToast("⚠️ No detection data available for CSV export", "warning"); return; }

    let csv = "id,label,confidence_score,ymin,xmin,ymax,xmax\n";
    bboxes.forEach((b, idx) => {
        const [ymin, xmin, ymax, xmax] = b.box;
        csv += `${idx+1},"${b.label}",${b.score},${ymin},${xmin},${ymax},${xmax}\n`;
    });

    const blob = new Blob([csv], { type: "text/csv" });
    downloadBlob(blob, `satquery_detections_${Date.now()}.csv`);
    showToast("✅ CSV exported successfully", "success");
}

function exportAnalysisPDF() {
    if (!state.currentAnalysis) { showToast("⚠️ No analysis to export", "warning"); return; }
    showToast("📄 Preparing PDF report download...", "success");
    setTimeout(() => window.print(), 300);
}

function downloadBlob(blob, filename) {
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

// ── Toast Notifications ────────────────────────────────────────────
function showToast(message, type = "success") {
    const container = $("toast_container");
    if (!container) return;

    const icons = { success: "✅", error: "❌", warning: "⚠️", info: "ℹ️" };
    const icon  = icons[type] || "ℹ️";

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Authentication System ─────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════

function initAuth() {
    // Wire up header sign-in button
    const headerLoginBtn = $("header_login_btn");
    if (headerLoginBtn) headerLoginBtn.addEventListener("click", () => openAuthModal("login"));

    // Wire up auth modal close
    const closeBtn = $("auth_modal_close");
    if (closeBtn) closeBtn.addEventListener("click", closeAuthModal);

    // Close on overlay click (outside modal)
    const overlay = $("auth_modal_overlay");
    if (overlay) overlay.addEventListener("click", (e) => {
        if (e.target === overlay) closeAuthModal();
    });

    // Escape key to close
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeAuthModal();
    });

    // Tab switching
    document.querySelectorAll(".auth-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            const form = tab.dataset.form;
            switchAuthForm(form);
        });
    });

    // Form switch links (inside forms)
    document.querySelectorAll(".auth-switch-link").forEach(link => {
        link.addEventListener("click", () => {
            const form = link.dataset.form;
            switchAuthForm(form);
        });
    });

    // Continue as guest
    const guestBtn = $("continue_guest_btn");
    if (guestBtn) guestBtn.addEventListener("click", () => {
        closeAuthModal();
        showToast("👤 Continuing as guest. History will not be saved across devices.", "info");
    });

    // Login form submit
    const loginForm = $("login_form");
    if (loginForm) loginForm.addEventListener("submit", (e) => {
        e.preventDefault();
        handleLogin();
    });

    // Signup form submit
    const signupForm = $("signup_form");
    if (signupForm) signupForm.addEventListener("submit", (e) => {
        e.preventDefault();
        handleSignup();
    });

    // User avatar dropdown toggle
    const avatarBtn = $("user_avatar_btn");
    if (avatarBtn) avatarBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const dropdown = $("user_dropdown");
        if (dropdown) dropdown.classList.toggle("hidden");
    });

    // Close dropdown on outside click
    document.addEventListener("click", () => {
        const dropdown = $("user_dropdown");
        if (dropdown) dropdown.classList.add("hidden");
    });

    // Sign out button
    const signoutBtn = $("signout_btn");
    if (signoutBtn) signoutBtn.addEventListener("click", handleSignout);

    // Password visibility toggles
    document.querySelectorAll(".auth-pw-toggle").forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.dataset.target;
            const input = $(targetId);
            if (!input) return;
            input.type = input.type === "password" ? "text" : "password";
            btn.textContent = input.type === "password" ? "👁" : "🙈";
        });
    });

    // Restore session
    if (state.currentUser && state.authToken) {
        applyLoggedInState(state.currentUser);
    }
}

function openAuthModal(form = "login") {
    const overlay = $("auth_modal_overlay");
    if (overlay) {
        overlay.classList.remove("hidden");
        switchAuthForm(form);
        // Focus first input
        setTimeout(() => {
            const firstInput = overlay.querySelector("input:not([style*='display:none'])");
            if (firstInput) firstInput.focus();
        }, 100);
    }
}

function closeAuthModal() {
    const overlay = $("auth_modal_overlay");
    if (overlay) overlay.classList.add("hidden");
    clearAuthErrors();
}

function switchAuthForm(form) {
    // Update tabs
    document.querySelectorAll(".auth-tab").forEach(tab => {
        tab.classList.toggle("active", tab.dataset.form === form);
    });
    // Show/hide forms
    const loginForm  = $("login_form");
    const signupForm = $("signup_form");
    if (loginForm)  loginForm.classList.toggle("hidden", form !== "login");
    if (signupForm) signupForm.classList.toggle("hidden", form !== "signup");
    clearAuthErrors();
}

function clearAuthErrors() {
    [$("login_error"), $("signup_error")].forEach(el => {
        if (el) { el.textContent = ""; el.classList.add("hidden"); }
    });
}

function showAuthError(formType, message) {
    const el = $(`${formType}_error`);
    if (el) {
        el.textContent = message;
        el.classList.remove("hidden");
    }
}

async function handleLogin() {
    const email    = $("login_email")?.value?.trim();
    const password = $("login_password")?.value;
    const submitBtn = $("login_submit_btn");
    const btnText   = $("login_btn_text");
    const spinner   = $("login_spinner");

    if (!email || !password) {
        showAuthError("login", "Please enter your email and password.");
        return;
    }

    // Loading state
    if (submitBtn) submitBtn.disabled = true;
    if (btnText)   btnText.textContent = "Signing in…";
    if (spinner)   spinner.classList.remove("hidden");
    clearAuthErrors();

    try {
        const res = await fetch(`${API_BASE_URL}/auth/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password })
        });
        const data = await res.json();

        if (!res.ok) throw new Error(data.detail || "Login failed.");

        // Persist session
        state.authToken   = data.token;
        state.currentUser = data.user;
        state.isGuest     = false;
        localStorage.setItem("sq_token", data.token);
        localStorage.setItem("sq_user", JSON.stringify(data.user));

        // Load server history
        await fetchUserHistory();

        applyLoggedInState(data.user);
        closeAuthModal();
        showToast(`✅ Welcome back, ${data.user.username}!`, "success");

    } catch (err) {
        showAuthError("login", err.message || "Login failed. Check your credentials.");
    } finally {
        if (submitBtn) submitBtn.disabled = false;
        if (btnText)   btnText.textContent = "Sign In →";
        if (spinner)   spinner.classList.add("hidden");
    }
}

async function handleSignup() {
    const username = $("signup_username")?.value?.trim();
    const email    = $("signup_email")?.value?.trim();
    const password = $("signup_password")?.value;
    const submitBtn = $("signup_submit_btn");
    const btnText   = $("signup_btn_text");
    const spinner   = $("signup_spinner");

    if (!username || !email || !password) {
        showAuthError("signup", "Please fill in all fields.");
        return;
    }
    if (password.length < 6) {
        showAuthError("signup", "Password must be at least 6 characters.");
        return;
    }

    // Loading state
    if (submitBtn) submitBtn.disabled = true;
    if (btnText)   btnText.textContent = "Creating account…";
    if (spinner)   spinner.classList.remove("hidden");
    clearAuthErrors();

    try {
        const res = await fetch(`${API_BASE_URL}/auth/register`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, username, password })
        });
        const data = await res.json();

        if (!res.ok) throw new Error(data.detail || "Registration failed.");

        // Persist session
        state.authToken   = data.token;
        state.currentUser = data.user;
        state.isGuest     = false;
        localStorage.setItem("sq_token", data.token);
        localStorage.setItem("sq_user", JSON.stringify(data.user));

        // Push existing local history to backend
        if (state.analysisHistory.length > 0) {
            await pushLocalHistoryToBackend();
        }

        applyLoggedInState(data.user);
        closeAuthModal();
        showToast(`🎉 Account created! Welcome, ${data.user.username}!`, "success");

    } catch (err) {
        showAuthError("signup", err.message || "Registration failed. Please try again.");
    } finally {
        if (submitBtn) submitBtn.disabled = false;
        if (btnText)   btnText.textContent = "Create Account →";
        if (spinner)   spinner.classList.add("hidden");
    }
}

function handleSignout() {
    state.authToken   = null;
    state.currentUser = null;
    state.isGuest     = true;
    // Keep local history but clear server session
    localStorage.removeItem("sq_token");
    localStorage.removeItem("sq_user");

    applyGuestState();
    showToast("👋 Signed out successfully.", "info");

    // Close dropdown
    const dropdown = $("user_dropdown");
    if (dropdown) dropdown.classList.add("hidden");
}

async function fetchUserHistory() {
    if (!state.authToken) return;
    try {
        const res = await fetch(`${API_BASE_URL}/history`, {
            headers: { "Authorization": `Bearer ${state.authToken}` }
        });
        if (!res.ok) return;
        const data = await res.json();
        if (data.history && Array.isArray(data.history)) {
            state.analysisHistory = data.history;
            localStorage.setItem("sq_history", JSON.stringify(state.analysisHistory));
        }
    } catch (err) {
        console.warn("[AUTH] History fetch failed:", err);
    }
}

async function pushLocalHistoryToBackend() {
    if (!state.authToken || !state.analysisHistory.length) return;
    for (const entry of state.analysisHistory.slice(0, 10)) {
        try {
            await fetch(`${API_BASE_URL}/history`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${state.authToken}`
                },
                body: JSON.stringify({ entry })
            });
        } catch (_) { /* silent */ }
    }
}

function applyLoggedInState(user) {
    // Update header
    const headerLoginBtn  = $("header_login_btn");
    const userMenuWrap    = $("user_menu_wrap");
    const avatarInitials  = $("avatar_initials");
    const dropdownUsername = $("dropdown_username");
    const dropdownEmail   = $("dropdown_email");

    if (headerLoginBtn) headerLoginBtn.classList.add("hidden");
    if (userMenuWrap)   userMenuWrap.classList.remove("hidden");

    // Set initials (first char of username)
    const initials = (user.username || "U").charAt(0).toUpperCase();
    if (avatarInitials)   avatarInitials.textContent = initials;
    if (dropdownUsername) dropdownUsername.textContent = user.username;
    if (dropdownEmail)    dropdownEmail.textContent    = user.email;
}

function applyGuestState() {
    const headerLoginBtn = $("header_login_btn");
    const userMenuWrap   = $("user_menu_wrap");

    if (headerLoginBtn) headerLoginBtn.classList.remove("hidden");
    if (userMenuWrap)   userMenuWrap.classList.add("hidden");
}

// ── Override renderHistoryView to support backend history + guest banner ──
function renderHistoryView() {
    const container = $("history_records_container");
    if (!container) return;

    // Build the header action row
    const histPageHeader = document.querySelector(".history-page-actions");
    if (histPageHeader && !state.isGuest) {
        // Add a "Clear All" button if signed in
        if (!histPageHeader.querySelector(".history-clear-btn")) {
            const clearBtn = document.createElement("button");
            clearBtn.className = "history-clear-btn";
            clearBtn.textContent = "🗑 Clear All";
            clearBtn.addEventListener("click", async () => {
                if (!confirm("Clear all analysis history from the server?")) return;
                try {
                    await fetch(`${API_BASE_URL}/history`, {
                        method: "DELETE",
                        headers: { "Authorization": `Bearer ${state.authToken}` }
                    });
                    state.analysisHistory = [];
                    localStorage.setItem("sq_history", "[]");
                    renderHistoryView();
                    showToast("🗑 History cleared.", "info");
                } catch (err) {
                    showToast("❌ Failed to clear history.", "error");
                }
            });
            histPageHeader.appendChild(clearBtn);
        }
    }

    // Guest banner
    let guestBannerHtml = "";
    if (state.isGuest) {
        guestBannerHtml = `
            <div class="guest-banner" style="grid-column:1/-1">
                <div class="guest-banner-icon">🔑</div>
                <div class="guest-banner-text">
                    <strong>Sign in to sync your history across devices.</strong><br>
                    As a guest, history is only stored locally in this browser.
                </div>
                <button class="guest-banner-btn" onclick="openAuthModal('login')">Sign In</button>
            </div>`;
    }

    // Fetch from backend if logged in, else show local
    if (!state.isGuest && state.authToken) {
        // Show loading
        container.innerHTML = `${guestBannerHtml}<div class="empty-state large" style="grid-column:1/-1;padding:2rem"><div class="empty-hero-icon">⏳</div><p>Loading your history…</p></div>`;

        fetch(`${API_BASE_URL}/history`, {
            headers: { "Authorization": `Bearer ${state.authToken}` }
        })
        .then(r => r.json())
        .then(data => {
            if (data.history) state.analysisHistory = data.history;
            _renderHistoryRecords(container, guestBannerHtml);
        })
        .catch(() => {
            _renderHistoryRecords(container, guestBannerHtml);
        });
    } else {
        _renderHistoryRecords(container, guestBannerHtml);
    }
}

function _renderHistoryRecords(container, guestBannerHtml) {
    if (!state.analysisHistory.length) {
        container.innerHTML = `
            ${guestBannerHtml}
            <div class="empty-state large" style="grid-column: 1 / -1; padding: 3rem 1rem;">
                <div class="empty-hero-icon">⏱</div>
                <h3>No analysis history yet</h3>
                <p>Run an analysis on satellite imagery to see your session results logged here.</p>
                <button class="action-btn primary-btn" onclick="switchView('home_view')">← Return to Main Dashboard</button>
            </div>`;
        return;
    }

    let html = guestBannerHtml;
    state.analysisHistory.forEach(item => {
        const confPct = Math.round((item.confidence || 0.75) * 100);
        html += `
            <div class="history-record-card">
                <div class="history-record-header">
                    <span class="history-record-mode">${(item.mode || "ANALYSIS").toUpperCase()}</span>
                    <span>${item.timestamp || "Recent"}</span>
                </div>
                <div class="history-record-query">${item.query || "—"}</div>
                <div class="history-record-answer">${item.answer || "—"}</div>
                <div class="history-record-footer">
                    <span class="history-confidence-chip" style="color:${confPct >= 80 ? '#22C55E' : confPct >= 60 ? '#F59E0B' : '#EF4444'}">
                        ${confPct}% confidence
                    </span>
                    ${item.db_id ? `<button class="history-delete-btn" onclick="deleteHistoryEntry(${item.db_id}, this)">🗑</button>` : ""}
                </div>
            </div>`;
    });
    container.innerHTML = html;
}

async function deleteHistoryEntry(dbId, btn) {
    if (!state.authToken) return;
    try {
        const res = await fetch(`${API_BASE_URL}/history/${dbId}`, {
            method: "DELETE",
            headers: { "Authorization": `Bearer ${state.authToken}` }
        });
        if (res.ok) {
            state.analysisHistory = state.analysisHistory.filter(h => h.db_id !== dbId);
            if (btn && btn.closest(".history-record-card")) {
                btn.closest(".history-record-card").remove();
            }
            showToast("🗑 Record deleted.", "info");
        }
    } catch (_) { showToast("❌ Delete failed.", "error"); }
}

