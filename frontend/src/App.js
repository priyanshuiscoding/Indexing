import React, { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";
import { documentCatalog } from "./documentCatalog";

const API = process.env.REACT_APP_API_BASE || "";
const SAVED_PDF_MEMORY_CACHE_LIMIT = 8;
const BATCH_UPLOAD_CHUNK_SIZE = 20;
const EMPTY_QUEUE_SNAPSHOT = {
  index_ready: [],
  stage1_batch: [],
  pending_vectorization: [],
  vectorized: [],
  reindex_review: [],
  errors: [],
  runner: { running: false, processed: 0, total: 0, current_pdf_id: "", current_filename: "", last_error: "", pause_requested: false, paused: false, heartbeat_ts: 0 },
  index_runner: { running: false, current_pdf_id: "", current_filename: "", last_error: "", finished_pdf_id: "", finished_filename: "", status: "idle" },
  stage1_batch_runner: { running: false, processed: 0, total: 0, current_pdf_id: "", current_filename: "", last_error: "", heartbeat_ts: 0, status: "idle" },
  audit_runner: { running: false, processed: 0, total: 0, flagged: 0, current_pdf_id: "", current_filename: "", last_error: "", heartbeat_ts: 0, status: "idle" },
  reindex_runner: { running: false, processed: 0, total: 0, fixed: 0, current_pdf_id: "", current_filename: "", last_error: "", heartbeat_ts: 0, status: "idle" },
};

const getFetchErrorMessage = (err, path = "") => {
  if (err instanceof TypeError && /fetch/i.test(err.message)) {
    return `Cannot reach the backend${path ? ` for ${path}` : ""}. Check the server, proxy, and backend logs${API ? ` at ${API}` : ""}.`;
  }
  const rawMessage = err?.message || "Request failed";
  if (/^500:/i.test(rawMessage)) {
    return `Backend returned 500${path ? ` for ${path}` : ""}. ${rawMessage.replace(/^500:\s*/i, "")}`.trim();
  }
  if (/^404:/i.test(rawMessage)) {
    return `Requested data was not found${path ? ` for ${path}` : ""}. ${rawMessage.replace(/^404:\s*/i, "")}`.trim();
  }
  return rawMessage;
};

const loadPdfJs = () =>
  new Promise((resolve) => {
    if (window.pdfjsLib) {
      resolve(window.pdfjsLib);
      return;
    }

    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
    script.onload = () => {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc =
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
      resolve(window.pdfjsLib);
    };
    document.head.appendChild(script);
  });

const apiFetch = async (path, opts = {}) => {
  try {
    const resp = await fetch(API + path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`${resp.status}: ${err}`);
    }
    return resp.json();
  } catch (err) {
    throw new Error(getFetchErrorMessage(err, path));
  }
};

const apiUpload = async (path, formData) => {
  try {
    const resp = await fetch(API + path, { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`${resp.status}: ${err}`);
    }
    return resp.json();
  } catch (err) {
    throw new Error(getFetchErrorMessage(err, path));
  }
};

const describeSavedPdfStatus = (item = {}, flags = {}) => {
  const status = (item.status || "").toLowerCase();
  const retrieval = (item.retrieval_status || "").toLowerCase();
  const pending = Number(item.pending_pages || 0);

  if (flags.isDeferredBusy) {
    return "Full ingestion and vectorization are running for this PDF.";
  }
  if (flags.isIndexBusy || flags.isStage1BatchBusy || status === "indexing_running") {
    return "Full-document indexing is running in the background for this PDF.";
  }
  if (item.review_reason) {
    return `Flagged for review: ${item.review_reason}`;
  }
  if (status === "queued_for_stage1" || item.queue_bucket === "stage1_batch") {
    return "Uploaded and queued for full-document indexing. The backend will continue even if you close the browser.";
  }
  if ((retrieval === "full_ingestion_running" || status === "full_ingestion_running") && pending > 0) {
    return "Full-document extraction and vectorization are running before final indexing.";
  }
  if (status === "index_ready" && pending > 0 && retrieval === "queued_for_full_ingestion") {
    return "Legacy partial record detected. Remaining pages are queued for full ingestion.";
  }
  if (status === "index_ready" && pending > 0) {
    return "Legacy partial record detected. Remaining pages can be vectorized from the deferred queue.";
  }
  if (pending > 0) {
    return "Legacy partial record detected. Remaining pages are still pending vectorization.";
  }
  return "This PDF is fully ready in the backend library.";
};

const formatSavedPdfStatus = (item = {}) => {
  const status = (item.status || "").toLowerCase();
  const retrieval = (item.retrieval_status || "").toLowerCase();

  if (status === "queued_for_stage1") return "Queued for Indexing";
  if (status === "indexing_running") return "Indexing Running";
  if (status === "extracting_text") return "Extracting Text";
  if (status === "vectorizing") return "Vectorizing";
  if (status === "building_index") return "Building Index";
  if (status === "index_ready") return retrieval === "queued_for_full_ingestion" ? "Index Ready, Review Pending" : "Index Ready";
  if (status === "full_ingestion_running" || retrieval === "full_ingestion_running") return "Full Ingestion Running";
  if (status === "vectorized" || retrieval === "vectorized") return "Vectorized";
  if (status === "failed" || retrieval === "failed") return "Failed";
  return item.status || "Unknown";
};

const getCurrentStepLabel = (item = {}) => {
  const status = String(item.status || "").toLowerCase();
  const retrieval = String(item.retrieval_status || "").toLowerCase();
  if (item.review_reason || status === "needs_review" || item.queue_bucket === "reindex_review") return "Sent to review";
  if (status === "building_index") return "Building index";
  if (status === "vectorizing") return "Vectorizing";
  if (status === "extracting_text" || retrieval === "full_ingestion_running") return "Extracting text";
  if (status === "vectorized" || item.index_ready || item.chat_ready) return "Finalized";
  if (status === "queued_for_stage1") return "Queued";
  return "Ready";
};

const TIMING_STAGE_LABELS = {
  file_open: "File open",
  first_10_page_extraction: "Initial extraction",
  ocr_time: "OCR",
  toc_detection: "TOC detection",
  llm_indexing_time: "Index generation",
  json_generation_time: "JSON export",
  full_text_extraction_time: "Full text extraction",
  chunking_time: "Chunking",
  embedding_time: "Embeddings",
  vector_db_insert_time: "Vector DB insert",
  total_vectorization_time: "Total vectorization",
  saved_pdf_load_time: "PDF load in viewer",
};

const formatTimingLabel = (key = "") => TIMING_STAGE_LABELS[key] || key.replace(/_/g, " ").trim();

const formatEventTimestamp = (value = "") => {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
};

const apiFetchBlob = async (path, opts = {}) => {
  try {
    const resp = await fetch(API + path, opts);
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`${resp.status}: ${err}`);
    }
    return resp.blob();
  } catch (err) {
    throw new Error(getFetchErrorMessage(err, path));
  }
};

const isRealFile = (value) => value instanceof File || typeof value?.arrayBuffer === "function";

const deriveDocumentMeta = (fileName = "") => {
  const stem = fileName
    .replace(/\.pdf$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const batchMatch = stem.match(/\bbatch\s*(?:no|number)?\.?\s*[:#-]?\s*([a-z0-9/-]+)/i);
  const caseMatch = stem.match(/\b(?:case|appeal|petition|revision|application|writ|suit)\s*(?:no|number)?\.?\s*[:#-]?\s*([a-z0-9/-]+)/i);
  const vsParts = stem.split(/\s+(?:vs\.?|versus|v\.)\s+/i);
  const courtMatch = stem.match(/\b([a-z\s]+court)\b/i);

  return {
    title: stem,
    batchNo: batchMatch?.[1]?.toUpperCase() || "",
    caseNo: caseMatch ? caseMatch[0].trim() : "",
    plaintiff: vsParts.length > 1 ? vsParts[0].trim() : "",
    defendant: vsParts.length > 1 ? vsParts.slice(1).join(" Vs ").trim() : "",
    court: courtMatch?.[1]?.replace(/\s+/g, " ").trim() || "",
  };
};

const normalizeCatalogLabel = (value = "") =>
  value.toLowerCase().replace(/[^a-z0-9]+/gi, "");

const catalogParentsByName = new Map();
const catalogSubdocumentsByParent = new Map();
const catalogCombinedLookup = new Map();
const catalogUniqueSubdocumentLookup = new Map();
const catalogAllSubdocuments = [];
const catalogAllSubdocumentNames = new Set();

documentCatalog.forEach((doc) => {
  const parentName = doc.name.trim();
  const parentKey = normalizeCatalogLabel(parentName);
  catalogParentsByName.set(parentKey, parentName);
  catalogSubdocumentsByParent.set(parentName, doc.subDocuments || []);

  (doc.subDocuments || []).forEach((subDoc) => {
    const subName = subDoc.name.trim();
    const subKey = normalizeCatalogLabel(subName);
    if (!catalogAllSubdocumentNames.has(subName)) {
      catalogAllSubdocumentNames.add(subName);
      catalogAllSubdocuments.push({ ...subDoc, parentName });
    }
    [
      `${parentName}${subName}`,
      `${parentName} ${subName}`,
      `${parentName}-${subName}`,
    ].forEach((combined) => {
      const combinedKey = normalizeCatalogLabel(combined);
      if (combinedKey) {
        catalogCombinedLookup.set(combinedKey, { title: parentName, subDocument: subName });
      }
    });

    const existing = catalogUniqueSubdocumentLookup.get(subKey) || [];
    existing.push({ title: parentName, subDocument: subName });
    catalogUniqueSubdocumentLookup.set(subKey, existing);
  });
});

const applyCatalogClassification = (item) => {
  const title = (item.title || "").trim();
  const subDocument = (item.subDocument || "").trim();
  const titleKey = normalizeCatalogLabel(title);

  if (!titleKey) return item;

  if (catalogParentsByName.has(titleKey)) {
    const parentTitle = catalogParentsByName.get(titleKey);
    const subOptions = catalogSubdocumentsByParent.get(parentTitle) || [];
    return {
      ...item,
      title: parentTitle,
      subDocument: subDocument || (subOptions.length === 0 ? "null" : ""),
    };
  }

  if (!subDocument && catalogCombinedLookup.has(titleKey)) {
    return {
      ...item,
      ...catalogCombinedLookup.get(titleKey),
    };
  }

  const subMatches = catalogUniqueSubdocumentLookup.get(titleKey);
  if (!subDocument && subMatches?.length === 1) {
    return {
      ...item,
      ...subMatches[0],
    };
  }

  return item;
};

catalogAllSubdocuments.sort((a, b) => {
  const aCode = Number(a.code);
  const bCode = Number(b.code);
  if (Number.isFinite(aCode) && Number.isFinite(bCode)) {
    return aCode - bCode || a.name.localeCompare(b.name);
  }
  return a.name.localeCompare(b.name);
});

const normalizeSections = (items = [], batchNo = "") =>
  items.map((item) => ({
    ...applyCatalogClassification(item),
    displayTitle: (item.displayTitle || item.originalTitle || "").trim(),
    originalTitle: (item.originalTitle || item.displayTitle || item.title || "").trim(),
    batchNo: (item.batchNo || batchNo || "").trim(),
    pageFrom: item.pageFrom || item.pdfPageFrom || 1,
    pageTo: item.pageTo || item.pdfPageTo || item.pageFrom || item.pdfPageFrom || 1,
    pdfPageFrom: item.pdfPageFrom || item.pageFrom || 1,
    pdfPageTo: item.pdfPageTo || item.pageTo || item.pageFrom || 1,
  }));

const formatPageValue = (primaryFrom, primaryTo, fallbackFrom, fallbackTo, primaryLabel, fallbackLabel) => {
  const primaryStart = Number(primaryFrom);
  const primaryEnd = Number(primaryTo || primaryFrom);
  if (!Number.isFinite(primaryStart) || primaryStart < 1) {
    const fallbackStart = Number(fallbackFrom);
    const fallbackEnd = Number(fallbackTo || fallbackFrom);
    if (!Number.isFinite(fallbackStart) || fallbackStart < 1) return "-";
    return fallbackStart === fallbackEnd ? String(fallbackStart) : `${fallbackStart}-${fallbackEnd}`;
  }
  const primaryText = primaryStart === primaryEnd ? `${primaryStart}` : `${primaryStart}-${primaryEnd}`;
  const fallbackStart = Number(fallbackFrom);
  const fallbackEnd = Number(fallbackTo || fallbackFrom);
  if (!Number.isFinite(fallbackStart) || fallbackStart < 1) return primaryText;
  const fallbackText = fallbackStart === fallbackEnd ? `${fallbackStart}` : `${fallbackStart}-${fallbackEnd}`;
  if (primaryText === fallbackText) return primaryText;
  return `${primaryLabel} ${primaryText} | ${fallbackLabel} ${fallbackText}`;
};

const getSavedPdfTone = (item = {}) => {
  const status = String(item.status || "").toLowerCase();
  const retrieval = String(item.retrieval_status || "").toLowerCase();
  const pending = Number(item.pending_pages || 0);

  if (item.queue_bucket === "reindex_review" || status === "needs_review" || item.review_reason) return "review";
  if (status === "failed" || retrieval === "failed") return "error";
  if (retrieval.includes("running") || status.includes("running")) return "running";
  if (retrieval === "vectorized" || status === "vectorized" || item.chat_ready) return "ready";
  if (status === "index_ready" || item.index_ready) return pending > 0 ? "staged" : "indexed";
  if (pending > 0 || retrieval === "pending_deferred_ingestion" || retrieval === "queued_for_full_ingestion") return "queued";
  return "neutral";
};

const getSavedPdfToneLabel = (tone) => {
  if (tone === "review") return "Review Queue";
  if (tone === "ready") return "Vectorized";
  if (tone === "indexed") return "Indexed";
  if (tone === "staged") return "Indexed Only";
  if (tone === "queued") return "Pending Queue";
  if (tone === "running") return "Processing";
  if (tone === "error") return "Needs Review";
  return "Saved";
};

function PDFPage({ doc, pageNum, scale, isActive, onVisible }) {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const rendered = useRef(false);
  const renderTask = useRef(null);

  const draw = useCallback(async () => {
    if (!doc || !canvasRef.current) return;
    if (renderTask.current) {
      try {
        renderTask.current.cancel();
      } catch (_) {}
    }

    try {
      const page = await doc.getPage(pageNum);
      const viewport = page.getViewport({ scale });
      const canvas = canvasRef.current;
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const task = page.render({ canvasContext: canvas.getContext("2d"), viewport });
      renderTask.current = task;
      await task.promise;
      rendered.current = true;
    } catch (err) {
      if (err.name !== "RenderingCancelledException") console.error(err);
    }
  }, [doc, pageNum, scale]);

  useEffect(() => {
    rendered.current = false;
    draw();
  }, [scale, draw]);

  useEffect(() => {
    const handlePrerender = () => {
      if (!rendered.current) draw();
    };
    const el = wrapRef.current;
    if (el) el.addEventListener("prerender", handlePrerender);
    return () => {
      if (el) el.removeEventListener("prerender", handlePrerender);
    };
  }, [draw]);

  useEffect(() => {
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        onVisible(pageNum);
        if (!rendered.current) draw();
        for (let i = 1; i <= 3; i += 1) {
          document.getElementById(`pg-${pageNum + i}`)?.dispatchEvent(new CustomEvent("prerender"));
        }
      },
      { rootMargin: "400px", threshold: 0.01 }
    );

    if (wrapRef.current) obs.observe(wrapRef.current);
    return () => obs.disconnect();
  }, [pageNum, draw, onVisible]);

  return (
    <div ref={wrapRef} id={`pg-${pageNum}`} className={`pdf-page-wrap ${isActive ? "active" : ""}`}>
      <div className="pdf-page-num">{pageNum}</div>
      <canvas ref={canvasRef} />
    </div>
  );
}

function ConfirmModal({ message, onConfirm, onCancel }) {
  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onCancel()}>
      <div className="modal-box" style={{ maxWidth: 380 }}>
        <p style={{ margin: "0 0 20px", color: "#374151", fontSize: 14 }}>{message}</p>
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
          <button className="btn-ghost" onClick={onCancel}>Cancel</button>
          <button className="btn-danger" onClick={onConfirm}>Delete</button>
        </div>
      </div>
    </div>
  );
}

function QueueStatusModal({
  open,
  onClose,
  savedPdfs,
  queueSnapshot,
  runner,
  runnerStateLabel,
  runnerLooksStuck,
  runnerStopping,
  heartbeatAgeSeconds,
  indexRunner,
  stage1BatchRunner,
  auditRunner,
  reindexRunner,
  loading,
  runAllDeferredQueue,
  controlDeferredQueue,
  forceResetQueue,
  startIndexAudit,
  runReviewReindexQueue,
  stopAllProcessing,
  timingEvents,
  viewerTimingEvent,
  showTimingPanel,
  setShowTimingPanel,
  selectedPdfLabel,
  auditRowStart,
  auditRowEnd,
  setAuditRowStart,
  setAuditRowEnd,
  pdfSearch,
  batchFilter,
  batchReports = [],
}) {
  if (!open) return null;

  const reviewQueue = queueSnapshot.reindex_review || [];
  const batchQueue = queueSnapshot.stage1_batch || [];
  const pendingQueue = queueSnapshot.pending_vectorization || [];
  const reviewQueueCount = reviewQueue.length;
  const stage1QueueCount = batchQueue.length;
  const auditHeartbeatAgeSeconds = auditRunner?.heartbeat_ts ? Math.max(0, Math.round(Date.now() / 1000 - auditRunner.heartbeat_ts)) : 0;
  const reindexHeartbeatAgeSeconds = reindexRunner?.heartbeat_ts ? Math.max(0, Math.round(Date.now() / 1000 - reindexRunner.heartbeat_ts)) : 0;
  const currentReviewPosition = reindexRunner.running ? Math.min((reindexRunner.processed || 0) + 1, reindexRunner.total || 0) : 0;
  const currentBatchPosition = stage1BatchRunner.running ? Math.min((stage1BatchRunner.processed || 0) + 1, stage1BatchRunner.total || 0) : 0;

  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-box queue-modal-box">
        <div className="queue-modal-header">
          <div>
            <div className="section-title">Operations Panel</div>
            <div className="section-subtitle">Keep batch indexing, audit, and review reindexing in one clean operations panel.</div>
          </div>
          <button className="modal-close" onClick={onClose}>x</button>
        </div>
        <div className="queue-toolbar">
          <span className="queue-pill">Index queue: {savedPdfs.filter((item) => !item.index_ready).length}</span>
          <span className="queue-pill">Batch indexing queue: {stage1QueueCount}</span>
          <span className="queue-pill">Deferred queue: {(queueSnapshot.pending_vectorization || []).length}</span>
          <span className="queue-pill warning">Review queue: {reviewQueueCount}</span>
          {(queueSnapshot.runner?.running || queueSnapshot.runner?.paused) && (
            <span className="queue-pill success">{runnerStateLabel} {queueSnapshot.runner.processed}/{queueSnapshot.runner.total}: {queueSnapshot.runner.current_filename || queueSnapshot.runner.current_pdf_id || "Queue saved"}</span>
          )}
          {indexRunner.running && (
            <span className="queue-pill success">Indexing: {indexRunner.current_filename || indexRunner.current_pdf_id || "Indexing running"}</span>
          )}
          {stage1BatchRunner.running && (
            <span className="queue-pill success">Batch indexing: {stage1BatchRunner.processed || 0}/{stage1BatchRunner.total || 0}</span>
          )}
          {auditRunner.running && (
            <span className="queue-pill warning">Audit: {auditRunner.processed || 0}/{auditRunner.total || 0} - flagged {auditRunner.flagged || 0}</span>
          )}
          {reindexRunner.running && (
            <span className="queue-pill success">Review Reindex: {reindexRunner.processed || 0}/{reindexRunner.total || 0}</span>
          )}
        </div>
        <div className="queue-audit-panel">
          <div className="queue-audit-copy">
            Audit uses the current PDF search and Batch No. filters, then checks only fully vectorized PDFs for stale or weak indexes.
          </div>
          <div className="queue-audit-inputs">
            <input
              type="number"
              min={1}
              className="range-input queue-range-input"
              value={auditRowStart}
              onChange={(e) => setAuditRowStart(e.target.value)}
              placeholder="From #"
            />
            <input
              type="number"
              min={1}
              className="range-input queue-range-input"
              value={auditRowEnd}
              onChange={(e) => setAuditRowEnd(e.target.value)}
              placeholder="To #"
            />
            <button className="queue-run-btn" onClick={startIndexAudit} disabled={loading || auditRunner.running || reindexRunner.running}>Audit Filtered PDFs</button>
            <button className="queue-run-btn" onClick={runReviewReindexQueue} disabled={loading || reindexRunner.running || reviewQueueCount === 0}>Reindex Review Queue</button>
          </div>
          <div className="queue-audit-note">
            Filters now: PDF Search <strong>{pdfSearch || "All"}</strong> - Batch No. <strong>{batchFilter || "All"}</strong> - Rows <strong>{auditRowStart || "1"}{auditRowEnd ? `-${auditRowEnd}` : "+"}</strong>
          </div>
        </div>
        <div className="queue-actions-grid">
          <button className="queue-run-btn" onClick={runAllDeferredQueue} disabled={loading || queueSnapshot.runner?.running || queueSnapshot.runner?.paused}>Run All Deferred Queue</button>
          <button className="queue-pause-btn" onClick={() => controlDeferredQueue("stop")} disabled={loading || !queueSnapshot.runner?.running}>Stop Queue</button>
          <button className="queue-resume-btn" onClick={() => controlDeferredQueue("resume")} disabled={loading || !queueSnapshot.runner?.paused || queueSnapshot.runner?.running}>Resume Queue</button>
          <button className="queue-stop-all-btn" onClick={stopAllProcessing} disabled={loading}>Stop All Processing</button>
          <button className="queue-reset-btn" onClick={() => forceResetQueue("index")} disabled={loading || queueSnapshot.runner?.running}>Force Reset Index Queue</button>
          <button className="queue-reset-btn" onClick={() => forceResetQueue("deferred")} disabled={loading || queueSnapshot.runner?.running}>Force Reset Deferred Queue</button>
          <button className="queue-reset-btn" onClick={() => forceResetQueue("stage1_batch")} disabled={loading || stage1BatchRunner.running}>Force Reset Batch Queue</button>
          <button className="queue-reset-btn" onClick={() => forceResetQueue("reindex")} disabled={loading || reindexRunner.running}>Clear Review Queue</button>
        </div>
        <div className="library-help">
          Chunked batch upload feeds the backend indexing queue, which processes whole PDFs before generating final indexes. The review queue is separate and only repairs suspicious saved indexes from already vectorized PDFs.
        </div>
        <div className="queue-list-grid">
          <div className="queue-list-card">
            <div className="queue-list-title">Review Queue</div>
            <div className="queue-list-subtitle">Top of queue is shown first, so operators always know the exact PDF order.</div>
            {reviewQueue.length ? reviewQueue.slice(0, 8).map((item, idx) => (
              <div key={`review-${item.pdf_id}`} className="queue-list-row">
                <div className="queue-list-number">#{idx + 1}</div>
                <div className="queue-list-content">
                  <div className="queue-list-name">{item.cnr_number || item.filename || item.pdf_id}</div>
                  <div className="queue-list-meta">{item.filename || item.pdf_id}{item.review_reason ? ` - ${item.review_reason}` : ""}</div>
                </div>
              </div>
            )) : <div className="queue-list-empty">No PDFs are waiting in the review queue.</div>}
          </div>
          <div className="queue-list-card">
            <div className="queue-list-title">Batch Queue</div>
            <div className="queue-list-subtitle">Current position: {stage1BatchRunner.running ? `${currentBatchPosition} of ${stage1BatchRunner.total || 0}` : 'Idle'}</div>
            {batchQueue.length ? batchQueue.slice(0, 8).map((item, idx) => (
              <div key={`batch-${item.pdf_id}`} className="queue-list-row">
                <div className="queue-list-number">#{idx + 1}</div>
                <div className="queue-list-content">
                  <div className="queue-list-name">{item.cnr_number || item.filename || item.pdf_id}</div>
                  <div className="queue-list-meta">{item.filename || item.pdf_id}</div>
                </div>
              </div>
            )) : <div className="queue-list-empty">No PDFs are waiting in the batch queue.</div>}
          </div>
        </div>
        <div className="runner-panel compact batch-report-panel">
          <div className="runner-panel-header">
            <span className="runner-state idle">Recent Batch Reports</span>
            <span className="runner-meta">Saved in backend for overnight comparison</span>
          </div>
          {(Array.isArray(batchReports) && batchReports.length) ? (
            <div className="batch-report-list">
              {batchReports.map((report) => {
                const summary = report.summary || {};
                return (
                  <div key={report.batch_run_id} className="batch-report-card">
                    <div className="batch-report-head">
                      <strong>{report.batch_run_id}</strong>
                      <span className={`runner-state ${report.status === 'completed' ? 'idle' : 'running'}`}>{report.status || 'unknown'}</span>
                    </div>
                    <div className="batch-report-meta">Started {formatEventTimestamp(report.started_at)} | Last update {formatEventTimestamp(report.generated_at)}</div>
                    <div className="batch-report-grid">
                      <div><span>Total PDFs</span><strong>{summary.total_pdfs || 0}</strong></div>
                      <div><span>Completed</span><strong>{summary.completed_pdfs || 0}</strong></div>
                      <div><span>Review</span><strong>{summary.review_pdfs || 0}</strong></div>
                      <div><span>Vectorized</span><strong>{summary.vectorized_pdfs || 0}</strong></div>
                      <div><span>Logged time</span><strong>{summary.total_logged_seconds || 0}s</strong></div>
                    </div>
                    <div className="batch-report-meta">{report.path}</div>
                  </div>
                );
              })}
            </div>
          ) : <div className="queue-list-empty">No batch reports have been saved yet.</div>}
        </div>

        <div className="runner-panel compact timing-panel">
          <div className="runner-panel-header">
            <span className="runner-state idle">Timing History</span>
            <span className="runner-meta">{selectedPdfLabel || "No PDF selected"}</span>
            <button className="add-toggle-btn" onClick={() => setShowTimingPanel((value) => !value)}>{showTimingPanel ? "Hide Logs" : "Show Logs"}</button>
          </div>
          {showTimingPanel && (
            <div className="timing-history-list">
              {viewerTimingEvent ? (
                <div className="timing-history-card timing-history-card-local">
                  <div className="timing-history-head">
                    <span className="timing-run-label">{viewerTimingEvent.run_label}</span>
                    <span className="timing-run-meta">{formatEventTimestamp(viewerTimingEvent.recorded_at)} - total {viewerTimingEvent.total_seconds}s</span>
                  </div>
                  <div className="timing-history-grid">
                    {Object.entries(viewerTimingEvent.timings || {}).map(([key, value]) => (
                      <div key={`${viewerTimingEvent.recorded_at}-${key}`} className="timing-history-row">
                        <span>{formatTimingLabel(key)}</span>
                        <strong>{value}s</strong>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
              {timingEvents.length ? timingEvents.map((event, idx) => (
                <div key={`timing-${idx}-${event.recorded_at}`} className="timing-history-card">
                  <div className="timing-history-head">
                    <span className="timing-run-label">{event.run_label}</span>
                    <span className="timing-run-meta">{formatEventTimestamp(event.recorded_at)} - total {event.total_seconds}s</span>
                  </div>
                  <div className="timing-history-grid">
                    {Object.entries(event.timings || {}).map(([key, value]) => (
                      <div key={`${event.recorded_at}-${key}`} className="timing-history-row">
                        <span>{formatTimingLabel(key)}</span>
                        <strong>{value}s</strong>
                      </div>
                    ))}
                  </div>
                </div>
              )) : (!viewerTimingEvent ? <div className="queue-list-empty">No timing history recorded yet for this PDF.</div> : null)}
            </div>
          )}
        </div>
        <div className="runner-panel compact">
          <div className="runner-panel-header">
            <span className={`runner-state ${runnerLooksStuck ? "stuck" : runner.paused ? "paused" : runner.running ? "running" : "idle"}`}>{runnerStateLabel}</span>
            <span className="runner-meta">Processed {runner.processed || 0} of {runner.total || 0}</span>
            {runner.heartbeat_ts ? <span className="runner-meta">Last update {heartbeatAgeSeconds}s ago</span> : null}
          </div>
          <div className="runner-grid compact">
            <div>
              <div className="runner-label">Current PDF</div>
              <div className="runner-value">{runner.current_filename || runner.current_pdf_id || "None"}</div>
            </div>
            <div>
              <div className="runner-label">Queue State</div>
              <div className="runner-value">{runner.paused ? "Deferred queue is paused in backend" : runnerStopping ? "Deferred queue will stop after the current PDF" : runner.running ? "Deferred queue is processing in backend" : "No background queue is running"}</div>
            </div>
            <div>
              <div className="runner-label">Last Error</div>
              <div className={`runner-value ${runner.last_error ? "error" : "muted"}`}>{runner.last_error || "No recent error"}</div>
            </div>
            <div>
              <div className="runner-label">Operator Hint</div>
              <div className={`runner-value ${runnerLooksStuck ? "error" : "muted"}`}>
                {runnerLooksStuck
                  ? "No heartbeat for more than 90 seconds. Check backend logs or use Force Reset if needed."
                  : runner.paused
                    ? "Queue is paused safely. Resume when you want backend processing to continue."
                    : runnerStopping
                      ? "Stop has been requested. The queue will pause after the current PDF finishes."
                      : runner.running
                        ? "Queue is active. You can keep working in the UI while it continues."
                        : "Start the deferred queue when you want to vectorize all pending PDFs."}
              </div>
            </div>
          </div>
        </div>
        <div className="runner-panel compact">
          <div className="runner-panel-header">
            <span className={`runner-state ${stage1BatchRunner.running ? "running" : "idle"}`}>{stage1BatchRunner.running ? "Batch Indexing Running" : "Batch Indexing Idle"}</span>
            <span className="runner-meta">Processed {stage1BatchRunner.processed || 0} of {stage1BatchRunner.total || 0}</span>
            {stage1BatchRunner.heartbeat_ts ? <span className="runner-meta">Last update {Math.max(0, Math.round(Date.now() / 1000 - stage1BatchRunner.heartbeat_ts))}s ago</span> : null}
          </div>
          <div className="runner-grid compact">
            <div>
              <div className="runner-label">Current PDF</div>
              <div className="runner-value">{stage1BatchRunner.current_filename || stage1BatchRunner.current_pdf_id || "None"}</div>
              {stage1BatchRunner.running ? <div className="runner-value muted">Position {currentBatchPosition} of {stage1BatchRunner.total || 0} from top of queue</div> : null}
            </div>
            <div>
              <div className="runner-label">Queue State</div>
              <div className="runner-value">{stage1BatchRunner.running ? "Chunked batch queue is processing full-document indexing in the backend." : "No batch indexing queue is running."}</div>
            </div>
            <div>
              <div className="runner-label">Last Error</div>
              <div className={`runner-value ${stage1BatchRunner.last_error ? "error" : "muted"}`}>{stage1BatchRunner.last_error || "No recent error"}</div>
            </div>
            <div>
              <div className="runner-label">Operator Hint</div>
              <div className="runner-value muted">Upload large batches in chunks and let the backend finish full-document indexing even after the browser tab is closed.</div>
            </div>
          </div>
        </div>
        <div className="runner-panel compact">
          <div className="runner-panel-header">
            <span className={`runner-state ${auditRunner.running ? "running" : "idle"}`}>{auditRunner.running ? "Audit Running" : "Audit Idle"}</span>
            <span className="runner-meta">Processed {auditRunner.processed || 0} of {auditRunner.total || 0}</span>
            <span className="runner-meta">Flagged {auditRunner.flagged || 0}</span>
            {auditRunner.heartbeat_ts ? <span className="runner-meta">Last update {auditHeartbeatAgeSeconds}s ago</span> : null}
          </div>
          <div className="runner-grid compact">
            <div>
              <div className="runner-label">Current PDF</div>
              <div className="runner-value">{auditRunner.current_filename || auditRunner.current_pdf_id || "None"}</div>
            </div>
            <div>
              <div className="runner-label">Audit Scope</div>
              <div className="runner-value muted">Checks saved indexes for full-document coverage and oversized gap patterns.</div>
            </div>
            <div>
              <div className="runner-label">Last Error</div>
              <div className={`runner-value ${auditRunner.last_error ? "error" : "muted"}`}>{auditRunner.last_error || "No recent error"}</div>
            </div>
            <div>
              <div className="runner-label">Operator Hint</div>
              <div className="runner-value muted">Run audit after batch vectorization when you want to isolate suspicious indexes instead of reindexing everything.</div>
            </div>
          </div>
        </div>
        <div className="runner-panel compact">
          <div className="runner-panel-header">
            <span className={`runner-state ${reindexRunner.running ? "running" : "idle"}`}>{reindexRunner.running ? "Review Reindex Running" : "Review Reindex Idle"}</span>
            <span className="runner-meta">Processed {reindexRunner.processed || 0} of {reindexRunner.total || 0}</span>
            <span className="runner-meta">Fixed {reindexRunner.fixed || 0}</span>
            {reindexRunner.heartbeat_ts ? <span className="runner-meta">Last update {reindexHeartbeatAgeSeconds}s ago</span> : null}
          </div>
          <div className="runner-grid compact">
            <div>
              <div className="runner-label">Current PDF</div>
              <div className="runner-value">{reindexRunner.current_filename || reindexRunner.current_pdf_id || "None"}</div>
              {reindexRunner.running ? <div className="runner-value muted">Position {currentReviewPosition} of {reindexRunner.total || 0} from top of review queue</div> : null}
            </div>
            <div>
              <div className="runner-label">Review Queue</div>
              <div className="runner-value">{reviewQueueCount > 0 ? `${reviewQueueCount} PDFs waiting for index repair` : "No PDFs are waiting in the review queue"}</div>
            </div>
            <div>
              <div className="runner-label">Last Error</div>
              <div className={`runner-value ${reindexRunner.last_error ? "error" : "muted"}`}>{reindexRunner.last_error || "No recent error"}</div>
            </div>
            <div>
              <div className="runner-label">Operator Hint</div>
              <div className="runner-value muted">This runner reuses saved page text from vectorized PDFs and repairs only the flagged index records.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function InlineEditRow({ section, totalPages, onSave, onCancel }) {
  const [title, setTitle] = useState(section.title || "");
  const [subDoc, setSubDoc] = useState(section.subDocument || "");
  const [note, setNote] = useState(section.note || "");
  const [batchNo, setBatchNo] = useState(section.batchNo || "");
  const [pageFrom, setPageFrom] = useState(section.pageFrom || section.pdfPageFrom || 1);
  const [pageTo, setPageTo] = useState(section.pageTo || section.pdfPageTo || section.pageFrom || 1);
  const [recDate, setRecDate] = useState(section.receivingDate || "");

  const save = () => {
    if (!title.trim()) return;
    const rawFrom = parseInt(pageFrom, 10);
    const rawTo = parseInt(pageTo, 10);
    const resolvedFrom = Math.max(1, Math.min(rawFrom || section.pageFrom || 1, totalPages));
    const resolvedTo = Math.max(resolvedFrom, Math.min(rawTo || resolvedFrom, totalPages));
    const normalizedSubDoc = subDoc.trim() || "null";
    onSave({
      ...section,
      title: title.trim(),
      subDocument: normalizedSubDoc,
      note: note.trim(),
      batchNo: batchNo.trim(),
      pageFrom: resolvedFrom,
      pageTo: resolvedTo,
      pdfPageFrom: resolvedFrom,
      pdfPageTo: resolvedTo,
      receivingDate: recDate,
      source: section.source === "gap" ? "manual" : section.source,
    });
  };

  return (
    <tr className="inline-edit-row">
      <td colSpan={8} style={{ padding: 0 }}>
        <div className="inline-edit-panel">
          <div className="inline-edit-header">
            <span>Edit Entry</span>
            <button className="modal-close" onClick={onCancel}>x</button>
          </div>
          <div className="form-grid">
            <div className="form-row">
              <label className="form-label">Document Title</label>
              <input className="form-input" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Document name" />
            </div>
            <div className="form-row">
              <label className="form-label">Sub Document</label>
              <select className="form-input" value={subDoc} onChange={(e) => setSubDoc(e.target.value)}>
                <option value="">Select</option>
                <option value="null">null</option>
                {catalogAllSubdocuments.map((item) => (
                  <option key={`${item.parentName}-${item.code}`} value={item.name}>{item.name}</option>
                ))}
              </select>
            </div>
            <div className="form-row-full">
              <label className="form-label">Note</label>
              <input className="form-input" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Additional note (optional)" />
            </div>
            <div className="form-row">
              <label className="form-label">Batch No.</label>
              <input className="form-input" value={batchNo} onChange={(e) => setBatchNo(e.target.value)} placeholder="Batch number (optional)" />
            </div>
            <div className="form-row-pages">
              <div>
                <label className="form-label">From Page</label>
                <input type="number" className="form-input" min={1} max={totalPages} value={pageFrom} onChange={(e) => setPageFrom(e.target.value)} />
              </div>
              <div>
                <label className="form-label">To Page</label>
                <input type="number" className="form-input" min={1} max={totalPages} value={pageTo} onChange={(e) => setPageTo(e.target.value)} />
              </div>
              <div>
                <label className="form-label">Receiving Date</label>
                <input type="date" className="form-input" value={recDate} onChange={(e) => setRecDate(e.target.value)} />
              </div>
            </div>
          </div>
          <div className="inline-edit-footer">
            <button className="btn-ghost" onClick={onCancel}>Cancel</button>
            <button className="btn-save" onClick={save}>Save</button>
          </div>
        </div>
      </td>
    </tr>
  );
}

function AddEntryPanel({ totalPages, caseApplicant, defaultBatchNo, onAdd, documents }) {
  const [doc, setDoc] = useState("");
  const [subDoc, setSubDoc] = useState("");
  const [customDocName, setCustomDocName] = useState("");
  const [note, setNote] = useState("");
  const [batchNo, setBatchNo] = useState(defaultBatchNo || "");
  const [pageFrom, setPageFrom] = useState("");
  const [pageTo, setPageTo] = useState("");
  const [recDate, setRecDate] = useState("");
  const [total, setTotal] = useState("");

  useEffect(() => {
    const preferredFrom = parseInt(pageFrom, 10);
    const preferredTo = parseInt(pageTo, 10);
    if (preferredFrom && preferredTo && preferredTo >= preferredFrom) setTotal(String(preferredTo - preferredFrom + 1));
  }, [pageFrom, pageTo]);

  useEffect(() => {
    setBatchNo(defaultBatchNo || "");
  }, [defaultBatchNo]);

  useEffect(() => {
    if (!doc) {
      setSubDoc("");
    }
  }, [doc, subDoc]);

  const save = () => {
    const title = doc === "Other" ? customDocName.trim() : doc.trim();
    if (!title) return;
    const rawFrom = parseInt(pageFrom, 10);
    const rawTo = parseInt(pageTo, 10);
    const resolvedFrom = Math.max(1, Math.min(rawFrom || 1, totalPages));
    const resolvedTo = Math.max(resolvedFrom, Math.min(rawTo || resolvedFrom, totalPages));

    onAdd({
      title,
      subDocument: subDoc.trim() || "null",
      note: note.trim(),
      batchNo: batchNo.trim(),
      pageFrom: resolvedFrom,
      pageTo: resolvedTo,
      pdfPageFrom: resolvedFrom,
      pdfPageTo: resolvedTo,
      totalPage: parseInt(total, 10) || (resolvedTo - resolvedFrom + 1),
      receivingDate: recDate,
      source: "manual",
    });
    setDoc("");
    setSubDoc("");
    setCustomDocName("");
    setNote("");
    setPageFrom("");
    setPageTo("");
    setRecDate("");
    setTotal("");
  };

  return (
    <div className="add-entry-panel">
      <div className="add-field-row">
        <span className="add-field-label">Applicant</span>
        <span className="add-field-value applicant-name">{caseApplicant || "-"}</span>
      </div>
      <div className="add-field-row">
        <span className="add-field-label">Document</span>
        <div className="add-field-input-wrap">
          <select className="add-select" value={doc} onChange={(e) => setDoc(e.target.value)}>
            <option value="">Select</option>
            {documents.map((item) => (
              <option key={item.code} value={item.name}>{item.name}</option>
            ))}
            <option>Other</option>
          </select>
          {doc === "Other" && (
            <input className="add-input" placeholder="Type document name..." value={customDocName} onChange={(e) => setCustomDocName(e.target.value)} />
          )}
        </div>
      </div>
      <div className="add-field-row">
        <span className="add-field-label">Sub Document</span>
        <div className="add-field-input-wrap">
          <select className="add-select" value={subDoc} onChange={(e) => setSubDoc(e.target.value)}>
            <option value="">Select</option>
            <option value="null">null</option>
            {catalogAllSubdocuments.map((item) => (
              <option key={`${item.parentName}-${item.code}`} value={item.name}>{item.name}</option>
            ))}
          </select>
        </div>
      </div>
      <div className="add-field-row">
        <span className="add-field-label">Other</span>
        <input className="add-input full" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Additional details..." />
      </div>
      <div className="add-field-row">
        <span className="add-field-label">Batch No.</span>
        <input className="add-input full" value={batchNo} onChange={(e) => setBatchNo(e.target.value)} placeholder="Batch number (optional)" />
      </div>
      <div className="add-pages-row">
        {[["From Page", pageFrom, setPageFrom], ["To Page", pageTo, setPageTo], ["Total Page", total, setTotal]].map(([label, value, setter]) => (
          <div key={label} className="add-page-field">
            <label className="add-page-label">{label}</label>
            <input type="number" className="add-page-input" value={value} onChange={(e) => setter(e.target.value)} />
          </div>
        ))}
        <div className="add-page-field">
          <label className="add-page-label">Receiving Date</label>
          <input type="date" className="add-page-input" value={recDate} onChange={(e) => setRecDate(e.target.value)} />
        </div>
      </div>
      <div style={{ display: "flex", justifyContent: "center", marginTop: 12 }}>
        <button className="btn-save-main" onClick={save} disabled={!doc.trim() || (doc === "Other" && !customDocName.trim())}>Save</button>
      </div>
    </div>
  );
}

function ChatPanel({ pdfId, currentPage, messages, setMessages, input, setInput, onJumpToPage }) {
  const [loading, setLoading] = useState(false);
  const [transformingKey, setTransformingKey] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const transformMessage = async (idx, action) => {
    const msg = messages[idx];
    if (!msg?.text || msg.role !== "assistant") return;

    const cacheKey = action === "translate" ? "translatedText" : "transliteratedText";
    if (msg[cacheKey]) {
      setMessages((items) => items.map((item, itemIdx) => (
        itemIdx === idx
          ? { ...item, showTransformed: item.showTransformed === action ? "" : action }
          : item
      )));
      return;
    }

    setTransformingKey(`${idx}-${action}`);
    try {
      const resp = await apiFetch("/api/text-transform", {
        method: "POST",
        body: JSON.stringify({ text: msg.text, action }),
      });
      setMessages((items) => items.map((item, itemIdx) => {
        if (itemIdx !== idx) return item;
        return {
          ...item,
          [cacheKey]: resp.text,
          showTransformed: action,
        };
      }));
    } catch (err) {
      setMessages((items) => items.map((item, itemIdx) => (
        itemIdx === idx
          ? { ...item, transformError: err.message }
          : item
      )));
    } finally {
      setTransformingKey("");
    }
  };

  const send = async () => {
    if (!input.trim() || loading) return;
    const question = input.trim();
    setInput("");
    setMessages((items) => [...items, { role: "user", text: question }]);
    setLoading(true);

    try {
      const resp = await apiFetch("/api/query", {
        method: "POST",
        body: JSON.stringify({ pdf_id: pdfId, question, top_k: 8, current_page: currentPage }),
      });
      setMessages((items) => [
        ...items,
        { role: "assistant", text: resp.answer, pageRefs: resp.page_refs },
      ]);
    } catch (err) {
      setMessages((items) => [...items, { role: "assistant", text: `Error: ${err.message}`, isError: true }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <span className="chat-title">Document Assistant</span>
        <span className="chat-sub">Ask anything in Hindi or English</span>
      </div>
      <div className="chat-messages">
        {messages.map((msg, idx) => {
          const shownText = msg.showTransformed === "translate"
            ? (msg.translatedText || msg.text)
            : msg.showTransformed === "transliterate"
              ? (msg.transliteratedText || msg.text)
              : msg.text;
          const isTransforming = transformingKey === `${idx}-translate` || transformingKey === `${idx}-transliterate`;

          return (
            <div key={idx} className={`chat-msg ${msg.role} ${msg.isError ? "error" : ""}`}>
              <div className="chat-bubble">{shownText}</div>
              {msg.role === "assistant" && !msg.isError && (
                <div className="chat-page-refs">
                  <button className="page-ref-btn" onClick={() => transformMessage(idx, "translate")} disabled={isTransforming}>
                    {transformingKey === `${idx}-translate` ? "Translating..." : msg.showTransformed === "translate" ? "Original" : "Translate"}
                  </button>
                  <button className="page-ref-btn" onClick={() => transformMessage(idx, "transliterate")} disabled={isTransforming}>
                    {transformingKey === `${idx}-transliterate` ? "Transliterating..." : msg.showTransformed === "transliterate" ? "Original" : "Transliterate"}
                  </button>
                </div>
              )}
              {msg.transformError && <div className="chat-sub" style={{ marginTop: 6, color: "#b91c1c" }}>{msg.transformError}</div>}
              {msg.pageRefs?.length > 0 && (
                <div className="chat-page-refs">
                  {msg.pageRefs.map((page) => (
                    <button key={page} className="page-ref-btn" onClick={() => onJumpToPage(page)}>
                      p.{page}
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
        {loading && (
          <div className="chat-msg assistant">
            <div className="chat-bubble chat-loading">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="chat-input-row">
        <input
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
          placeholder="Type a question about the document..."
        />
        <button className="chat-send" onClick={send} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [pdfFile, setPdfFile] = useState(null);
  const [pdfDoc, setPdfDoc] = useState(null);
  const [pdfId, setPdfId] = useState(null);
  const [totalPages, setTotalPages] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [scale, setScale] = useState(0.9);

  const [sections, setSections] = useState([]);
  const [caseInfo, setCaseInfo] = useState(null);
  const [coverageMap, setCoverageMap] = useState({});
  const [activeIdx, setActiveIdx] = useState(null);
  const [indexedRange, setIndexedRange] = useState(null);
  const [indexStartPage, setIndexStartPage] = useState(1);
  const [indexEndPage, setIndexEndPage] = useState("");
  const [workflowStatus, setWorkflowStatus] = useState("");
  const [retrievalStatus, setRetrievalStatus] = useState("");
  const [pendingPages, setPendingPages] = useState(0);
  const [chatReady, setChatReady] = useState(false);
  const [pdfSearch, setPdfSearch] = useState("");
  const [savedPdfs, setSavedPdfs] = useState([]);
  const [queueSnapshot, setQueueSnapshot] = useState(EMPTY_QUEUE_SNAPSHOT);
  const [selectedSavedPdfId, setSelectedSavedPdfId] = useState("");
  const [timingEvents, setTimingEvents] = useState([]);
  const [batchReports, setBatchReports] = useState([]);
  const [showTimingPanel, setShowTimingPanel] = useState(false);
  const [viewerTimingEvent, setViewerTimingEvent] = useState(null);

  const [tab, setTab] = useState("index");
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [error, setError] = useState("");
  const [batchSearch, setBatchSearch] = useState("");
  const [batchFilter, setBatchFilter] = useState("");
  const [chatMessages, setChatMessages] = useState([
    { role: "assistant", text: "Ask me anything about this document. I can answer in Hindi or English." },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [showAddPanel, setShowAddPanel] = useState(true);
  const [editingIdx, setEditingIdx] = useState(null);
  const [confirmIdx, setConfirmIdx] = useState(null);
  const [confirmDeletePdf, setConfirmDeletePdf] = useState(null);
  const [showQueueModal, setShowQueueModal] = useState(false);
  const [auditRowStart, setAuditRowStart] = useState("1");
  const [auditRowEnd, setAuditRowEnd] = useState("");
  const [indexSpotlight, setIndexSpotlight] = useState(false);
  const handledIndexRunnerRef = useRef("");
  const handledReviewRunnerRef = useRef("");
  const activeLoadRequestRef = useRef(0);
  const indexSectionRef = useRef(null);
  const indexSpotlightTimeoutRef = useRef(null);
  const pdfCacheRef = useRef(new Map());
  const apiSupportRef = useRef({
    queues: true,
    batchReports: true,
  });

  const rebuildCoverage = useCallback((items, total) => {
    const cmap = {};
    items.forEach((sec, idx) => {
      for (let page = sec.pageFrom; page <= Math.min(sec.pageTo, total); page += 1) {
        cmap[page] = idx;
      }
    });
    setCoverageMap(cmap);
  }, []);

  useEffect(() => {
    if (coverageMap[currentPage] !== undefined) setActiveIdx(coverageMap[currentPage]);
    else setActiveIdx(null);
  }, [currentPage, coverageMap]);

  useEffect(() => () => {
    if (indexSpotlightTimeoutRef.current) {
      window.clearTimeout(indexSpotlightTimeoutRef.current);
    }
    Array.from(pdfCacheRef.current.values()).forEach((entry) => {
      if (entry?.objectUrl) {
        try {
          URL.revokeObjectURL(entry.objectUrl);
        } catch (_) {}
      }
    });
    pdfCacheRef.current.clear();
  }, []);

  const handlePageVisible = useCallback((page) => setCurrentPage(page), []);

  const goToPage = (pageNum, secIdx = null) => {
    const page = Math.max(1, Math.min(pageNum, totalPages));
    setCurrentPage(page);
    document.getElementById(`pg-${page}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    if (secIdx !== null) setActiveIdx(secIdx);
  };

  const touchCachedPdf = useCallback((targetPdfId) => {
    const cache = pdfCacheRef.current;
    const cached = cache.get(targetPdfId);
    if (!cached) return null;
    cache.delete(targetPdfId);
    cached.lastUsedAt = Date.now();
    cache.set(targetPdfId, cached);
    return cached;
  }, []);

  const storeCachedPdf = useCallback((targetPdfId, payload) => {
    const cache = pdfCacheRef.current;
    if (cache.has(targetPdfId)) {
      const previous = cache.get(targetPdfId);
      if (previous?.objectUrl && previous.objectUrl !== payload.objectUrl) {
        try {
          URL.revokeObjectURL(previous.objectUrl);
        } catch (_) {}
      }
      cache.delete(targetPdfId);
    }

    cache.set(targetPdfId, {
      ...payload,
      lastUsedAt: Date.now(),
    });

    while (cache.size > SAVED_PDF_MEMORY_CACHE_LIMIT) {
      const oldestKey = cache.keys().next().value;
      const oldest = cache.get(oldestKey);
      if (oldest?.objectUrl) {
        try {
          URL.revokeObjectURL(oldest.objectUrl);
        } catch (_) {}
      }
      cache.delete(oldestKey);
    }
  }, []);

  const removeCachedPdf = useCallback((targetPdfId) => {
    const cache = pdfCacheRef.current;
    const cached = cache.get(targetPdfId);
    if (cached?.objectUrl) {
      try {
        URL.revokeObjectURL(cached.objectUrl);
      } catch (_) {}
    }
    cache.delete(targetPdfId);
  }, []);

  const persistSections = async (nextSections) => {
    if (!pdfId) return nextSections;
    const resp = await apiFetch(`/api/pdfs/${pdfId}/index`, {
      method: "POST",
      body: JSON.stringify({ index: nextSections }),
    });
    return normalizeSections(resp.index || nextSections, caseInfo?.batchNo || "");
  };

  const handleSaveEdit = async (updated) => {
    const normalizedUpdated = applyCatalogClassification({
      ...updated,
      pdfPageFrom: updated.pageFrom,
      pdfPageTo: updated.pageTo,
      tocPageFrom: "",
      tocPageTo: "",
    });
    const next = sections
      .map((section, idx) => (idx === editingIdx ? normalizedUpdated : section))
      .sort((a, b) => a.pageFrom - b.pageFrom);
    const persisted = await persistSections(next);
    setSections(persisted);
    rebuildCoverage(persisted, totalPages);
    setEditingIdx(null);
  };

  const handleDelete = async (idx) => {
    const next = sections.filter((_, sectionIdx) => sectionIdx !== idx);
    const persisted = await persistSections(next);
    setSections(persisted);
    rebuildCoverage(persisted, totalPages);
    setConfirmIdx(null);
    if (activeIdx === idx) setActiveIdx(null);
  };

  const handleAdd = async (newSection) => {
    const next = [
      ...sections,
      applyCatalogClassification({
        ...newSection,
        pdfPageFrom: newSection.pageFrom,
        pdfPageTo: newSection.pageTo,
        tocPageFrom: "",
        tocPageTo: "",
      }),
    ].sort((a, b) => a.pageFrom - b.pageFrom);
    const persisted = await persistSections(next);
    setSections(persisted);
    rebuildCoverage(persisted, totalPages);
  };

  const applyBatchSearch = () => setBatchFilter(batchSearch.trim());

  const fetchSavedPdfs = useCallback(async (query = "") => {
    try {
      const resp = await apiFetch(`/api/pdfs${query ? `?search=${encodeURIComponent(query)}` : ""}`);
      setSavedPdfs(resp.pdfs || []);
    } catch (err) {
      console.error(err);
    }
  }, []);

  const fetchQueues = useCallback(async () => {
    if (!apiSupportRef.current.queues) {
      setQueueSnapshot(EMPTY_QUEUE_SNAPSHOT);
      return;
    }
    try {
      const resp = await apiFetch("/api/queues");
      setQueueSnapshot(resp || EMPTY_QUEUE_SNAPSHOT);
    } catch (err) {
      const msg = String(err?.message || "");
      if (msg.includes("/api/queues") && msg.toLowerCase().includes("not found")) {
        apiSupportRef.current.queues = false;
        setQueueSnapshot(EMPTY_QUEUE_SNAPSHOT);
        return;
      }
      console.error(err);
    }
  }, []);

  const fetchPdfTimings = useCallback(async (targetPdfId = "") => {
    if (!targetPdfId) {
      setTimingEvents([]);
      setViewerTimingEvent(null);
      return;
    }
    try {
      const resp = await apiFetch(`/api/pdfs/${targetPdfId}/timings`);
      setTimingEvents(resp.events || []);
    } catch (err) {
      console.error(err);
      setTimingEvents([]);
    }
  }, []);

  const fetchBatchReports = useCallback(async () => {
    if (!apiSupportRef.current.batchReports) {
      setBatchReports([]);
      return;
    }
    try {
      const resp = await apiFetch('/api/batch-reports?limit=8');
      setBatchReports(resp.reports || []);
    } catch (err) {
      const msg = String(err?.message || "");
      if (msg.includes("/api/batch-reports") && msg.toLowerCase().includes("not found")) {
        apiSupportRef.current.batchReports = false;
        setBatchReports([]);
        return;
      }
      console.error(err);
      setBatchReports([]);
    }
  }, []);

  const loadSavedPdf = useCallback(async (targetPdfId) => {
    if (!targetPdfId) return;
    const requestId = activeLoadRequestRef.current + 1;
    activeLoadRequestRef.current = requestId;
    const loadStartedAt = performance.now();
    setError("");
    const cached = touchCachedPdf(targetPdfId);
    if (!cached) {
      setLoading(true);
      setLoadingStep("Loading saved PDF from backend...");
    }
    try {
      let details = cached?.details;
      let objectUrl = cached?.objectUrl;
      let doc = cached?.pdfDoc;

      if (!cached) {
        const [fetchedDetails, blob, pdfjsLib] = await Promise.all([
          apiFetch(`/api/pdfs/${targetPdfId}`),
          apiFetchBlob(`/api/pdfs/${targetPdfId}/file`),
          loadPdfJs(),
        ]);
        if (activeLoadRequestRef.current !== requestId) {
          return;
        }
        details = fetchedDetails;
        objectUrl = URL.createObjectURL(blob);
        const arrayBuf = await blob.arrayBuffer();
        doc = await pdfjsLib.getDocument({ data: arrayBuf.slice(0) }).promise;
        if (activeLoadRequestRef.current !== requestId) {
          try {
            URL.revokeObjectURL(objectUrl);
          } catch (_) {}
          return;
        }
        storeCachedPdf(targetPdfId, {
          details,
          objectUrl,
          pdfDoc: doc,
        });
      }

      if (activeLoadRequestRef.current !== requestId) {
        return;
      }

      const pdf = details?.pdf || {};
      const nextCaseInfo = deriveDocumentMeta(pdf.filename || targetPdfId);
      resetDocumentState(nextCaseInfo);
      setPdfDoc(doc);
      setPdfFile({ name: pdf.filename || `${targetPdfId}.pdf`, objectUrl, source: "saved", pdfId: targetPdfId });
      setPdfId(targetPdfId);
      setSelectedSavedPdfId(targetPdfId);
      setTotalPages(pdf.total_pages || doc.numPages);
      setCurrentPage(1);
      setIndexStartPage(String(pdf.selected_start_page || 1));
      setIndexEndPage(String(pdf.selected_end_page || doc.numPages));
      setWorkflowStatus(pdf.status || "");
      setRetrievalStatus(pdf.retrieval_status || "");
      setPendingPages(pdf.pending_pages || 0);
      setChatReady(Boolean(pdf.chat_ready));
      setIndexedRange({
        start: pdf.selected_start_page || 1,
        end: pdf.selected_end_page || doc.numPages,
        count: pdf.indexed_pages || 0,
      });
      const normalized = normalizeSections(details?.index || [], nextCaseInfo.batchNo || "");
      setSections(normalized);
      rebuildCoverage(normalized, pdf.total_pages || doc.numPages);
      setViewerTimingEvent({
        recorded_at: new Date().toISOString(),
        run_label: cached ? "viewer cache load" : "viewer load",
        total_seconds: Number(((performance.now() - loadStartedAt) / 1000).toFixed(3)),
        timings: {
          saved_pdf_load_time: Number(((performance.now() - loadStartedAt) / 1000).toFixed(3)),
        },
      });
      fetchPdfTimings(targetPdfId);
    } catch (err) {
      console.error(err);
      if (activeLoadRequestRef.current === requestId) {
        setError(`Could not open the selected PDF smoothly. ${err.message}`);
      }
    } finally {
      if (!cached && activeLoadRequestRef.current === requestId) {
        setLoading(false);
        setLoadingStep("");
      }
    }
  }, [fetchPdfTimings, rebuildCoverage, storeCachedPdf, touchCachedPdf]);

  const spotlightIndexSection = useCallback(() => {
    if (indexSpotlightTimeoutRef.current) {
      window.clearTimeout(indexSpotlightTimeoutRef.current);
    }
    setIndexSpotlight(true);
    window.requestAnimationFrame(() => {
      indexSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    indexSpotlightTimeoutRef.current = window.setTimeout(() => setIndexSpotlight(false), 1800);
  }, []);

  const jumpToIndexView = useCallback(async (targetPdfId = "") => {
    const targetRecord = savedPdfs.find((item) => item.pdf_id === (targetPdfId || pdfId));
    if (targetPdfId && targetPdfId !== pdfId) {
      await loadSavedPdf(targetPdfId);
    }
    if (targetRecord && !targetRecord.index_ready) {
      setError("Index is not finalized yet for this PDF. It is still queued for full-document indexing or review.");
    }
    setTab("index");
    setShowQueueModal(false);
    window.setTimeout(() => spotlightIndexSection(), 120);
  }, [loadSavedPdf, pdfId, savedPdfs, spotlightIndexSection]);

  useEffect(() => {
    fetchSavedPdfs();
    fetchQueues();
    fetchBatchReports();
    const timer = setInterval(() => {
      fetchSavedPdfs(pdfSearch);
      if (apiSupportRef.current.queues) fetchQueues();
      if (apiSupportRef.current.batchReports) fetchBatchReports();
      if (selectedSavedPdfId) {
        fetchPdfTimings(selectedSavedPdfId);
      }
    }, 3000);
    return () => clearInterval(timer);
  }, [fetchBatchReports, fetchPdfTimings, fetchSavedPdfs, fetchQueues, pdfSearch, selectedSavedPdfId]);

  useEffect(() => {
    const indexRunner = queueSnapshot.index_runner || {};
    const finishedPdfId = indexRunner.finished_pdf_id || "";
    if (!finishedPdfId || indexRunner.running || handledIndexRunnerRef.current === finishedPdfId) return;
    handledIndexRunnerRef.current = finishedPdfId;

    if (pdfId && finishedPdfId === pdfId) {
      apiFetch(`/api/pdfs/${finishedPdfId}`)
        .then((details) => {
          const pdf = details.pdf || {};
          const normalized = normalizeSections(details.index || [], caseInfo?.batchNo || "");
          setSections(normalized);
          rebuildCoverage(normalized, totalPages || pdf.total_pages || 0);
          setWorkflowStatus(pdf.status || "");
          setRetrievalStatus(pdf.retrieval_status || "");
          setPendingPages(pdf.pending_pages || 0);
          setChatReady(Boolean(pdf.chat_ready));
          setIndexedRange({
            start: pdf.selected_start_page || 1,
            end: pdf.selected_end_page || (pdf.total_pages || totalPages || 1),
            count: pdf.indexed_pages || 0,
          });
        })
        .catch((err) => setError(`Error: ${err.message}`));
    }

    fetchSavedPdfs(pdfSearch);
    fetchQueues();
  }, [queueSnapshot.index_runner, pdfId, caseInfo?.batchNo, rebuildCoverage, totalPages, fetchSavedPdfs, fetchQueues, pdfSearch]);

  useEffect(() => {
    const reindexRunner = queueSnapshot.reindex_runner || {};
    const runnerKey = `${reindexRunner.running}-${reindexRunner.processed || 0}-${reindexRunner.fixed || 0}-${reindexRunner.heartbeat_ts || 0}`;
    if (reindexRunner.running || (reindexRunner.processed || 0) === 0 || handledReviewRunnerRef.current === runnerKey) return;
    handledReviewRunnerRef.current = runnerKey;

    if (pdfId) {
      apiFetch(`/api/pdfs/${pdfId}`)
        .then((details) => {
          const pdf = details.pdf || {};
          const normalized = normalizeSections(details.index || [], caseInfo?.batchNo || "");
          setSections(normalized);
          rebuildCoverage(normalized, totalPages || pdf.total_pages || 0);
          setWorkflowStatus(pdf.status || "");
          setRetrievalStatus(pdf.retrieval_status || "");
          setPendingPages(pdf.pending_pages || 0);
          setChatReady(Boolean(pdf.chat_ready));
          setIndexedRange({
            start: pdf.selected_start_page || 1,
            end: pdf.selected_end_page || (pdf.total_pages || totalPages || 1),
            count: pdf.indexed_pages || 0,
          });
        })
        .catch((err) => setError(`Error: ${err.message}`));
    }

    fetchSavedPdfs(pdfSearch);
    fetchQueues();
  }, [queueSnapshot.reindex_runner, pdfId, caseInfo?.batchNo, rebuildCoverage, totalPages, fetchSavedPdfs, fetchQueues, pdfSearch]);

  const handleBatchUpload = async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setError("");
    setLoading(true);
    try {
      let firstQueuedPdfId = "";
      for (let idx = 0; idx < files.length; idx += BATCH_UPLOAD_CHUNK_SIZE) {
        const chunk = files.slice(idx, idx + BATCH_UPLOAD_CHUNK_SIZE);
        setLoadingStep(`Queueing PDFs ${idx + 1}-${Math.min(files.length, idx + chunk.length)} of ${files.length} for batch indexing...`);
        const formData = new FormData();
        chunk.forEach((file) => formData.append("files", file));
        formData.append("start_page", "1");
        formData.append("end_page", "10");
        const resp = await apiUpload("/api/stage1-batch/enqueue", formData);
        if (!firstQueuedPdfId) {
          const firstQueued = (resp.pdfs || []).find((item) => item.pdf_id && !item.skipped_duplicate);
          firstQueuedPdfId = firstQueued?.pdf_id || "";
        }
        await fetchQueues();
      }
      await fetchSavedPdfs(pdfSearch);
      await fetchQueues();
      if (firstQueuedPdfId) {
        await loadSavedPdf(firstQueuedPdfId);
      }
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
      e.target.value = "";
    }
  };

  const resetDocumentState = (nextCaseInfo) => {
    setSections([]);
    setCoverageMap({});
    setIndexedRange(null);
    setActiveIdx(null);
    setEditingIdx(null);
    setConfirmIdx(null);
    setShowAddPanel(true);
    setWorkflowStatus("");
    setRetrievalStatus("");
    setPendingPages(0);
    setChatReady(false);
    setSelectedSavedPdfId("");
    setPdfId(null);
    setTab("index");
    setChatMessages([
      { role: "assistant", text: "Ask me anything about this document. I can answer in Hindi or English." },
    ]);
    setChatInput("");
    setCaseInfo(nextCaseInfo);
    setBatchSearch(nextCaseInfo.batchNo || "");
    setBatchFilter(nextCaseInfo.batchNo || "");
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const nextCaseInfo = deriveDocumentMeta(file.name);

    setError("");
    setLoading(true);
    setLoadingStep("Loading PDF preview...");
    setPdfFile(file);
    resetDocumentState(nextCaseInfo);

    try {
      const pdfjsLib = await loadPdfJs();
      const arrayBuf = await file.arrayBuffer();
      const doc = await pdfjsLib.getDocument({ data: arrayBuf.slice(0) }).promise;
      setPdfDoc(doc);
      setTotalPages(doc.numPages);
      setCurrentPage(1);
      setIndexStartPage("1");
      setIndexEndPage(String(doc.numPages));
    } catch (err) {
      console.error(err);
      setPdfDoc(null);
      setPdfFile(null);
      setTotalPages(0);
      setError(getFetchErrorMessage(err));
    } finally {
      setLoading(false);
      setLoadingStep("");
      e.target.value = "";
    }
  };

  const startIndexing = async () => {
    if (!pdfFile || !pdfDoc) {
      setError("Upload a PDF before starting indexing.");
      return;
    }

    const start = Math.max(1, Math.min(parseInt(indexStartPage, 10) || 1, totalPages));
    const end = Math.max(start, Math.min(parseInt(indexEndPage, 10) || totalPages, totalPages));

    setError("");
    setLoading(true);

    try {
      setLoadingStep("Checking backend connection...");
      await apiFetch("/health");

      let resp;
      let activePdfId = pdfId;
      if (isRealFile(pdfFile)) {
        setLoadingStep(`Uploading and extracting pages for ${totalPages} pages...`);
        const formData = new FormData();
        formData.append("file", pdfFile);
        formData.append("start_page", String(start));
        formData.append("end_page", String(end));
        resp = await apiUpload("/api/ingest", formData);
        activePdfId = resp.pdf_id;
        setPdfId(activePdfId);
      } else if (pdfId) {
        activePdfId = pdfId;
      } else {
        throw new Error("No upload source found for this PDF.");
      }

      if (!activePdfId) {
        throw new Error("Backend did not return a PDF ID.");
      }

      setLoadingStep("Building document index...");
      const indexResp = await apiFetch("/api/generate-index", {
        method: "POST",
        body: JSON.stringify({ pdf_id: activePdfId }),
      });
      const normalized = normalizeSections(indexResp.index || [], caseInfo?.batchNo || "");
      setSections(normalized);
      rebuildCoverage(normalized, totalPages);
      setIndexedRange({
        start: indexResp.indexed_page_start,
        end: indexResp.indexed_page_end,
        count: indexResp.indexed_pages,
      });
      setWorkflowStatus(indexResp.status || resp?.status || "index_ready");
      setRetrievalStatus(indexResp.retrieval_status || resp?.retrieval_status || "");
      setPendingPages(indexResp.pending_pages ?? resp?.pending_pages ?? 0);
      setChatReady(Boolean(indexResp.chat_ready ?? resp?.chat_ready));
      setBatchFilter((value) => value || caseInfo?.batchNo || "");
      await fetchSavedPdfs(pdfSearch);
      fetchQueues().catch(() => {});
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };
  const regenIndex = async () => {
    if (!pdfId) return;
    setLoading(true);
    setLoadingStep("Re-building index from stored vectors...");
    try {
      const resp = await apiFetch("/api/generate-index", {
        method: "POST",
        body: JSON.stringify({ pdf_id: pdfId }),
      });
      const normalized = normalizeSections(resp.index || [], caseInfo?.batchNo || "");
      setSections(normalized);
      rebuildCoverage(normalized, totalPages);
      setIndexedRange({
        start: resp.indexed_page_start,
        end: resp.indexed_page_end,
        count: resp.indexed_pages,
      });
      setWorkflowStatus(resp.status || workflowStatus);
      setRetrievalStatus(resp.retrieval_status || retrievalStatus);
      setPendingPages(resp.pending_pages ?? pendingPages);
      setChatReady(Boolean(resp.chat_ready ?? chatReady));
      fetchSavedPdfs(pdfSearch);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const processPendingPages = async () => {
    if (!pdfId) return;
    setError("");
    setLoading(true);
    setLoadingStep(`Legacy deferred ingestion - processing ${pendingPages} pending pages...`);
    try {
      const resp = await apiFetch(`/api/process-pending/${pdfId}`, { method: "POST" });
      setWorkflowStatus(resp.status || "vectorized");
      setRetrievalStatus(resp.retrieval_status || "vectorized");
      setPendingPages(resp.pending_pages || 0);
      setChatReady(Boolean(resp.chat_ready));
      fetchSavedPdfs(pdfSearch);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const saveToDeferredQueue = async () => {
    if (!pdfId) return;
    setError("");
    try {
      await apiUpload(`/api/pdfs/${pdfId}/deferred-choice`, (() => {
        const formData = new FormData();
        formData.append("choice", "queue");
        return formData;
      })());
      setRetrievalStatus("queued_for_full_ingestion");
      fetchSavedPdfs(pdfSearch);
      if (apiSupportRef.current.queues) fetchQueues();
    } catch (err) {
      setError(err.message);
    }
  };

  const forceResetQueue = async (queueName) => {
    setError("");
    setLoading(true);
    setLoadingStep(`Force resetting the ${queueName} queue...`);
    try {
      const formData = new FormData();
      formData.append("queue_name", queueName);
      const resp = await apiUpload("/api/queues/reset", formData);
      await fetchSavedPdfs(pdfSearch);
      await fetchQueues();
      setError(resp.reset_count > 0 ? "" : `No PDFs needed a reset in the ${queueName} queue.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const deleteSavedPdf = async (item) => {
    if (!item?.pdf_id) return;
    const isDeferredBusy = queueSnapshot.runner?.running && queueSnapshot.runner?.current_pdf_id === item.pdf_id;
    const isIndexBusy = queueSnapshot.index_runner?.running && queueSnapshot.index_runner?.current_pdf_id === item.pdf_id;
    if (isDeferredBusy || isIndexBusy) {
      setError("This PDF is currently being processed. Stop or wait for the running job before deleting it.");
      return;
    }

    setError("");
    setLoading(true);
    setLoadingStep(`Deleting ${item.filename || item.pdf_id} from backend...`);
    try {
      await apiFetch(`/api/pdfs/${item.pdf_id}`, { method: "DELETE" });
      removeCachedPdf(item.pdf_id);
      if (selectedSavedPdfId === item.pdf_id || pdfId === item.pdf_id) {
        if (pdfFile?.objectUrl) {
          try {
            URL.revokeObjectURL(pdfFile.objectUrl);
          } catch (_) {}
        }
        setPdfDoc(null);
        setPdfFile(null);
        setPdfId(null);
        setTotalPages(0);
        setCurrentPage(1);
        setSections([]);
        setCoverageMap({});
        setIndexedRange(null);
        setActiveIdx(null);
        setWorkflowStatus("");
        setRetrievalStatus("");
        setPendingPages(0);
        setChatReady(false);
        setSelectedSavedPdfId("");
        setTab("index");
      }
      await fetchSavedPdfs(pdfSearch);
      await fetchQueues();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
      setConfirmDeletePdf(null);
    }
  };

  const runAllDeferredQueue = async () => {
    setError("");
    setLoading(true);
    setLoadingStep("Starting deferred queue in the background...");
    try {
      const resp = await apiFetch("/api/process-pending-runner", { method: "POST" });
      await fetchQueues();
      await fetchSavedPdfs(pdfSearch);
      if (!resp.started && resp.message) {
        setError(resp.message);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const controlDeferredQueue = async (action) => {
    setError("");
    setLoading(true);
    setLoadingStep(action === "stop" ? "Stopping deferred queue after current PDF..." : "Resuming deferred queue...");
    try {
      const formData = new FormData();
      formData.append("action", action);
      const resp = await apiUpload("/api/process-pending-runner/control", formData);
      await fetchQueues();
      await fetchSavedPdfs(pdfSearch);
      if (resp.message && !resp.accepted) {
        setError(resp.message);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const startIndexAudit = async () => {
    setError("");
    setLoading(true);
    setLoadingStep("Auditing saved indexes for the current library filters...");
    try {
      const formData = new FormData();
      formData.append("search", pdfSearch.trim());
      formData.append("batch_filter", batchFilter.trim());
      formData.append("row_start", String(Math.max(1, parseInt(auditRowStart, 10) || 1)));
      if ((auditRowEnd || "").trim()) formData.append("row_end", String(Math.max(1, parseInt(auditRowEnd, 10) || 1)));
      const resp = await apiUpload("/api/index-audit-runner", formData);
      await fetchQueues();
      await fetchSavedPdfs(pdfSearch);
      if (!resp.started && resp.message) setError(resp.message);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const stopAllProcessing = async () => {
    setError("");
    setLoading(true);
    setLoadingStep("Requesting all background runners to stop safely...");
    try {
      const formData = new FormData();
      formData.append("action", "stop");
      const resp = await apiUpload("/api/runners/control-all", formData);
      await fetchQueues();
      await fetchSavedPdfs(pdfSearch);
      if (resp.message && !resp.accepted) setError(resp.message);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const runReviewReindexQueue = async () => {
    setError("");
    setLoading(true);
    setLoadingStep("Reindexing the review queue using saved vectorized page text...");
    try {
      const resp = await apiFetch("/api/reindex-review-runner", { method: "POST" });
      await fetchQueues();
      await fetchSavedPdfs(pdfSearch);
      if (!resp.started && resp.message) setError(resp.message);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const srcBadge = (src) => {
    if (src === "toc" || src === "toc-image") return "badge-toc";
    if (src === "auto") return "badge-auto";
    if (src === "manual") return "badge-manual";
    return "badge-gap";
  };

  const srcLabel = (src) => {
    if (src === "toc" || src === "toc-image") return "TOC";
    if (src === "auto") return "AI";
    if (src === "manual") return "MAN";
    return "GAP";
  };

  const runner = queueSnapshot.runner || {};
  const indexRunner = queueSnapshot.index_runner || {};
  const stage1BatchRunner = queueSnapshot.stage1_batch_runner || {};
  const auditRunner = queueSnapshot.audit_runner || {};
  const reindexRunner = queueSnapshot.reindex_runner || {};
  const heartbeatAgeSeconds = runner.heartbeat_ts ? Math.max(0, Math.round(Date.now() / 1000 - runner.heartbeat_ts)) : 0;
  const runnerLooksStuck = Boolean(runner.running && heartbeatAgeSeconds > 90);
  const runnerStopping = Boolean(runner.running && runner.pause_requested);
  const runnerStateLabel = runnerLooksStuck ? "Stuck" : runner.paused ? "Paused" : runnerStopping ? "Stopping" : runner.running ? "Running" : "Idle";
  const selectedSavedPdf = savedPdfs.find((item) => item.pdf_id === selectedSavedPdfId) || (pdfId ? {
    pdf_id: pdfId,
    status: workflowStatus,
    retrieval_status: retrievalStatus,
    pending_pages: pendingPages,
    chat_ready: chatReady,
    index_ready: workflowStatus === "index_ready",
  } : null);
  const activeTone = getSavedPdfTone(selectedSavedPdf || {});
  const currentStepLabel = getCurrentStepLabel(selectedSavedPdf || {});
  const anyRunnerStopping = runnerStopping || indexRunner.status === "stopping" || stage1BatchRunner.status === "stopping" || auditRunner.status === "stopping" || reindexRunner.status === "stopping";
  const anyRunnerPaused = runner.paused;
  const systemStatusLabel = runnerLooksStuck ? "stuck" : anyRunnerStopping ? "stopping" : anyRunnerPaused ? "paused" : (runner.running || indexRunner.running || stage1BatchRunner.running || auditRunner.running || reindexRunner.running) ? "running" : "idle";
  const indexedPdfCount = savedPdfs.filter((item) => getSavedPdfTone(item) === "indexed" || getSavedPdfTone(item) === "staged").length;
  const vectorizedPdfCount = savedPdfs.filter((item) => getSavedPdfTone(item) === "ready").length;
  const reviewPdfCount = savedPdfs.filter((item) => getSavedPdfTone(item) === "review").length;

  const covered = Object.keys(coverageMap).length;
  const applicantName = caseInfo?.plaintiff
    ? `${caseInfo.plaintiff} Vs ${caseInfo.defendant || "-"}`
    : caseInfo?.title || pdfFile?.name?.replace(".pdf", "").toUpperCase() || "";
  const visibleSections = sections
    .map((sec, idx) => ({ sec, idx }))
    .filter(({ sec }) => !batchFilter || (sec.batchNo || "").toLowerCase().includes(batchFilter.toLowerCase()));

  return (
    <div className="app-root">
      <nav className="top-nav">
        <div className="nav-left">
          <div className="nav-logo">CF</div>
          <span className="nav-title">Court File Indexer</span>
          {caseInfo?.court && <span className="nav-court">{caseInfo.court}</span>}
        </div>
        <div className="nav-right">
          <div className="batch-search-wrap">
            <span className="batch-label">PDF Search</span>
            <input
              className="batch-input"
              value={pdfSearch}
              onChange={(e) => setPdfSearch(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && fetchSavedPdfs(pdfSearch)}
              placeholder="Search by CNR / filename..."
            />
            <button className="btn-search" onClick={() => fetchSavedPdfs(pdfSearch)}>Fetch</button>
          </div>
          <div className="batch-search-wrap">
            <span className="batch-label">Batch No.</span>
            <input
              className="batch-input"
              value={batchSearch}
              onChange={(e) => setBatchSearch(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && applyBatchSearch()}
              placeholder="Filter rows by batch..."
            />
            <button className="btn-search" onClick={applyBatchSearch}>Search</button>
          </div>
          <label className="btn-upload">
            Upload PDF
            <input type="file" accept=".pdf" style={{ display: "none" }} onChange={handleUpload} />
          </label>
          <label className="btn-upload">
            Batch Upload
            <input type="file" accept=".pdf" multiple style={{ display: "none" }} onChange={handleBatchUpload} />
          </label>
          {pdfFile && <span className="nav-filename">{pdfFile.name} - {totalPages}p</span>}
          {sections.length > 0 && (
            <span className="coverage-pill">OK {covered}/{totalPages} pages - {sections.length} entries</span>
          )}
        </div>
      </nav>

      {loading && (
        <div className="loading-bar">
          <div className="loading-bar-inner" />
          <span className="loading-text">{loadingStep}</span>
        </div>
      )}
      {error && <div className="error-bar">Warning: {error}</div>}

      <div className="main-body">
        <div className={`right-panel index-panel tone-shell-${activeTone}`}>
          <div className="workspace-header">
            <div className="section-card library-card">
              <div className="section-card-head library-card-head">
                <div>
                  <div className="section-title">Saved PDF Library</div>
                  <div className="section-subtitle">Open a file, jump straight to indexing, and keep queue controls tucked away until you need them.</div>
                </div>
                <div className="library-head-actions">
                  <button className="queue-launch-btn" onClick={() => setShowQueueModal(true)}>
                    Operations Panel
                    {(queueSnapshot.runner?.running || queueSnapshot.runner?.paused || indexRunner.running || stage1BatchRunner.running) && <span className="queue-launch-dot" />}
                  </button>
                  <div className="section-count">{savedPdfs.length} saved PDFs</div>
                </div>
              </div>
              <div className="library-summary-strip">
                <span className="queue-pill">Indexed: {indexedPdfCount}</span>
                <span className="queue-pill success">Vectorized: {vectorizedPdfCount}</span>
                <span className="queue-pill">Batch Queue: {(queueSnapshot.stage1_batch || []).length}</span>
                <span className="queue-pill">Pending Queue: {(queueSnapshot.pending_vectorization || []).length}</span>
                <span className="queue-pill warning">Review Queue: {reviewPdfCount}</span>
              </div>
              <div className="library-help subtle">
                Open any saved PDF directly. Queue numbers, review reasons, and current step labels help new operators understand what is happening without opening backend logs.
              </div>
              <div className="saved-pdf-list">
                {savedPdfs.map((item, itemIndex) => {
                  const isDeferredBusy = queueSnapshot.runner?.running && queueSnapshot.runner?.current_pdf_id === item.pdf_id;
                  const isIndexBusy = queueSnapshot.index_runner?.running && queueSnapshot.index_runner?.current_pdf_id === item.pdf_id;
                  const isStage1BatchBusy = queueSnapshot.stage1_batch_runner?.running && queueSnapshot.stage1_batch_runner?.current_pdf_id === item.pdf_id;
                  const deleteDisabled = loading || isDeferredBusy || isIndexBusy || isStage1BatchBusy;
                  const indexViewDisabled = isIndexBusy && !item.index_ready;
                  const cardTone = getSavedPdfTone(item);
                  const batchQueuePosition = (queueSnapshot.stage1_batch || []).findIndex((queued) => queued.pdf_id === item.pdf_id);
                  const reviewQueuePosition = (queueSnapshot.reindex_review || []).findIndex((queued) => queued.pdf_id === item.pdf_id);
                  return (
                    <div
                      key={item.pdf_id}
                      className={`saved-pdf-card tone-${cardTone} ${item.pdf_id === selectedSavedPdfId ? "selected" : ""}`}
                    >
                      <div className="saved-pdf-row">
                        <div className="saved-pdf-heading">
                          <div className="saved-pdf-title">#{itemIndex + 1} - {item.cnr_number || item.filename}</div>
                          <span className={`saved-pdf-tone-pill tone-pill-${cardTone}`}>{getSavedPdfToneLabel(cardTone)}</span>
                        </div>
                        <button
                          type="button"
                          className="saved-pdf-delete"
                          disabled={deleteDisabled}
                          onClick={() => setConfirmDeletePdf(item)}
                          title={deleteDisabled ? "This PDF is currently being processed" : "Delete this PDF and all saved data"}
                        >
                          Delete
                        </button>
                      </div>
                      <div className="saved-pdf-meta">{item.filename} - {item.total_pages || 0} pages</div>
                      <div className="saved-pdf-tags">
                        <span>status: {formatSavedPdfStatus(item)}</span>
                        <span>retrieval: {item.retrieval_status}</span>
                        <span>pending: {item.pending_pages || 0}</span>
                        {batchQueuePosition >= 0 ? <span>batch queue no.: #{batchQueuePosition + 1} from top</span> : null}
                        {reviewQueuePosition >= 0 ? <span>review queue no.: #{reviewQueuePosition + 1} from top</span> : null}
                        {item.review_reason ? <span>review reason: {item.review_reason}</span> : null}
                      </div>
                      <div className="saved-pdf-note">
                        {describeSavedPdfStatus(item, { isDeferredBusy, isIndexBusy, isStage1BatchBusy })}
                      </div>
                      <div className="saved-pdf-actions">
                        <button className="saved-pdf-open" onClick={() => loadSavedPdf(item.pdf_id)}>
                          Open PDF
                        </button>
                        <button className="saved-pdf-index" onClick={() => jumpToIndexView(item.pdf_id)} disabled={indexViewDisabled} title={indexViewDisabled ? "Wait for the current indexing run to finish for this PDF" : "Open index view"}>
                          Index View
                        </button>
                      </div>
                    </div>
                  );
                })}
                {savedPdfs.length === 0 && (
                  <div className="saved-pdf-empty">No saved PDFs yet. Upload one or use batch upload.</div>
                )}
              </div>
            </div>
          </div>

          {caseInfo && (caseInfo.caseNo || applicantName) && (
            <div className="case-header">
              <span className="case-label">Case</span>
              <span className="case-value">{caseInfo.caseNo || "Current file"}</span>
              <span className="case-vs">{applicantName}</span>
            </div>
          )}

          {pdfDoc && (
            <div className={`system-status-strip system-${systemStatusLabel}`}>
              <span className="system-status-label">System Status</span>
              <span className="system-status-pill">{systemStatusLabel}</span>
              <span className="system-status-meta">Current step: {currentStepLabel}</span>
              {selectedSavedPdf?.review_reason ? <span className="system-status-meta">Review reason: {selectedSavedPdf.review_reason}</span> : null}
            </div>
          )}

          {pdfDoc && (
            <div ref={indexSectionRef} className={`indexing-toolbar tone-panel-${activeTone} ${indexSpotlight ? "index-spotlight" : ""}`}>
              <div className="indexing-copy">
                <div className="indexing-title">Indexing Controls</div>
                <div className="indexing-hint">
                  Upload loads the preview first. The backend then extracts and vectorizes the full PDF before building the final index from whole-document evidence and routing weak results to review.
                </div>
              </div>
              <div className="current-step-banner">
                <span className="current-step-label">Current Step</span>
                <span className="current-step-value">{currentStepLabel}</span>
                {selectedSavedPdf?.review_reason ? <span className="current-step-note">{selectedSavedPdf.review_reason}</span> : null}
              </div>
              <div className="workflow-help">
                <span>1. Upload or fetch a PDF.</span>
                <span>2. Click Start Indexing to run full-document OCR, vectorization, and final indexing in the backend.</span>
                <span>3. Weak indexes go to review automatically instead of being finalized silently.</span>
              </div>
              <div className="indexing-form">
                <div className="range-group">
                  <label className="range-label">Start Page</label>
                  <input
                    type="number"
                    min={1}
                    max={Math.max(totalPages, 1)}
                    className="range-input"
                    value={indexStartPage}
                    onChange={(e) => setIndexStartPage(e.target.value)}
                    disabled={!pdfDoc || loading}
                  />
                </div>
                <div className="range-group">
                  <label className="range-label">End Page</label>
                  <input
                    type="number"
                    min={1}
                    max={Math.max(totalPages, 1)}
                    className="range-input"
                    value={indexEndPage}
                    onChange={(e) => setIndexEndPage(e.target.value)}
                    disabled={!pdfDoc || loading}
                  />
                </div>
                <button className="start-index-btn" onClick={startIndexing} disabled={!pdfDoc || loading || indexRunner.running}>
                  {loading ? "Starting..." : indexRunner.running ? "Indexing In Background..." : "Start Indexing"}
                </button>
                {pdfId && pendingPages > 0 && (
                  <button className="start-index-btn" onClick={processPendingPages} disabled={loading}>
                    {loading ? "Processing..." : `Process Pending (${pendingPages})`}
                  </button>
                )}
                {pdfId && pendingPages > 0 && (
                  <button className="add-toggle-btn" onClick={saveToDeferredQueue} disabled={loading}>
                    Save To Queue
                  </button>
                )}
              </div>
              <div className="index-status-row">
                <span className="status-chip">{pdfDoc ? `Loaded ${totalPages} pages` : "No PDF loaded"}</span>
                {workflowStatus && <span className="status-chip">Workflow: {workflowStatus}</span>}
                {retrievalStatus && <span className="status-chip">Retrieval: {retrievalStatus}</span>}
                {indexedRange && (
                  <span className="status-chip success">
                    Indexed {indexedRange.start}-{indexedRange.end} ({indexedRange.count} pages)
                  </span>
                )}
                {pendingPages > 0 && <span className="status-chip">Pending {pendingPages} pages</span>}
                {chatReady && <span className="status-chip success">Chat Ready</span>}
                {pdfId && <span className="status-chip">{pdfId}</span>}
              </div>
            </div>
          )}

          {pdfDoc && (
            <div className={`tab-bar tone-panel-${activeTone}`}>
              <button className={`tab-btn ${tab === "index" ? "active" : ""}`} onClick={() => setTab("index")}>
                Index Table
              </button>
              <button className={`tab-btn ${tab === "review" ? "active" : ""}`} onClick={() => setTab("review")}>
                Review Queue
              </button>
              <button
                className={`tab-btn ${tab === "chat" ? "active" : ""}`}
                onClick={() => pdfId && chatReady && setTab("chat")}
                disabled={!pdfId || !chatReady}
              >
                Document Chat
              </button>
              {tab === "index" && (
                <div className="tab-actions">
                  <button className="add-toggle-btn" onClick={() => setShowAddPanel((value) => !value)}>
                    {showAddPanel ? "Hide" : "+ Add Entry"}
                  </button>
                  {pdfId && (
                    <button className="add-toggle-btn" onClick={regenIndex} disabled={loading}>
                      Re-index
                    </button>
                  )}
                </div>
              )}
            </div>
          )}

          {tab === "index" && (
            <div className={`content-panel index-content-panel tone-panel-${activeTone}`}>
              {showAddPanel && (
                <AddEntryPanel
                  totalPages={totalPages}
                  caseApplicant={applicantName}
                  defaultBatchNo={caseInfo?.batchNo || ""}
                  onAdd={handleAdd}
                  documents={documentCatalog}
                />
              )}

              {sections.length > 0 && visibleSections.length > 0 ? (
                <div className="table-wrap">
                  <table className="index-table">
                    <thead>
                      <tr>
                        <th className="col-num">#</th>
                        <th className="col-doc">Document</th>
                        <th className="col-batch">Batch No.</th>
                        <th className="col-page">From Page</th>
                        <th className="col-page">To Page</th>
                        <th className="col-action">Update</th>
                        <th className="col-action">Delete</th>
                      </tr>
                    </thead>
                    <tbody>
                      {visibleSections.map(({ sec, idx }) => (
                        <React.Fragment key={`row-${idx}-${sec.pageFrom}`}>
                          <tr
                            className={`index-row ${activeIdx === idx ? "active-row" : ""} ${editingIdx === idx ? "editing-row" : ""}`}
                            onClick={() => editingIdx !== idx && goToPage(sec.pageFrom, idx)}
                          >
                            <td className="col-num">{idx + 1}</td>
                            <td className="col-doc">
                              <div className="doc-cell">
                                <span className={`src-badge ${srcBadge(sec.source)}`}>{srcLabel(sec.source)}</span>
                                <span className="doc-name">{sec.displayTitle || sec.title}</span>
                                {sec.displayTitle && sec.displayTitle !== sec.title && (
                                  <span className="mapped-doc">Mapped to: {sec.title}</span>
                                )}
                                {sec.subDocument && <span className="sub-doc">- {sec.subDocument}</span>}
                              </div>
                            </td>
                            <td className="col-batch"><span className="batch-pill">{sec.batchNo || "-"}</span></td>
                            <td className="col-page">{sec.pageFrom || sec.pdfPageFrom || "-"}</td>
                            <td className="col-page">{sec.pageTo || sec.pdfPageTo || "-"}</td>
                            <td className="col-action">
                              <button className="action-btn edit-btn" onClick={(e) => { e.stopPropagation(); setEditingIdx(editingIdx === idx ? null : idx); }}>Edit</button>
                            </td>
                            <td className="col-action">
                              <button className="action-btn delete-btn" onClick={(e) => { e.stopPropagation(); setConfirmIdx(idx); }}>Del</button>
                            </td>
                          </tr>
                          {editingIdx === idx && (
                            <InlineEditRow section={sec} totalPages={totalPages} onSave={handleSaveEdit} onCancel={() => setEditingIdx(null)} />
                          )}
                        </React.Fragment>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="table-empty">
                  {loading ? (
                    <div style={{ textAlign: "center", padding: 40, color: "#6b7280" }}>
                      <div className="pdf-spinner" style={{ margin: "0 auto 12px" }} />
                      <div>{loadingStep}</div>
                    </div>
                  ) : sections.length > 0 ? (
                    <div style={{ textAlign: "center", padding: 40, color: "#9ca3af" }}>
                      <div style={{ fontWeight: 500, marginBottom: 6 }}>No rows match this batch number</div>
                      <div style={{ fontSize: 13 }}>Try another batch filter or clear the search box</div>
                    </div>
                  ) : pdfDoc ? (
                    <div style={{ textAlign: "center", padding: 60, color: "#9ca3af" }}>
                      <div style={{ fontSize: 48, marginBottom: 12 }}>IDX</div>
                      <div style={{ fontWeight: 500, marginBottom: 6 }}>Preview loaded</div>
                      <div style={{ fontSize: 13 }}>Full-document indexing now scans every page before finalizing the index</div>
                    </div>
                  ) : (
                    <div style={{ textAlign: "center", padding: 60, color: "#9ca3af" }}>
                      <div style={{ fontSize: 48, marginBottom: 12 }}>IDX</div>
                      <div style={{ fontWeight: 500, marginBottom: 6 }}>No document loaded</div>
                      <div style={{ fontSize: 13 }}>Upload a PDF to preview it and then start indexing</div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {tab === "review" && (
            <div className="content-panel review-content-panel">
              <div className="review-queue-panel">
                <div className="review-queue-head">
                  <div>
                    <div className="section-title">Review Queue</div>
                    <div className="section-subtitle">Top of queue is shown first with case number, filename, queue number, and review reason.</div>
                  </div>
                  <span className="queue-pill warning">{(queueSnapshot.reindex_review || []).length} waiting</span>
                </div>
                {(queueSnapshot.reindex_review || []).length ? (
                  <div className="review-queue-list">
                    {(queueSnapshot.reindex_review || []).map((item, idx) => (
                      <button key={`review-tab-${item.pdf_id}`} className="review-queue-row" onClick={() => loadSavedPdf(item.pdf_id)}>
                        <span className="review-queue-number">#{idx + 1}</span>
                        <span className="review-queue-case"><strong>Case:</strong> {item.cnr_number || item.pdf_id}</span>
                        <span className="review-queue-file"><strong>File:</strong> {item.filename || item.pdf_id}</span>
                        <span className="review-queue-reason"><strong>Reason:</strong> {item.review_reason || "Needs manual review"}</span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="table-empty">
                    <div style={{ textAlign: "center", padding: 50, color: "#9ca3af" }}>
                      <div style={{ fontSize: 48, marginBottom: 12 }}>REVIEW</div>
                      <div style={{ fontWeight: 500, marginBottom: 6 }}>No PDFs are waiting for review</div>
                      <div style={{ fontSize: 13 }}>Suspicious indexes will appear here automatically.</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {tab === "chat" && pdfId && (
            <div className="content-panel chat-content-panel">
              <ChatPanel
                pdfId={pdfId}
                currentPage={currentPage}
                messages={chatMessages}
                setMessages={setChatMessages}
                input={chatInput}
                setInput={setChatInput}
                onJumpToPage={(page) => goToPage(page)}
              />
            </div>
          )}
          {tab === "chat" && !pdfId && (
            <div className="content-panel chat-content-panel">
              <div className="table-empty">
                <div style={{ textAlign: "center", padding: 60, color: "#9ca3af" }}>
                  <div style={{ fontSize: 48, marginBottom: 12 }}>CHAT</div>
                  <div style={{ fontWeight: 500 }}>Finish deferred ingestion before using chat</div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className={`left-panel viewer-panel viewer-tone-${activeTone}`}>
          {pdfDoc && (
            <div className="viewer-toolbar">
              <span className="toolbar-filename">{pdfFile?.name?.split(".")[0]}</span>
              <div className="toolbar-page"><span>{currentPage} / {totalPages}</span></div>
              <div className="toolbar-zoom">
                <button onClick={() => setScale((value) => Math.max(0.3, +(value - 0.1).toFixed(1)))}>-</button>
                <span>{Math.round(scale * 100)}%</span>
                <button onClick={() => setScale((value) => Math.min(3.0, +(value + 0.1).toFixed(1)))}>+</button>
              </div>
              <button className="toolbar-btn" onClick={() => setScale(0.9)}>Fit</button>
              <button className="toolbar-btn" onClick={() => goToPage(currentPage - 1)} disabled={currentPage <= 1}>Prev</button>
              <button className="toolbar-btn" onClick={() => goToPage(currentPage + 1)} disabled={currentPage >= totalPages}>Next</button>
            </div>
          )}
          <div className="pdf-scroll-area">
            {!pdfDoc && !loading && (
              <div className="pdf-empty">
                <div className="pdf-empty-icon">PDF</div>
                <div className="pdf-empty-title">No document loaded</div>
                <div className="pdf-empty-sub">Upload a PDF to begin</div>
              </div>
            )}
            {loading && !pdfDoc && (
              <div className="pdf-empty">
                <div className="pdf-spinner" />
                <div className="pdf-empty-sub" style={{ marginTop: 12 }}>{loadingStep}</div>
              </div>
            )}
            {pdfDoc && Array.from({ length: totalPages }, (_, idx) => idx + 1).map((pageNum) => (
              <PDFPage
                key={pageNum}
                doc={pdfDoc}
                pageNum={pageNum}
                scale={scale}
                isActive={coverageMap[pageNum] !== undefined && coverageMap[pageNum] === activeIdx}
                onVisible={handlePageVisible}
              />
            ))}
          </div>
        </div>
      </div>

      {confirmIdx !== null && (
        <ConfirmModal
          message={`Delete "${sections[confirmIdx]?.title}"?`}
          onConfirm={() => handleDelete(confirmIdx)}
          onCancel={() => setConfirmIdx(null)}
        />
      )}

      {confirmDeletePdf && (
        <ConfirmModal
          message={`Delete "${confirmDeletePdf.cnr_number || confirmDeletePdf.filename}" and erase its saved index, vectors, cached pages, and stored PDF?`}
          onConfirm={() => deleteSavedPdf(confirmDeletePdf)}
          onCancel={() => setConfirmDeletePdf(null)}
        />
      )}

      <QueueStatusModal
        open={showQueueModal}
        onClose={() => setShowQueueModal(false)}
        savedPdfs={savedPdfs}
        queueSnapshot={queueSnapshot}
        runner={runner}
        runnerStateLabel={runnerStateLabel}
        runnerLooksStuck={runnerLooksStuck}
        runnerStopping={runnerStopping}
        heartbeatAgeSeconds={heartbeatAgeSeconds}
        indexRunner={indexRunner}
        stage1BatchRunner={stage1BatchRunner}
        auditRunner={auditRunner}
        reindexRunner={reindexRunner}
        loading={loading}
        runAllDeferredQueue={runAllDeferredQueue}
        controlDeferredQueue={controlDeferredQueue}
        forceResetQueue={forceResetQueue}
        startIndexAudit={startIndexAudit}
        runReviewReindexQueue={runReviewReindexQueue}
        stopAllProcessing={stopAllProcessing}
        timingEvents={timingEvents}
        viewerTimingEvent={viewerTimingEvent}
        showTimingPanel={showTimingPanel}
        setShowTimingPanel={setShowTimingPanel}
        selectedPdfLabel={selectedSavedPdf?.filename || selectedSavedPdf?.cnr_number || selectedSavedPdf?.pdf_id || ""}
        auditRowStart={auditRowStart}
        auditRowEnd={auditRowEnd}
        setAuditRowStart={setAuditRowStart}
        setAuditRowEnd={setAuditRowEnd}
        pdfSearch={pdfSearch}
        batchFilter={batchFilter}
      />
    </div>
  );
}
