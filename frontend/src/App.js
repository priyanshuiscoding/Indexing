import React, { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";
import { documentCatalog } from "./documentCatalog";

const API = process.env.REACT_APP_API_BASE || "";

const getFetchErrorMessage = (err) => {
  if (err instanceof TypeError && /fetch/i.test(err.message)) {
    return "Cannot reach the backend at http://localhost:8000. Start the FastAPI server and try again.";
  }
  return err.message || "Request failed";
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
    throw new Error(getFetchErrorMessage(err));
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
    throw new Error(getFetchErrorMessage(err));
  }
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
    throw new Error(getFetchErrorMessage(err));
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
    pdfPageFrom: item.pdfPageFrom || item.pageFrom || 1,
    pdfPageTo: item.pdfPageTo || item.pageTo || item.pageFrom || 1,
    tocPageFrom: item.tocPageFrom || "",
    tocPageTo: item.tocPageTo || "",
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

function InlineEditRow({ section, totalPages, onSave, onCancel }) {
  const [title, setTitle] = useState(section.title || "");
  const [subDoc, setSubDoc] = useState(section.subDocument || "");
  const [note, setNote] = useState(section.note || "");
  const [batchNo, setBatchNo] = useState(section.batchNo || "");
  const [pageFrom, setPageFrom] = useState(section.pageFrom || 1);
  const [pageTo, setPageTo] = useState(section.pageTo || 1);
  const [recDate, setRecDate] = useState(section.receivingDate || "");

  const save = () => {
    if (!title.trim()) return;
    const pf = Math.max(1, Math.min(parseInt(pageFrom, 10) || 1, totalPages));
    const pt = Math.max(pf, Math.min(parseInt(pageTo, 10) || pf, totalPages));
    const normalizedSubDoc = subDoc.trim() || "null";
    onSave({
      ...section,
      title: title.trim(),
      subDocument: normalizedSubDoc,
      note: note.trim(),
      batchNo: batchNo.trim(),
      pageFrom: pf,
      pageTo: pt,
      receivingDate: recDate,
      source: section.source === "gap" ? "manual" : section.source,
    });
  };

  return (
    <tr className="inline-edit-row">
      <td colSpan={7} style={{ padding: 0 }}>
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
    const pf = parseInt(pageFrom, 10);
    const pt = parseInt(pageTo, 10);
    if (pf && pt && pt >= pf) setTotal(String(pt - pf + 1));
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
    const pf = Math.max(1, Math.min(parseInt(pageFrom, 10) || 1, totalPages));
    const pt = Math.max(pf, Math.min(parseInt(pageTo, 10) || pf, totalPages));
    onAdd({
      title,
      subDocument: subDoc.trim() || "null",
      note: note.trim(),
      batchNo: batchNo.trim(),
      pageFrom: pf,
      pageTo: pt,
      totalPage: parseInt(total, 10) || (pt - pf + 1),
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
  const [queueSnapshot, setQueueSnapshot] = useState({ index_ready: [], pending_vectorization: [], vectorized: [], errors: [], runner: { running: false, processed: 0, total: 0, current_pdf_id: "", current_filename: "", last_error: "" }, index_runner: { running: false, current_pdf_id: "", current_filename: "", last_error: "", finished_pdf_id: "", finished_filename: "", status: "idle" } });
  const [selectedSavedPdfId, setSelectedSavedPdfId] = useState("");

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
  const handledIndexRunnerRef = useRef("");

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

  const handlePageVisible = useCallback((page) => setCurrentPage(page), []);

  const goToPage = (pageNum, secIdx = null) => {
    const page = Math.max(1, Math.min(pageNum, totalPages));
    setCurrentPage(page);
    document.getElementById(`pg-${page}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    if (secIdx !== null) setActiveIdx(secIdx);
  };

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
    try {
      const resp = await apiFetch("/api/queues");
      setQueueSnapshot(resp || { index_ready: [], pending_vectorization: [], vectorized: [], errors: [], runner: { running: false, processed: 0, total: 0, current_pdf_id: "", current_filename: "", last_error: "" }, index_runner: { running: false, current_pdf_id: "", current_filename: "", last_error: "", finished_pdf_id: "", finished_filename: "", status: "idle" } });
    } catch (err) {
      console.error(err);
    }
  }, []);

  const loadSavedPdf = useCallback(async (targetPdfId) => {
    if (!targetPdfId) return;
    setError("");
    setLoading(true);
    setLoadingStep("Loading saved PDF from backend...");
    try {
      const [details, blob, pdfjsLib] = await Promise.all([
        apiFetch(`/api/pdfs/${targetPdfId}`),
        apiFetchBlob(`/api/pdfs/${targetPdfId}/file`),
        loadPdfJs(),
      ]);
      const objectUrl = URL.createObjectURL(blob);
      const arrayBuf = await blob.arrayBuffer();
      const doc = await pdfjsLib.getDocument({ data: arrayBuf.slice(0) }).promise;
      const pdf = details.pdf || {};
      const nextCaseInfo = deriveDocumentMeta(pdf.filename || targetPdfId);
      resetDocumentState(nextCaseInfo);
      setPdfDoc(doc);
      setPdfFile({ name: pdf.filename || `${targetPdfId}.pdf`, objectUrl, source: "saved", pdfId: targetPdfId });
      setPdfId(targetPdfId);
      setSelectedSavedPdfId(targetPdfId);
      setTotalPages(pdf.total_pages || doc.numPages);
      setCurrentPage(1);
      setIndexStartPage(String(pdf.selected_start_page || 1));
      setIndexEndPage(String(pdf.selected_end_page || Math.min(doc.numPages, 10)));
      setWorkflowStatus(pdf.status || "");
      setRetrievalStatus(pdf.retrieval_status || "");
      setPendingPages(pdf.pending_pages || 0);
      setChatReady(Boolean(pdf.chat_ready));
      setIndexedRange({
        start: pdf.selected_start_page || 1,
        end: pdf.selected_end_page || Math.min(doc.numPages, 10),
        count: pdf.indexed_pages || 0,
      });
      const normalized = normalizeSections(details.index || [], nextCaseInfo.batchNo || "");
      setSections(normalized);
      rebuildCoverage(normalized, pdf.total_pages || doc.numPages);
    } catch (err) {
      console.error(err);
      setError("Error: " + err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  }, [rebuildCoverage]);

  useEffect(() => {
    fetchSavedPdfs();
    fetchQueues();
    const timer = setInterval(() => {
      fetchSavedPdfs(pdfSearch);
      fetchQueues();
    }, 3000);
    return () => clearInterval(timer);
  }, [fetchSavedPdfs, fetchQueues, pdfSearch]);

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
            end: pdf.selected_end_page || Math.min(pdf.total_pages || totalPages || 1, 10),
            count: pdf.indexed_pages || 0,
          });
        })
        .catch((err) => setError(`Error: ${err.message}`));
    }

    fetchSavedPdfs(pdfSearch);
    fetchQueues();
  }, [queueSnapshot.index_runner, pdfId, caseInfo?.batchNo, rebuildCoverage, totalPages, fetchSavedPdfs, fetchQueues, pdfSearch]);

  const handleBatchUpload = async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setError("");
    setLoading(true);
    setLoadingStep(`Uploading ${files.length} PDFs, scanning pages 1-10, and saving Stage 1 indexes...`);
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));
      formData.append("start_page", "1");
      formData.append("end_page", "10");
      const resp = await apiUpload("/api/ingest-batch", formData);
      await fetchSavedPdfs(pdfSearch);
      const firstReady = (resp.pdfs || []).find((item) => item.pdf_id);
      if (firstReady?.pdf_id) {
        await loadSavedPdf(firstReady.pdf_id);
      }
    } catch (err) {
      console.error(err);
      setError("Error: " + err.message);
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
      setIndexEndPage(String(Math.min(doc.numPages, 10)));
    } catch (err) {
      console.error(err);
      setPdfDoc(null);
      setPdfFile(null);
      setTotalPages(0);
      setError("Error: " + getFetchErrorMessage(err));
    } finally {
      setLoading(false);
      setLoadingStep("");
      e.target.value = "";
    }
  };

  const startIndexing = async () => {
    if (!pdfFile || !pdfDoc) {
      setError("Error: Upload a PDF before starting indexing.");
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
      if (isRealFile(pdfFile)) {
        setLoadingStep(`Starting background indexing for pages ${start} to ${end}...`);
        const formData = new FormData();
        formData.append("file", pdfFile);
        formData.append("start_page", String(start));
        formData.append("end_page", String(end));
        resp = await apiUpload("/api/index-runner/upload", formData);
        if (!resp.started) {
          if (resp.skipped_duplicate && resp.existing?.pdf_id) {
            setPdfId(resp.existing.pdf_id);
            await loadSavedPdf(resp.existing.pdf_id);
            await fetchSavedPdfs(pdfSearch);
            await fetchQueues();
            return;
          }
          throw new Error(resp.message || "Background indexing is already running.");
        }
        setPdfId(resp.pdf_id);
        handledIndexRunnerRef.current = "";
      } else if (pdfId) {
        setLoadingStep(`Starting background re-indexing for pages ${start} to ${end}...`);
        const formData = new FormData();
        formData.append("start_page", String(start));
        formData.append("end_page", String(end));
        resp = await apiUpload(`/api/index-runner/saved/${pdfId}`, formData);
        if (!resp.started) {
          throw new Error(resp.message || "Background indexing is already running.");
        }
        handledIndexRunnerRef.current = "";
      } else {
        throw new Error("No upload source found for this PDF.");
      }

      setWorkflowStatus("indexing_running");
      setChatReady(false);
      setBatchFilter((value) => value || caseInfo?.batchNo || "");
      await fetchSavedPdfs(pdfSearch);
      await fetchQueues();
    } catch (err) {
      console.error(err);
      setError("Error: " + err.message);
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
      setError("Error: " + err.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const processPendingPages = async () => {
    if (!pdfId) return;
    setError("");
    setLoading(true);
    setLoadingStep(`Deferred ingestion - processing ${pendingPages} pending pages...`);
    try {
      const resp = await apiFetch(`/api/process-pending/${pdfId}`, { method: "POST" });
      setWorkflowStatus(resp.status || "vectorized");
      setRetrievalStatus(resp.retrieval_status || "vectorized");
      setPendingPages(resp.pending_pages || 0);
      setChatReady(Boolean(resp.chat_ready));
      fetchSavedPdfs(pdfSearch);
    } catch (err) {
      setError("Error: " + err.message);
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
      fetchQueues();
    } catch (err) {
      setError("Error: " + err.message);
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
      setError("Error: " + err.message);
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
      setError("Error: This PDF is currently being processed. Stop or wait for the running job before deleting it.");
      return;
    }

    setError("");
    setLoading(true);
    setLoadingStep(`Deleting ${item.filename || item.pdf_id} from backend...`);
    try {
      await apiFetch(`/api/pdfs/${item.pdf_id}`, { method: "DELETE" });
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
      setError("Error: " + err.message);
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
      setError("Error: " + err.message);
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
      setError("Error: " + err.message);
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
  const heartbeatAgeSeconds = runner.heartbeat_ts ? Math.max(0, Math.round(Date.now() / 1000 - runner.heartbeat_ts)) : 0;
  const runnerLooksStuck = Boolean(runner.running && heartbeatAgeSeconds > 90);
  const runnerStopping = Boolean(runner.running && runner.pause_requested);
  const runnerStateLabel = runnerLooksStuck ? "Stuck" : runner.paused ? "Paused" : runnerStopping ? "Stopping" : runner.running ? "Running" : "Idle";

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
          {pdfFile && <span className="nav-filename">{pdfFile.name} · {totalPages}p</span>}
          {sections.length > 0 && (
            <span className="coverage-pill">OK {covered}/{totalPages} pages · {sections.length} entries</span>
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
        <div className="right-panel index-panel">
          <div className="workspace-header">
            <div className="section-card library-card">
              <div className="section-card-head">
                <div>
                  <div className="section-title">Saved PDF Library</div>
                  <div className="section-subtitle">Open any saved file and continue indexing, review, or chat without re-uploading.</div>
                </div>
                <div className="section-count">{savedPdfs.length} saved PDFs</div>
              </div>
              <div className="saved-pdf-list">
                {savedPdfs.map((item) => {
                  const isDeferredBusy = queueSnapshot.runner?.running && queueSnapshot.runner?.current_pdf_id === item.pdf_id;
                  const isIndexBusy = queueSnapshot.index_runner?.running && queueSnapshot.index_runner?.current_pdf_id === item.pdf_id;
                  const deleteDisabled = loading || isDeferredBusy || isIndexBusy;
                  return (
                    <button
                      key={item.pdf_id}
                      type="button"
                      onClick={() => loadSavedPdf(item.pdf_id)}
                      className={`saved-pdf-card ${item.pdf_id === selectedSavedPdfId ? "selected" : ""}`}
                    >
                      <div className="saved-pdf-row">
                        <div className="saved-pdf-title">{item.cnr_number || item.filename}</div>
                        <button
                          type="button"
                          className="saved-pdf-delete"
                          disabled={deleteDisabled}
                          onClick={(e) => {
                            e.stopPropagation();
                            setConfirmDeletePdf(item);
                          }}
                          title={deleteDisabled ? "This PDF is currently being processed" : "Delete this PDF and all saved data"}
                        >
                          Delete
                        </button>
                      </div>
                      <div className="saved-pdf-meta">{item.filename} ? {item.total_pages || 0} pages</div>
                      <div className="saved-pdf-tags">
                        <span>status: {item.status}</span>
                        <span>retrieval: {item.retrieval_status}</span>
                        <span>pending: {item.pending_pages || 0}</span>
                      </div>
                      <div className="saved-pdf-note">
                        {isDeferredBusy || isIndexBusy
                          ? "This PDF is currently being processed, so deletion is temporarily locked."
                          : item.pending_pages > 0
                            ? "Stage 1 is saved. You can process pending pages now or leave this PDF in the queue."
                            : "This PDF is fully ready in the backend library."}
                      </div>
                    </button>
                  );
                })}
                {savedPdfs.length === 0 && (
                  <div className="saved-pdf-empty">No saved PDFs yet. Upload one or use batch upload.</div>
                )}
              </div>
            </div>

            <div className="section-card queue-card">
              <div className="section-card-head">
                <div>
                  <div className="section-title">Queue And Live Status</div>
                  <div className="section-subtitle">Control deferred processing and check whether the backend is running, paused, idle, or stuck.</div>
                </div>
              </div>
              <div className="queue-toolbar">
                <span className="queue-pill">Index queue: {savedPdfs.filter((item) => !item.index_ready).length}</span>
                <span className="queue-pill">Deferred queue: {(queueSnapshot.pending_vectorization || []).length}</span>
                {(queueSnapshot.runner?.running || queueSnapshot.runner?.paused) && (
                  <span className="queue-pill success">{runnerStateLabel} {queueSnapshot.runner.processed}/{queueSnapshot.runner.total}: {queueSnapshot.runner.current_filename || queueSnapshot.runner.current_pdf_id || "Queue saved"}</span>
                )}
                {indexRunner.running && (
                  <span className="queue-pill success">Indexing: {indexRunner.current_filename || indexRunner.current_pdf_id || "Stage 1 running"}</span>
                )}
              </div>
              <div className="queue-actions-grid">
                <button className="queue-run-btn" onClick={runAllDeferredQueue} disabled={loading || queueSnapshot.runner?.running || queueSnapshot.runner?.paused}>Run All Deferred Queue</button>
                <button className="queue-pause-btn" onClick={() => controlDeferredQueue("stop")} disabled={loading || !queueSnapshot.runner?.running}>Stop Queue</button>
                <button className="queue-resume-btn" onClick={() => controlDeferredQueue("resume")} disabled={loading || !queueSnapshot.runner?.paused || queueSnapshot.runner?.running}>Resume Queue</button>
                <button className="queue-reset-btn" onClick={() => forceResetQueue("index")} disabled={loading || queueSnapshot.runner?.running}>Force Reset Index Queue</button>
                <button className="queue-reset-btn" onClick={() => forceResetQueue("deferred")} disabled={loading || queueSnapshot.runner?.running}>Force Reset Deferred Queue</button>
              </div>
              <div className="library-help">
                Each saved PDF stays in the backend. Stage 1 saves the file and index first, then any remaining pages wait in the deferred queue until you process them.
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
            <div className="indexing-toolbar">
              <div className="indexing-copy">
                <div className="indexing-title">Indexing Controls</div>
                <div className="indexing-hint">
                  Upload loads the preview first. Stage 1 then scans pages 1-10 by default, builds and saves the index immediately, and leaves the remaining pages for deferred ingestion unless you process them now.
                </div>
              </div>
              <div className="workflow-help">
                <span>1. Upload or fetch a PDF.</span>
                <span>2. Click Start Indexing to save Stage 1 in the backend.</span>
                <span>3. Use Process Pending now for small PDFs, or Save To Queue for later full vectorization.</span>
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
            <div className="tab-bar">
              <button className={`tab-btn ${tab === "index" ? "active" : ""}`} onClick={() => setTab("index")}>
                Index Table
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
            <div className="content-panel index-content-panel">
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
                            <td className="col-page">{formatPageValue(sec.tocPageFrom, sec.tocPageFrom || sec.tocPageTo, sec.pdfPageFrom || sec.pageFrom, sec.pdfPageFrom || sec.pageFrom, "TOC", "PDF")}</td>
                            <td className="col-page">{formatPageValue(sec.tocPageTo, sec.tocPageTo || sec.tocPageFrom, sec.pdfPageTo || sec.pageTo, sec.pdfPageTo || sec.pageTo, "TOC", "PDF")}</td>
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
                      <div style={{ fontSize: 13 }}>Stage 1 will scan pages 1-10 by default unless you choose another range</div>
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

        <div className="left-panel viewer-panel">
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
    </div>
  );
}

