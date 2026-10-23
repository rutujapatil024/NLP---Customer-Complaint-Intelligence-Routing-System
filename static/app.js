/**
 * app.js — Nagrik Samadhan Complaint Portal Controller
 */
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("complaintForm");
  const input = document.getElementById("complaintInput");
  const charCount = document.getElementById("charCount");
  const clearBtn = document.getElementById("clearBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");

  const resultsSection = document.getElementById("resultsSection");
  const resultsEmpty = document.getElementById("resultsEmpty");
  const resultsContent = document.getElementById("resultsContent");


  const resultRoutingQueue = document.getElementById("resultRoutingQueue");
  const resultDeptIcon = document.getElementById("resultDeptIcon");
  const resultDeptName = document.getElementById("resultDeptName");

  const resultUrgencyBadge = document.getElementById("resultUrgencyBadge");
  const resultUrgencyName = document.getElementById("resultUrgencyName");
  const resultConfidenceVal = document.getElementById("resultConfidenceVal");
  const confRingFill = document.getElementById("confRingFill");

  const resultSentimentPill = document.getElementById("resultSentimentPill");
  const resultCuesContainer = document.getElementById("resultCuesContainer");
  const probList = document.getElementById("probList");
  const btnRedirect = document.getElementById("btnRedirect");

  // Department SVG icons
  const DEPT_ICONS = {
    "Water Supply & Drainage": `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>`,
    "Electricity & Power": `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`,
    "Roads & Civic Infrastructure": `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19L9 5h6l5 14"/><line x1="12" y1="8" x2="12" y2="10"/><line x1="12" y1="13" x2="12" y2="15"/></svg>`,
    "Garbage & Sanitation": `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>`,
    "Public Transport": `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="15" rx="2"/><circle cx="7" cy="18" r="2"/><circle cx="17" cy="18" r="2"/><line x1="3" y1="11" x2="21" y2="11"/></svg>`,
    "Citizen & Consumer Services": `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>`
  };

  const DEPT_COLORS = {
    "Water Supply & Drainage": "#0ea5e9",
    "Electricity & Power": "#eab308",
    "Roads & Civic Infrastructure": "#f97316",
    "Garbage & Sanitation": "#22c55e",
    "Public Transport": "#8b5cf6",
    "Citizen & Consumer Services": "#3b82f6"
  };

  const PRESETS = {
    water_supply: "The water supply has not been coming properly for the last two days. Please look into this issue as soon as possible.",
    garbage: "There is a lot of garbage near our building and it has not been collected for several days. Kindly arrange for cleaning.",
    street_light: "The street light in our area has not been working for the past week. It becomes very difficult to travel at night. Please fix it.",
    potholes: "The road near our society is damaged and has many potholes. Please repair it to avoid accidents.",
    application_status: "I have submitted my application but have not received any update yet. Kindly check the status and provide an update.",
    power_cut: "There is frequent power supply interruption in our area. Please check the issue and resolve it.",
    drainage: "The drainage is overflowing near our building, causing a bad smell and unhygienic conditions. Please take immediate action.",
    wrong_item: "I received the wrong item in my order. Please arrange for a replacement or refund.",
    bus_delay: "The bus did not arrive on time today, and there was no information about the delay. Please look into this.",
    service_delay: "The service is taking too long and my complaint has still not been resolved. Kindly provide a proper update."
  };

  // Char counter
  input.addEventListener("input", () => {
    charCount.textContent = `${input.value.length} chars`;
  });

  clearBtn.addEventListener("click", () => {
    input.value = "";
    charCount.textContent = "0 chars";
    input.focus();
  });

  input.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  // Presets
  document.querySelectorAll(".preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-preset");
      if (PRESETS[key]) {
        input.value = PRESETS[key];
        charCount.textContent = `${input.value.length} chars`;
        runAnalysis(PRESETS[key]);
      }
    });
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) { input.focus(); return; }
    runAnalysis(text);
  });

  // Redirect & Action button handlers
  btnRedirect.addEventListener("click", () => {
    const dept = resultDeptName.textContent;
    alert(`Redirecting to ${dept} Department Portal...\n\nIn a production deployment, this transfers the complaint directly into the department's active queue.`);
  });

  const btnTrack = document.getElementById("btnTrack");
  if (btnTrack) {
    btnTrack.addEventListener("click", () => {
      alert("Status: Ticket actively routed and queued for department officer triage.");
    });
  }

  const btnReceipt = document.getElementById("btnReceipt");
  if (btnReceipt) {
    btnReceipt.addEventListener("click", () => {
      window.print();
    });
  }

  async function runAnalysis(text) {
    analyzeBtn.classList.add("loading");
    analyzeBtn.disabled = true;

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      renderResults(json.data);

      setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 60);
    } catch (err) {
      alert("Analysis failed: " + err.message);
    } finally {
      analyzeBtn.classList.remove("loading");
      analyzeBtn.disabled = false;
    }
  }

  function renderResults(d) {
    resultsEmpty.classList.add("hidden");
    resultsContent.classList.remove("hidden");

    // Routing queue
    resultRoutingQueue.textContent = d.routing_queue;

    // Department
    resultDeptName.textContent = d.department;
    const deptColor = DEPT_COLORS[d.department] || "#3b82f6";
    resultDeptIcon.innerHTML = DEPT_ICONS[d.department] || "";
    resultDeptIcon.style.backgroundColor = deptColor;

    // Urgency
    resultUrgencyName.textContent = d.urgency;
    resultUrgencyBadge.className = "urgency-pill u-" + d.urgency.toLowerCase();

    // Confidence ring
    const confNum = parseFloat(d.confidence_pct);
    resultConfidenceVal.textContent = d.confidence_pct;
    confRingFill.setAttribute("stroke-dasharray", `${confNum}, 100`);

    // Sentiment
    let desc = "Neutral";
    if (d.sentiment_score <= -0.4) desc = "Distressed";
    else if (d.sentiment_score < 0) desc = "Concerned";
    else if (d.sentiment_score > 0.3) desc = "Polite";
    resultSentimentPill.textContent = `${d.sentiment_score > 0 ? "+" : ""}${d.sentiment_score.toFixed(2)}  (${desc})`;

    // Cues
    resultCuesContainer.innerHTML = "";
    if (d.detected_cues && d.detected_cues.length) {
      d.detected_cues.forEach((c) => {
        const s = document.createElement("span");
        s.className = "cue-tag";
        s.textContent = c;
        resultCuesContainer.appendChild(s);
      });
    } else {
      const s = document.createElement("span");
      s.className = "no-cues";
      s.textContent = "No high-urgency keywords";
      resultCuesContainer.appendChild(s);
    }

    // Probabilities
    probList.innerHTML = "";
    if (d.breakdown) {
      Object.entries(d.breakdown)
        .sort((a, b) => b[1] - a[1])
        .forEach(([dept, prob]) => {
          const pct = (prob * 100).toFixed(1);
          const row = document.createElement("div");
          row.className = `prob-row${dept === d.department ? " active" : ""}`;
          row.innerHTML = `
            <div class="prob-info"><span>${dept}</span><span>${pct}%</span></div>
            <div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:${pct}%"></div></div>`;
          probList.appendChild(row);
        });
    }
  }
});
