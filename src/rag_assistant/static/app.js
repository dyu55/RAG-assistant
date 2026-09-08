const $ = (selector) => document.querySelector(selector);
let currentAnswer = null;
const node = (tag, className, text) => {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text !== undefined) n.textContent = text;
  return n;
};
function notice(message, error = false) {
  const box = $("#notification");
  box.textContent = message;
  box.className = error ? "error" : "";
  box.hidden = false;
}
async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Please check your input and try again.",
    );
  return data;
}
const post = (url, data) =>
  request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
async function refresh() {
  const [health, docs] = await Promise.all([
    request("/api/health"),
    request("/api/documents"),
  ]);
  $("#provider-label").textContent =
    health.provider === "offline"
      ? "Offline · source excerpts"
      : `${health.provider} · model connected by configuration`;
  $("#document-count").textContent = docs.length;
  $("#documents").replaceChildren();
  const selected = $("#filter").value;
  $("#filter").replaceChildren(node("option", "", "All documents"));
  $("#filter").firstChild.value = "";
  docs.forEach((doc, index) => {
    const row = node("div", "document");
    row.append(
      node("span", "document-icon", String(index + 1).padStart(2, "0")),
    );
    const info = node("div", "document-info");
    info.append(
      node("strong", "", doc.filename),
      node(
        "small",
        "",
        `${doc.chunks} passage${doc.chunks === 1 ? "" : "s"} · ${(doc.bytes / 1024).toFixed(1)} KB`,
      ),
    );
    row.append(info);
    const remove = node("button", "document-remove", "×");
    remove.setAttribute("aria-label", `Remove ${doc.filename}`);
    remove.onclick = async () => {
      if (!confirm(`Remove ${doc.filename} from the index?`)) return;
      try {
        await request(`/api/documents/${doc.id}`, { method: "DELETE" });
        await refresh();
        notice("Document removed. Future answers use the updated library.");
      } catch (e) {
        notice(e.message, true);
      }
    };
    row.append(remove);
    $("#documents").append(row);
    const option = node("option", "", doc.filename);
    option.value = doc.id;
    $("#filter").append(option);
  });
  if ([...$("#filter").options].some((o) => o.value === selected))
    $("#filter").value = selected;
  if (!docs.length)
    $("#documents").append(
      node("p", "upload-hint", "Your library is ready for its first document."),
    );
}
async function showView(view) {
  for (const section of document.querySelectorAll(".view"))
    section.hidden = section.id !== `${view}-view`;
  for (const nav of document.querySelectorAll(".nav")) {
    nav.classList.toggle("active", nav.dataset.view === view);
    if (nav.dataset.view === view) nav.setAttribute("aria-current", "page");
    else nav.removeAttribute("aria-current");
  }
  const labels = {
    ask: "The reading room",
    library: "The collection",
    graph: "The connection atlas",
    history: "The notebook",
  };
  $("#breadcrumb").textContent = labels[view].toUpperCase();
  try {
    if (view === "graph") renderGraph(await request("/api/graph"));
    if (view === "history") renderHistory(await request("/api/history"));
  } catch (e) {
    notice(e.message, true);
  }
}
document
  .querySelectorAll(".nav")
  .forEach((button) => (button.onclick = () => showView(button.dataset.view)));
$("#open-collection").onclick = () => showView("library");
$("#upload-button").onclick = () => $("#upload").click();
$("#demo").onclick = async () => {
  $("#demo").disabled = true;
  try {
    await post("/api/demo", {});
    await refresh();
    notice(
      "Example library loaded: four fictional Atlas documents. Try a suggested question.",
    );
  } catch (e) {
    notice(e.message, true);
  } finally {
    $("#demo").disabled = false;
  }
};
$("#upload").onchange = async (event) => {
  const files = [...event.target.files];
  const errors = [];
  let count = 0;
  for (const file of files) {
    notice(`Indexing ${file.name}…`);
    const body = new FormData();
    body.append("file", file);
    try {
      await request("/api/documents", { method: "POST", body });
      count++;
    } catch (e) {
      errors.push(`${file.name}: ${e.message}`);
    }
  }
  await refresh();
  notice(
    `${count} document${count === 1 ? "" : "s"} indexed.${errors.length ? " " + errors.join(" ") : ""}`,
    !!errors.length,
  );
  event.target.value = "";
};
document.querySelectorAll(".suggestions button").forEach(
  (button) =>
    (button.onclick = () => {
      $("#question").value = button.textContent;
      $("#question-form").requestSubmit();
    }),
);
$("#question").onkeydown = (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!$("#ask-button").disabled) $("#question-form").requestSubmit();
  }
};
$("#question-form").onsubmit = async (event) => {
  event.preventDefault();
  if (!$("#question").value.trim() || $("#ask-button").disabled) return;
  const button = $("#ask-button");
  button.disabled = true;
  button.textContent = "Finding evidence…";
  $("#notification").hidden = true;
  try {
    const answer = await post("/api/ask", {
      text: $("#question").value,
      mode: $("#mode").value,
      document_ids: $("#filter").value ? [$("#filter").value] : [],
    });
    renderAnswer(answer);
  } catch (e) {
    notice(e.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "Find evidence ↗";
  }
};
function renderAnswer(answer) {
  currentAnswer = answer;
  document.body.classList.add("has-answer");
  $("#empty-state").hidden = true;
  $("#answer-layout").hidden = false;
  $("#answer-question").textContent = answer.question;
  $("#verdict").textContent =
    answer.status === "supported"
      ? "✓ Quotes verified"
      : "Insufficient evidence";
  $("#verdict").className =
    `pill ${answer.status === "abstained" ? "warning" : ""}`;
  $("#answer-text").replaceChildren();
  if (!answer.claims.length)
    $("#answer-text").append(node("p", "", answer.answer));
  answer.claims.forEach((claim) => {
    const p = node("p", "", claim.text);
    const cite = node("button", "citation-link", claim.source_id);
    cite.setAttribute("aria-label", `View source ${claim.source_id}`);
    cite.onclick = () => {
      document
        .querySelectorAll(".source-card")
        .forEach((c) => c.classList.remove("highlight"));
      const source = $(`#source-${claim.source_id}`);
      if (source) {
        source.classList.add("highlight");
        source.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    };
    p.append(cite);
    $("#answer-text").append(p);
  });
  $("#answer-meta").replaceChildren(
    ...[
      `${answer.evidence.length} sources retrieved`,
      answer.provider === "offline"
        ? "Source excerpts · no model"
        : answer.provider,
      `${answer.timings_ms.total.toFixed(0)} ms`,
      ...(answer.cached ? ["Cached · evidence retained"] : []),
    ].map((s) => node("span", "", s)),
  );
  $("#verification-title").textContent =
    answer.status === "supported"
      ? "Evidence checks complete"
      : "Answer withheld";
  $("#verification-text").textContent = answer.reason;
  $("#answer-warnings").replaceChildren(
    ...answer.warnings.map((w) => node("p", "warning-message", w)),
  );
  $("#sources").replaceChildren();
  $("#source-count").textContent = String(answer.evidence.length).padStart(
    2,
    "0",
  );
  answer.evidence.forEach((source) => {
    const card = node("article", "source-card");
    card.id = `source-${source.source_id}`;
    const title = node("div", "source-title");
    title.append(
      node("span", "source-id", source.source_id),
      node("span", "", source.filename),
    );
    card.append(
      title,
      node(
        "div",
        "source-location",
        `Page ${source.page} · characters ${source.start}–${source.end}`,
      ),
    );
    const claim = answer.claims.find((c) => c.source_id === source.source_id);
    card.append(
      node(
        "blockquote",
        "source-quote",
        claim
          ? claim.quote
          : source.text.slice(0, 230) + (source.text.length > 230 ? "…" : ""),
      ),
    );
    const tags = node("div");
    source.channels.forEach((channel) =>
      tags.append(node("span", "channel", channel)),
    );
    card.append(tags);
    if (source.path.length > 1)
      card.append(node("div", "source-path", source.path.join(" → ")));
    const detail = node("details");
    detail.append(
      node("summary", "", "Read passage"),
      node("p", "", source.text),
    );
    card.append(detail);
    $("#sources").append(card);
  });
}
$("#export").onclick = () => {
  if (!currentAnswer) return;
  const blob = new Blob([JSON.stringify(currentAnswer, null, 2)], {
    type: "application/json",
  });
  const link = node("a");
  link.href = URL.createObjectURL(blob);
  link.download = `rag-answer-${currentAnswer.id.slice(0, 8)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
};
function renderHistory(answers) {
  $("#history").replaceChildren();
  if (!answers.length)
    $("#history").append(
      node("p", "upload-hint", "Your questions will appear here."),
    );
  answers.forEach((answer) => {
    const button = node("button", "history-item");
    button.append(
      node("strong", "", answer.question),
      node(
        "small",
        "",
        `${new Date(answer.created_at).toLocaleString()} · ${answer.status} · ${answer.evidence.length} sources · library revision ${answer.revision}`,
      ),
    );
    button.onclick = () => {
      showView("ask");
      $("#question").value = answer.question;
      renderAnswer(answer);
    };
    $("#history").append(button);
  });
}
function renderGraph(data) {
  const svg = $("#network");
  svg.replaceChildren();
  const nodes = data.nodes.slice(0, 16);
  $("#graph-count").textContent =
    `${data.nodes.length} entities · ${data.edges.length} links`;
  const ns = "http://www.w3.org/2000/svg";
  const make = (tag, attrs) => {
    const e = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, String(v)));
    return e;
  };
  if (!nodes.length) {
    const text = make("text", {
      x: 480,
      y: 260,
      "text-anchor": "middle",
      fill: "#656870",
    });
    text.textContent = "Add documents to discover their connections.";
    svg.append(text);
    return;
  }
  const points = new Map(
    nodes.map((n, i) => {
      const angle = ((i - 1) / Math.max(nodes.length - 1, 1)) * Math.PI * 2;
      return [
        n.id,
        i === 0
          ? { x: 480, y: 260 }
          : { x: 480 + Math.cos(angle) * 340, y: 260 + Math.sin(angle) * 210 },
      ];
    }),
  );
  data.edges.forEach((edge) => {
    const a = points.get(edge.source),
      b = points.get(edge.target);
    if (a && b)
      svg.append(
        make("line", {
          x1: a.x,
          y1: a.y,
          x2: b.x,
          y2: b.y,
          class: "graph-edge",
        }),
      );
  });
  nodes.forEach((n, i) => {
    const p = points.get(n.id);
    const group = make("g", {
      class: `graph-node ${i === 0 ? "center" : ""}`,
      tabindex: 0,
      role: "button",
      "aria-label": `Explore ${n.id}`,
    });
    group.append(
      make("circle", {
        cx: p.x,
        cy: p.y,
        r: i === 0 ? 49 : Math.min(44, 26 + n.count * 2),
      }),
    );
    const label = make("text", { x: p.x, y: p.y + 5 });
    label.textContent = n.id;
    if (n.id.length > 16) {
      label.setAttribute("y", p.y + 55);
      if (i === 0) label.setAttribute("fill", "#224de0");
    }
    group.append(label);
    const action = () => {
      showView("ask");
      $("#question").value = `What do the documents say about ${n.id}?`;
      $("#question").focus();
    };
    group.onclick = action;
    group.onkeydown = (e) => {
      if (e.key === "Enter") action();
    };
    svg.append(group);
  });
  $("#graph-legend").replaceChildren(
    ...[
      "◈ Entities extracted from document text",
      "↔ Connections retain passage references",
      "⌕ Select a node to ask a question",
    ].map((s) => node("span", "", s)),
  );
}
refresh().catch((e) => notice(e.message, true));
