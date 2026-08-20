// ── Tier config ───────────────────────────────────────────────────────────────
const TIERS = {
  Budget:    { color: "#2ecc71", bg: "#2ecc7122" },
  "Mid-Range":{ color: "#f39c12", bg: "#f39c1222" },
  "Upper Mid":{ color: "#e67e22", bg: "#e67e2222" },
  Premium:   { color: "#e74c3c", bg: "#e74c3c22" },
};

const TIER_ORDER = ["Budget", "Mid-Range", "Upper Mid", "Premium"];

// ── Predict ───────────────────────────────────────────────────────────────────
async function predict() {
  const btn = document.querySelector(".predict-btn");
  btn.disabled = true;
  btn.textContent = "⏳ Predicting…";

  // Hide old results, show spinner
  document.getElementById("results").style.display     = "none";
  document.getElementById("placeholder").style.display = "flex";
  document.getElementById("placeholder").innerHTML =
    '<div class="placeholder-inner"><div class="spinner"></div><p>Analysing specs…</p></div>';

  // Parse resolution
  const resSel = document.getElementById("resolution").value.split("x");

  const payload = {
    brand:           document.getElementById("brand").value,
    ram_gb:          document.getElementById("ram_gb").value,
    storage_gb:      document.getElementById("storage_gb").value,
    battery_mah:     document.getElementById("battery_mah").value,
    screen_size_in:  document.getElementById("screen_size_in").value,
    camera_mp:       document.getElementById("camera_mp").value,
    front_camera_mp: document.getElementById("front_camera_mp").value,
    processor_ghz:   document.getElementById("processor_ghz").value,
    refresh_rate:    document.getElementById("refresh_rate").value,
    five_g:          document.getElementById("five_g").checked ? 1 : 0,
    nfc:             document.getElementById("nfc").checked ? 1 : 0,
    ir_blaster:      document.getElementById("ir_blaster").checked ? 1 : 0,
    ext_card:        document.getElementById("ext_card").checked ? 1 : 0,
    rating:          document.getElementById("rating").value,
    res_width:       resSel[0],
    res_height:      resSel[1],
  };

  try {
    const res  = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    renderResults(data);
  } catch (e) {
    document.getElementById("placeholder").innerHTML =
      '<div class="placeholder-inner"><p style="color:#e74c3c">❌ Error: ' + e.message + '</p></div>';
  }

  btn.disabled = false;
  btn.textContent = "🔍 Predict Price Range";
}

// ── Render ────────────────────────────────────────────────────────────────────
function renderResults(data) {
  const tier  = data.tier;
  const cfg   = TIERS[tier.label] || { color: "#888", bg: "#88888822" };

  // Show results panel, hide placeholder
  document.getElementById("placeholder").style.display = "none";
  document.getElementById("results").style.display     = "flex";

  // ── Result Card ──────────────────────────────────────────────────────────
  const card = document.getElementById("result-card");
  card.style.background   = cfg.bg;
  card.style.borderColor  = cfg.color;

  document.getElementById("result-emoji").textContent = tier.emoji;
  document.getElementById("result-tier").textContent  = tier.label;
  document.getElementById("result-tier").style.color  = cfg.color;
  document.getElementById("result-range").textContent = tier.range;
  document.getElementById("result-model").textContent = "Model: " + data.model_name;

  // ── Confidence Bars ───────────────────────────────────────────────────────
  const barsEl = document.getElementById("confidence-bars");
  barsEl.innerHTML = "";
  TIER_ORDER.forEach((name, i) => {
    const pct  = (data.proba[i] * 100).toFixed(1);
    const tcfg = TIERS[name] || { color: "#888" };
    barsEl.innerHTML += `
      <div class="conf-row">
        <div class="conf-label">
          <span>${name}</span><span>${pct}%</span>
        </div>
        <div class="conf-track">
          <div class="conf-fill" style="width:${pct}%;background:${tcfg.color}"></div>
        </div>
      </div>`;
  });

  // ── Similar Phones ────────────────────────────────────────────────────────
  const simEl = document.getElementById("similar-phones");
  simEl.innerHTML = "";

  if (!data.similar || data.similar.length === 0) {
    simEl.innerHTML = "<p style='color:#777'>No similar phones found.</p>";
    return;
  }

  data.similar.forEach(phone => {
    const tierName  = phone.tier || "Budget";
    const tierCfg   = TIERS[tierName] || { color: "#888" };
    const fiveG     = phone.five_g ? "✅" : "❌";
    simEl.innerHTML += `
      <div class="phone-card">
        <div>
          <div class="phone-name">${phone.model}</div>
          <div class="phone-brand">${phone.brand}</div>
        </div>
        <div class="phone-stat">
          <span class="val">${phone.ram_gb}GB</span>RAM
        </div>
        <div class="phone-stat">
          <span class="val">${phone.camera_mp}MP</span>Camera
        </div>
        <div class="phone-stat">
          <span class="val">${Math.round(phone.battery_mah)}</span>Battery
        </div>
        <div class="phone-stat">
          <span class="val">${phone.processor_ghz}GHz</span>CPU
        </div>
        <div class="phone-stat">
          <span class="val">${fiveG}</span>5G
        </div>
        <div class="phone-price">
          <div class="price-amount">₹${phone.price.toLocaleString("en-IN")}</div>
          <div class="price-badge"
               style="background:${tierCfg.color}22;color:${tierCfg.color}">
            ${tierName}
          </div>
        </div>
      </div>`;
  });
}
