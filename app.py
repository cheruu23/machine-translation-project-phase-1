"""
Flask deployment for the English -> Amharic NMT system.

Run:
    python app.py                      # http://127.0.0.1:5000
    ARTIFACT_DIR=/path/to/artifacts python app.py

Endpoints:
    GET  /            -> small browser demo page
    GET  /health      -> {"status": "ok", ...}
    POST /translate   -> {"text": "..."} => {"translation": "..."}
"""

import os
import time

from flask import Flask, request, jsonify, render_template_string

from nmt_translator import Translator

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")

app = Flask(__name__)
translator = Translator(artifact_dir=ARTIFACT_DIR)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>English → Amharic NMT</title>
<style>
 :root{--bg:#0f1117;--card:#181b24;--fg:#e8eaf0;--mut:#98a0b3;--acc:#7aa2f7}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
      display:flex;justify-content:center;padding:32px 16px}
 .wrap{width:100%;max-width:720px}
 h1{font-size:1.4rem;margin:0 0 4px}
 p.sub{color:var(--mut);margin:0 0 24px;font-size:.9rem}
 .card{background:var(--card);border:1px solid #262a36;border-radius:12px;
       padding:18px;margin-bottom:16px}
 textarea{width:100%;min-height:96px;background:#11141c;color:var(--fg);
          border:1px solid #2b303d;border-radius:8px;padding:12px;
          font-size:1rem;resize:vertical}
 button{background:var(--acc);color:#0b0d13;border:0;border-radius:8px;
        padding:10px 18px;font-weight:600;font-size:.95rem;cursor:pointer;
        margin-top:12px}
 button:disabled{opacity:.5;cursor:default}
 .out{font-size:1.25rem;line-height:1.9;min-height:2.4rem;word-break:break-word}
 .lbl{color:var(--mut);font-size:.78rem;text-transform:uppercase;
      letter-spacing:.06em;margin-bottom:6px}
 .meta{color:var(--mut);font-size:.8rem;margin-top:10px}
</style></head><body><div class="wrap">
 <h1>English → Amharic Neural Machine Translation</h1>
 <p class="sub">Attention-based Seq2Seq + LSTM &middot; greedy decoding</p>
 <div class="card">
   <div class="lbl">English input</div>
   <textarea id="src">I am going to the university.</textarea>
   <button id="go" onclick="go()">Translate</button>
 </div>
 <div class="card">
   <div class="lbl">Amharic translation</div>
   <div class="out" id="out">&nbsp;</div>
   <div class="meta" id="meta"></div>
 </div>
</div>
<script>
async function go(){
  const b=document.getElementById('go'); b.disabled=true;
  document.getElementById('out').textContent='…';
  document.getElementById('meta').textContent='';
  try{
    const r=await fetch('/translate',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:document.getElementById('src').value})});
    const j=await r.json();
    document.getElementById('out').textContent=j.translation||'(empty)';
    if(j.latency_ms!==undefined)
      document.getElementById('meta').textContent=j.latency_ms+' ms · '+j.model;
  }catch(e){document.getElementById('out').textContent='Error: '+e;}
  b.disabled=false;
}
</script></body></html>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(translator.device),
        "source_vocab": translator.src_vocab,
        "target_vocab": translator.tgt_vocab,
        "attention_checkpoint_epoch": translator.attn_epoch,
        "basic_model_loaded": translator.basic is not None,
    })


@app.post("/translate")
def translate():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    model = data.get("model", "attention")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be a "
                                 "non-empty string."}), 400
    if model not in ("attention", "basic"):
        return jsonify({"error": "Field 'model' must be 'attention' or "
                                 "'basic'."}), 400
    if model == "basic" and translator.basic is None:
        return jsonify({"error": "Basic checkpoint not loaded."}), 400

    t0 = time.perf_counter()
    try:
        out = translator.translate(text, model=model)
    except Exception as exc:                     # noqa: BLE001
        return jsonify({"error": f"Inference failed: {exc}"}), 500
    latency = (time.perf_counter() - t0) * 1000

    return jsonify({
        "text": text,
        "translation": out,
        "model": model,
        "latency_ms": round(latency, 1),
    })


@app.post("/translate_batch")
def translate_batch():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts")
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "Field 'texts' must be a non-empty list."}), 400
    if len(texts) > 128:
        return jsonify({"error": "Maximum 128 sentences per request."}), 400

    t0 = time.perf_counter()
    outs = [translator.translate(t) for t in texts]
    latency = (time.perf_counter() - t0) * 1000
    return jsonify({
        "translations": outs,
        "count": len(outs),
        "latency_ms": round(latency, 1),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
