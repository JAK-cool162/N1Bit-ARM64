#!/usr/bin/env python3
"""Small web UI for N1Bit: chat + live training.

    python app.py           # then open http://127.0.0.1:5000

No neuron grids or Vulkan — just a clean phone-friendly page that chats with the
model and can run/stop training in the background while showing loss live.
"""

import os
import threading
import torch
from flask import Flask, request, jsonify, Response

from n1bit.config import CHECKPOINT
from n1bit.model import model_from_config
from n1bit.tokenizer import ByteTokenizer

app = Flask(__name__)
tok = ByteTokenizer()

_model = None
_trainer = None
_train_thread = None
_stop = threading.Event()
_state = {"step": 0, "loss": 0.0, "best": float("inf"), "grade": "-",
          "tok_s": 0.0, "sample": "", "training": False}


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(CHECKPOINT):
            return None
        ck = torch.load(CHECKPOINT, map_location="cpu")
        m = model_from_config(ck["config"])
        m.load_state_dict(ck["model"])
        m.eval()
        _model = m
    return _model


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>N1Bit</title><style>
:root{color-scheme:dark}
body{margin:0;font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3}
header{padding:14px 16px;background:#161b22;font-weight:700;font-size:18px;border-bottom:1px solid #30363d}
.tabs{display:flex;background:#161b22;border-bottom:1px solid #30363d}
.tab{flex:1;text-align:center;padding:12px;cursor:pointer;color:#8b949e}
.tab.on{color:#58a6ff;border-bottom:2px solid #58a6ff}
.wrap{padding:16px;max-width:720px;margin:0 auto}
#log{height:52vh;overflow:auto;background:#010409;border:1px solid #30363d;border-radius:10px;padding:12px}
.msg{margin:8px 0;padding:10px 12px;border-radius:10px;white-space:pre-wrap;line-height:1.35}
.you{background:#1f6feb33;text-align:right}
.ai{background:#21262d}
.row{display:flex;gap:8px;margin-top:12px}
input,button{font-size:16px;border-radius:10px;border:1px solid #30363d}
input{flex:1;padding:12px;background:#0d1117;color:#e6edf3}
button{padding:12px 18px;background:#238636;color:#fff;border:0;font-weight:600}
button.stop{background:#da3633}
.stat{display:flex;justify-content:space-between;padding:10px 4px;border-bottom:1px solid #21262d}
.k{color:#8b949e}.v{font-variant-numeric:tabular-nums}
#sample{background:#010409;border:1px solid #30363d;border-radius:10px;padding:12px;min-height:60px;white-space:pre-wrap}
.hidden{display:none}
</style></head><body>
<header>🧠 N1Bit-ARM64</header>
<div class=tabs>
 <div class="tab on" onclick="show('chat')">Chat</div>
 <div class="tab" onclick="show('train')">Training</div>
</div>

<div id=chat class=wrap>
 <div id=log></div>
 <div class=row>
  <input id=box placeholder="Say something..." onkeydown="if(event.key=='Enter')send()">
  <button onclick=send()>Send</button>
 </div>
</div>

<div id=train class="wrap hidden">
 <div class=stat><span class=k>Step</span><span class=v id=t_step>0</span></div>
 <div class=stat><span class=k>Loss</span><span class=v id=t_loss>-</span></div>
 <div class=stat><span class=k>Best</span><span class=v id=t_best>-</span></div>
 <div class=stat><span class=k>Speed</span><span class=v id=t_tps>-</span></div>
 <div class=stat><span class=k>Grade</span><span class=v id=t_grade>-</span></div>
 <div class=row><button id=t_btn onclick=toggle()>Start training</button></div>
 <h4>Live sample</h4><div id=sample></div>
</div>

<script>
function show(t){for(const x of ['chat','train']){document.getElementById(x).classList.toggle('hidden',x!=t)}
document.querySelectorAll('.tab').forEach((e,i)=>e.classList.toggle('on',(i==0)==(t=='chat')))}
function add(cls,txt){const d=document.createElement('div');d.className='msg '+cls;d.textContent=txt;
const l=document.getElementById('log');l.appendChild(d);l.scrollTop=l.scrollHeight}
async function send(){const b=document.getElementById('box');const t=b.value.trim();if(!t)return;
b.value='';add('you',t);const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({text:t})});const j=await r.json();add('ai',j.reply)}
async function toggle(){await fetch('/train/toggle',{method:'POST'})}
async function poll(){const r=await fetch('/train/state');const s=await r.json();
t_step.textContent=s.step.toLocaleString();t_loss.textContent=s.loss.toFixed(4);
t_best.textContent=(s.best>1e9?'-':s.best.toFixed(4));t_tps.textContent=Math.round(s.tok_s).toLocaleString()+' tok/s';
t_grade.textContent=s.grade;sample.textContent=s.sample||'(waiting)';
t_btn.textContent=s.training?'Stop training':'Start training';t_btn.className=s.training?'stop':''}
setInterval(poll,1000);poll()
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/chat", methods=["POST"])
def chat_api():
    m = get_model()
    if m is None:
        return jsonify(reply="No model yet — train one first (Training tab or python train.py).")
    text = (request.json or {}).get("text", "")
    ids = torch.tensor([tok.encode(text + "\n")])
    out = m.generate(ids, max_new_tokens=180, temperature=0.8)
    return jsonify(reply=tok.decode(out[0].tolist()))


@app.route("/train/state")
def train_state():
    return jsonify(_state)


@app.route("/train/toggle", methods=["POST"])
def train_toggle():
    global _train_thread, _trainer
    if _state["training"]:
        _stop.set()
        _state["training"] = False
        return jsonify(ok=True, training=False)

    from n1bit.trainer import Trainer

    def run():
        global _model
        try:
            t = Trainer(on_update=lambda s: _state.update(s))
        except SystemExit as e:
            _state["sample"] = str(e)
            _state["training"] = False
            return
        _state["training"] = True
        t.train("inf", stop_flag=_stop.is_set)
        t.save()
        _model = None  # reload updated weights on next chat
        _state["training"] = False

    _stop.clear()
    _train_thread = threading.Thread(target=run, daemon=True)
    _train_thread.start()
    return jsonify(ok=True, training=True)


if __name__ == "__main__":
    print("N1Bit web UI -> http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
