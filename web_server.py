#!/usr/bin/env python3
"""
ELF2 绝缘子检测 - Web 仪表盘
=============================
在 RK3588 上运行 Flask 服务, PC 浏览器访问 http://<板端IP>:5000 查看实时检测结果。

功能:
  - MJPEG 实时视频流 (带检测框标注)
  - 检测统计 / 核验状态
  - 历史核验记录列表
"""

import json
import time
import threading
from typing import Optional, List, Dict
from pathlib import Path

import numpy as np
import cv2

try:
    from flask import Flask, Response, jsonify, send_file
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False

# ============================================================
# 通用 CSS + 导航栏 HTML 片段
# ============================================================
_BASE_STYLE = """*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Segoe UI","Microsoft YaHei",sans-serif;background:#0f0f1a;color:#e0e0e0;min-height:100vh}
.header{background:#1a1a2e;padding:10px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid #e94560}
.header h1{font-size:20px;color:#e94560}
.header .nav{display:flex;gap:8px}
.header .nav a{color:#a0a0a0;text-decoration:none;padding:6px 16px;border-radius:6px;font-size:14px;transition:all .2s}
.header .nav a:hover{color:#fff;background:#222244}
.header .nav a.active{color:#e94560;background:#2a1a1e;font-weight:bold}
.header .badge{font-size:13px;color:#a0a0a0}
.card{background:#1a1a2e;border-radius:10px;padding:16px;border:1px solid #16213e}
.card h3{font-size:14px;color:#00ff88;margin-bottom:10px;border-bottom:1px solid #16213e;padding-bottom:8px}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;margin-right:4px}
.tag-ok{background:#0a3d0a;color:#4caf50}
.tag-err{background:#3d0a0a;color:#e94560}
.tag-unk{background:#3d2e0a;color:#ff9800}
.tag-def{background:#3d0a2e;color:#ff4081}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:#333;border-radius:2px}"""

_NAV_HTML = """<div class="header">
  <h1>⚡ ELF2 绝缘子智能巡检</h1>
  <div class="nav">
    <a href="/" class="active" id="nav-live">📷 实时监控</a>
    <a href="/results" id="nav-results">📋 核验结果</a>
  </div>
  <span class="badge" id="conn-status">连接中...</span>
</div>"""

# ============================================================
# 第1页：实时监控 (主页 /)
# ============================================================
LIVE_PAGE = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ELF2 实时监控</title>
<style>{_BASE_STYLE}
.container{{display:flex;gap:16px;padding:16px;max-width:1400px;margin:0 auto;height:calc(100vh - 64px)}}
.video-panel{{flex:1;background:#000;border-radius:10px;overflow:hidden;border:2px solid #16213e;display:flex;align-items:center;justify-content:center}}
.video-panel img{{max-width:100%;max-height:100%;object-fit:contain}}
.side-panel{{width:380px;display:flex;flex-direction:column;gap:12px}}
.stat-row{{display:flex;justify-content:space-between;padding:4px 0;font-size:13px}}
.stat-row .val{{font-weight:bold;color:#00d4ff}}
#verify-status{{font-size:13px;padding:6px 10px;border-radius:6px;text-align:center;margin-top:8px}}
#verify-status.ready{{background:#0a3d0a;color:#4caf50}}
#verify-status.busy{{background:#3d2e0a;color:#ff9800}}
#history-list{{max-height:380px;overflow-y:auto;font-size:12px}}
.history-item{{padding:8px 10px;border-bottom:1px solid #16213e}}
.history-item .ts{{color:#888;font-size:11px}}
.history-item .summary{{color:#e0e0e0;margin-top:2px}}
</style>
</head>
<body>
{_NAV_HTML}
<div class="container">
  <div class="video-panel"><img id="stream" src="/video_feed" alt="实时画面"></div>
  <div class="side-panel">
    <div class="card">
      <h3>📊 实时状态</h3>
      <div class="stat-row"><span>检测目标</span><span class="val" id="det-count">--</span></div>
      <div class="stat-row"><span>FPS</span><span class="val" id="fps">--</span></div>
      <div class="stat-row"><span>破损</span><span class="val" id="broken-count">--</span></div>
      <div id="verify-status" class="ready">✓ 自动核验就绪</div>
    </div>
    <div class="card">
      <h3>📋 最近核验 (<span id="hist-total">0</span>)</h3>
      <div id="history-list">加载中...</div>
    </div>
  </div>
</div>
<script>
const CLS_NAME={{0:"Nora1",1:"JYZ",2:"Broken"}};
let lastHist=0;
async function fetchStatus(){{
  try{{
    const r=await fetch("/api/status");
    const d=await r.json();
    document.getElementById("det-count").textContent=d.det_count;
    document.getElementById("fps").textContent=(d.fps||0).toFixed(1);
    document.getElementById("broken-count").textContent=d.broken_count||0;
    document.getElementById("broken-count").style.color=(d.broken_count>0)?"#e94560":"#00d4ff";
    const vs=document.getElementById("verify-status");
    if(d.verify_busy){{vs.textContent="🔍 自动核验中...";vs.className="busy";}}
    else{{vs.textContent="✓ 自动核验就绪";vs.className="ready";}}
    document.getElementById("conn-status").textContent="已连接";
    document.getElementById("conn-status").style.color="#4caf50";
  }}catch(e){{
    document.getElementById("conn-status").textContent="断开";
    document.getElementById("conn-status").style.color="#e94560";
  }}
}}
async function fetchHistory(){{
  try{{
    const r=await fetch("/api/history?limit=20");
    const data=await r.json();
    if(data.length===lastHist)return;
    lastHist=data.length;
    document.getElementById("hist-total").textContent=data.length;
    const list=document.getElementById("history-list");
    if(data.length===0){{list.innerHTML='<div style="color:#666;text-align:center;padding:20px">暂无核验记录</div>';return;}}
    list.innerHTML=data.map(h=>{{
      let tags="";
      if(h.correct_count>0)tags+=`<span class="tag tag-ok">确认${{h.correct_count}}</span>`;
      if(h.reject_count>0)tags+=`<span class="tag tag-err">误检${{h.reject_count}}</span>`;
      if(h.uncertain_count>0)tags+=`<span class="tag tag-unk">不确定${{h.uncertain_count}}</span>`;
      if(h.broken_count>0)tags+=`<span class="tag tag-def">Broken</span>`;
      return `<a href="/results?id=${{h.id}}" style="text-decoration:none;color:inherit">
        <div class="history-item" style="cursor:pointer">
          <div class="ts">${{h.timestamp}}</div>
          <div class="summary">检测:${{h.target_count}}个 &nbsp; ${{tags||"无异常"}}</div>
        </div></a>`;
    }}).join("");
  }}catch(e){{}}
}}
document.getElementById("nav-live").classList.add("active");
setInterval(fetchStatus,1000);setInterval(fetchHistory,3000);
fetchStatus();fetchHistory();
</script>
</body>
</html>"""

# ============================================================
# 第2页：核验结果页面 (/results)
# ============================================================
RESULTS_PAGE = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ELF2 核验结果</title>
<style>{_BASE_STYLE}
.container{{max-width:100vw;margin:0 auto;padding:16px;overflow:hidden}}
.toolbar{{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;margin-bottom:16px;gap:8px}}
.toolbar .total{{font-size:14px;color:#888;white-space:nowrap}}
.result-list{{display:flex;flex-direction:column;gap:12px}}
.result-card{{background:#1a1a2e;border-radius:10px;padding:18px;border:1px solid #16213e;display:flex;flex-wrap:wrap;gap:18px;cursor:pointer;transition:all .2s;overflow:hidden}}
.result-card:hover{{border-color:#e94560;background:#1e1e3a}}
.result-card .thumb{{width:200px;height:140px;background:#000;border-radius:6px;overflow:hidden;flex-shrink:0;display:flex;align-items:center;justify-content:center}}
.result-card .thumb img{{max-width:100%;max-height:100%;object-fit:cover}}
.result-card .thumb .no-img{{color:#555;font-size:13px}}
.result-card .info{{flex:1 1 240px;min-width:240px;overflow:auto}}
.result-card .info .row1{{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:4px 12px;margin-bottom:8px}}
.result-card .info .ts{{color:#888;font-size:13px}}
.result-card .info .summary{{font-size:14px;color:#e0e0e0;margin:6px 0;line-height:1.6;word-break:break-all}}
.result-card .info .assessment{{font-size:13px;color:#a0a0a0;margin-top:6px;line-height:1.5;word-break:break-all}}
.detail-overlay{{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.7);z-index:1000;overflow-y:auto}}
.detail-overlay.show{{display:flex;align-items:flex-start;justify-content:center;padding:30px}}
.detail-panel{{background:#1a1a2e;border-radius:12px;padding:24px;max-width:800px;width:95%;border:2px solid #e94560;margin-top:30px}}
.detail-panel .close{{float:right;background:none;border:none;color:#888;font-size:24px;cursor:pointer;padding:4px 10px}}
.detail-panel .close:hover{{color:#e94560}}
.detail-panel h2{{font-size:18px;color:#e94560;margin-bottom:16px}}
.detail-panel .snap-img{{max-width:100%;border-radius:8px;margin:10px 0;border:1px solid #333}}
.detail-panel .verification{{padding:8px 12px;margin:6px 0;border-radius:6px;font-size:13px}}
.detail-panel .verification.correct{{background:#0a3d0a;border-left:3px solid #4caf50}}
.detail-panel .verification.reject{{background:#3d0a0a;border-left:3px solid #e94560}}
.detail-panel .verification.uncertain{{background:#3d2e0a;border-left:3px solid #ff9800}}
.detail-panel .verification .cls{{font-weight:bold}}
.detail-panel .suggestion{{background:#16213e;padding:12px;border-radius:6px;margin-top:12px;font-size:13px;white-space:pre-wrap}}
.pagination{{display:flex;justify-content:center;align-items:center;gap:12px;margin-top:20px}}
.pagination button{{background:#1a1a2e;color:#e0e0e0;border:1px solid #333;padding:8px 20px;border-radius:6px;cursor:pointer;font-size:13px}}
.pagination button:hover{{border-color:#e94560;color:#e94560}}
.pagination button:disabled{{color:#444;border-color:#222;cursor:default}}
.pagination .page-info{{color:#888;font-size:13px}}
.empty{{text-align:center;color:#555;padding:60px 20px}}
</style>
</head>
<body>
{_NAV_HTML}
<div class="container">
  <div class="toolbar">
    <h2 style="color:#e94560;font-size:18px">📋 核验结果</h2>
    <span class="total" id="total-info">加载中...</span>
  </div>
  <div class="result-list" id="result-list">加载中...</div>
  <div class="pagination">
    <button id="btn-prev" onclick="prevPage()">← 上一页</button>
    <span class="page-info" id="page-info"></span>
    <button id="btn-next" onclick="nextPage()">下一页 →</button>
  </div>
</div>
<div class="detail-overlay" id="detail-overlay" onclick="closeDetail(event)">
  <div class="detail-panel" id="detail-panel" onclick="event.stopPropagation()"></div>
</div>
<script>
document.getElementById("nav-results").classList.add("active");
const PAGE_SIZE=10;let curPage=1,totalPages=1,allData=[];
async function loadAll(){{try{{
  const r=await fetch("/api/history");
  allData=await r.json();
  allData.reverse();
  totalPages=Math.ceil(allData.length/PAGE_SIZE)||1;
  if(curPage>totalPages)curPage=totalPages;
  document.getElementById("total-info").textContent=`共 ${{allData.length}} 条记录`;
  renderPage();
}}catch(e){{document.getElementById("result-list").innerHTML='<div class="empty">⚠ 加载失败</div>';}}}}
function prevPage(){{if(curPage>1){{curPage--;renderPage();}}}}
function nextPage(){{if(curPage<totalPages){{curPage++;renderPage();}}}}
function renderPage(){{
  const start=(curPage-1)*PAGE_SIZE;
  const page=allData.slice(start,start+PAGE_SIZE);
  document.getElementById("page-info").textContent=`${{curPage}} / ${{totalPages}} 页`;
  document.getElementById("btn-prev").disabled=curPage<=1;
  document.getElementById("btn-next").disabled=curPage>=totalPages;
  const list=document.getElementById("result-list");
  if(page.length===0){{list.innerHTML='<div class="empty">📭 暂无核验记录<br><small>当实时监控检测到绝缘子并完成AI核验后，这里将显示结果</small></div>';return;}}
  list.innerHTML=page.map(r=>{{
    let tags="";
    if(r.correct_count>0)tags+=`<span class="tag tag-ok">确认${{r.correct_count}}</span>`;
    if(r.reject_count>0)tags+=`<span class="tag tag-err">误检${{r.reject_count}}</span>`;
    if(r.uncertain_count>0)tags+=`<span class="tag tag-unk">不确定${{r.uncertain_count}}</span>`;
    if(r.broken_count>0)tags+=`<span class="tag tag-def">Broken</span>`;
    return `<div class="result-card" onclick="openDetail(${{r.id}})">
      <div class="thumb">${{r.snapshot?`<img src="/api/snapshot/${{r.id}}" alt="snap">`:r.has_snap?`<img src="/api/snapshot/${{r.id}}" onerror="this.parentElement.innerHTML='<span class=no-img>无图片</span>'">`:'<span class="no-img">无图片</span>'}}</div>
      <div class="info">
        <div class="row1"><span class="ts">${{r.timestamp}}</span>${{r.verify_time>0?`<small style="color:#666">核验耗时 ${{r.verify_time.toFixed(1)}}s</small>`:''}}</div>
        <div class="summary">📷 检测目标: <b>${{r.target_count}}</b> 个 &nbsp; ${{tags||"<span style='color:#888'>无异常</span>"}}</div>
        <div class="assessment">${{(r.overall_assessment||"").substring(0,120)}}</div>
      </div>
    </div>`;
  }}).join("");
}}
async function openDetail(id){{
  try{{
    const r=await fetch("/api/history/"+id);
    const d=await r.json();
    let html=`<button class="close" onclick="document.getElementById('detail-overlay').classList.remove('show')">×</button>
    <h2>${{d.timestamp}}</h2>`;
    if(d.snapshot||d.has_snap)html+=`<img class="snap-img" src="/api/snapshot/${{d.id}}" alt="现场截图">`;
    html+=`<div style="margin:12px 0;display:flex;flex-wrap:wrap;gap:6px">
      <span class="tag tag-ok">检测目标 ${{d.target_count}} 个</span>
      <span class="tag tag-ok">确认 ${{d.correct_count}}</span>
      <span class="tag tag-err">误检 ${{d.reject_count}}</span>
      <span class="tag tag-unk">不确定 ${{d.uncertain_count}}</span>
      ${{d.detect_time>0?`<span class="tag" style="background:#16213e;color:#888">检测 ${{(d.detect_time*1000).toFixed(0)}}ms</span>`:''}}
      ${{d.verify_time>0?`<span class="tag" style="background:#16213e;color:#888">核验 ${{d.verify_time.toFixed(1)}}s</span>`:''}}</div>`;
    if(d.overall_assessment)html+=`<div style="color:#ccc;font-size:14px;margin:8px 0;line-height:1.6">${{d.overall_assessment}}</div>`;
    if(d.verifications&&d.verifications.length>0){{
      html+='<h3 style="color:#00ff88;font-size:14px;margin:14px 0 8px">🔬 逐项核验</h3>';
      d.verifications.forEach((v,i)=>{{
        let cls=v.judgment||'uncertain';
        html+=`<div class="verification ${{cls}}">
          <span class="cls">#${{i+1}} ${{v.class_name||"目标"}}</span>
          &nbsp;→ ${{cls==='correct'?'✅ 检测正确':cls==='reject'?'❌ 误检':'⚠ 不确定'}}
          ${{v.reason?`<br><small style="color:#999">${{v.reason}}</small>`:''}}</div>`;
      }});
    }}
    document.getElementById("detail-panel").innerHTML=html;
    document.getElementById("detail-overlay").classList.add("show");
  }}catch(e){{alert("加载详情失败: "+e);}}
}}
function closeDetail(e){{if(e.target===document.getElementById("detail-overlay"))document.getElementById("detail-overlay").classList.remove("show");}}
document.addEventListener("keydown",e=>{{if(e.key==="Escape")document.getElementById("detail-overlay").classList.remove("show");}});
loadAll();
</script>
</body>
</html>"""

# ============================================================
# Flask Web 服务
# ============================================================
class WebServer:
    """轻量 Web 仪表盘, 与主程序共享数据"""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        if not _HAS_FLASK:
            raise ImportError("Flask 未安装, 请运行: pip install flask")

        self.host = host
        self.port = port
        self._app = Flask("elf2_web")
        self._thread: Optional[threading.Thread] = None

        # 共享数据引用 (由 MainWindow 设置)
        self._current_frame: Optional[np.ndarray] = None
        self._current_dets: List = []
        self._current_big_boxes: List = []
        self._current_fps: float = 0.0
        self._history: List[Dict] = []
        self._verify_busy: bool = False
        self._frame_lock = threading.Lock()

        self._setup_routes()

    def _setup_routes(self):
        app = self._app

        @app.route("/")
        def index():
            return LIVE_PAGE, 200, {"Content-Type": "text/html; charset=utf-8"}

        @app.route("/results")
        def results_page():
            return RESULTS_PAGE, 200, {"Content-Type": "text/html; charset=utf-8"}

        @app.route("/video_feed")
        def video_feed():
            return Response(self._generate_mjpeg(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.route("/api/status")
        def api_status():
            dets = self._current_dets
            broken = sum(1 for d in dets if int(d[5]) == 2)
            return jsonify({
                "det_count": len(dets),
                "broken_count": broken,
                "fps": self._current_fps,
                "verify_busy": self._verify_busy,
                "history_count": len(self._history),
            })

        @app.route("/api/history")
        def api_history():
            from flask import request
            limit = request.args.get("limit", 0, type=int)
            records = self._history[-50:] if not limit else self._history[-limit:]
            summaries = []
            for rec in reversed(records):
                broken = sum(1 for d in rec.get("detections", [])
                           if d.get("class_id") == 2)
                snap = rec.get("snapshot_path", "")
                summaries.append({
                    "id": rec.get("id", 0),
                    "timestamp": rec.get("timestamp", ""),
                    "target_count": rec.get("target_count", 0),
                    "broken_count": broken,
                    "correct_count": rec.get("correct_count", 0),
                    "reject_count": rec.get("reject_count", 0),
                    "uncertain_count": rec.get("uncertain_count", 0),
                    "overall_assessment": rec.get("overall_assessment", ""),
                    "verify_time": rec.get("verify_time_ms", 0),
                    "has_snap": bool(snap and Path(snap).is_file()),
                })
            return jsonify(summaries)

        @app.route("/api/history/<int:rec_id>")
        def api_history_detail(rec_id):
            """返回单条核验记录的完整详情"""
            rec = None
            for r in self._history:
                if r.get("id") == rec_id:
                    rec = r
                    break
            if rec is None:
                return jsonify({"error": "记录不存在"}), 404

            broken = sum(1 for d in rec.get("detections", [])
                       if d.get("class_id") == 2)
            snap = rec.get("snapshot_path", "")
            vers = []
            dets = rec.get("detections", [])
            for v in rec.get("verifications", []):
                ti = v.get("target_index", 1) - 1
                if 0 <= ti < len(dets):
                    cls_id = dets[ti].get("class_id", 0)
                    cls_name = {0:"正常", 1:"绝缘子", 2:"破损"}.get(cls_id, f"类{cls_id}")
                else:
                    cls_name = "目标"
                vers.append({
                    "class_name": cls_name,
                    "judgment": v.get("verification", "uncertain"),
                    "reason": v.get("defect_description", "") or v.get("reason", ""),
                })
            return jsonify({
                "id": rec.get("id", 0),
                "timestamp": rec.get("timestamp", ""),
                "target_count": rec.get("target_count", 0),
                "broken_count": broken,
                "correct_count": rec.get("correct_count", 0),
                "reject_count": rec.get("reject_count", 0),
                "uncertain_count": rec.get("uncertain_count", 0),
                "overall_assessment": rec.get("overall_assessment", ""),
                "maintenance_suggestion": rec.get("maintenance_suggestion", ""),
                "verifications": vers,
                "detect_time": rec.get("detect_time_ms", 0),
                "verify_time": rec.get("verify_time_ms", 0),
                "has_snap": bool(snap and Path(snap).is_file()),
            })

        @app.route("/api/snapshot/<int:rec_id>")
        def api_snapshot(rec_id):
            """返回核验记录的现场截图"""
            rec = None
            for r in self._history:
                if r.get("id") == rec_id:
                    rec = r
                    break
            if rec is None:
                return "Not found", 404
            snap = rec.get("snapshot_path", "")
            if not snap or not Path(snap).is_file():
                return "No image", 404
            return send_file(snap, mimetype="image/jpeg")

    def _draw_dashed_rect(self, img, pt1, pt2, color, thickness=2, dash_len=10, gap=6):
        """在图像上画虚线矩形"""
        x1, y1 = pt1
        x2, y2 = pt2
        # top
        for x in range(x1, x2, dash_len + gap):
            xe = min(x + dash_len, x2)
            cv2.line(img, (x, y1), (xe, y1), color, thickness)
        # bottom
        for x in range(x1, x2, dash_len + gap):
            xe = min(x + dash_len, x2)
            cv2.line(img, (x, y2), (xe, y2), color, thickness)
        # left
        for y in range(y1, y2, dash_len + gap):
            ye = min(y + dash_len, y2)
            cv2.line(img, (x1, y), (x1, ye), color, thickness)
        # right
        for y in range(y1, y2, dash_len + gap):
            ye = min(y + dash_len, y2)
            cv2.line(img, (x2, y), (x2, ye), color, thickness)

    def _generate_mjpeg(self):
        """MJPEG 流生成器: 每帧用 JPEG 编码推送"""
        while True:
            with self._frame_lock:
                if self._current_frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Waiting...", (180, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    frame = self._current_frame.copy()

                dets = list(self._current_dets)
                big_boxes = list(self._current_big_boxes)

            # 画大框 (绝缘子串包围框) - 虚线, 只用于可视化
            for b in big_boxes:
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                self._draw_dashed_rect(frame, (x1, y1), (x2, y2),
                                       (0, 255, 0), thickness=3, dash_len=12, gap=6)
                cv2.putText(frame, "绝缘子串", (x1, max(10, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 画小框 (独立绝缘子)
            for d in dets:
                x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                cls = int(d[5])
                score = float(d[4])
                # 颜色: 0=绿, 1=蓝, 2=红
                if cls == 2:
                    color = (0, 0, 255)
                elif cls == 1:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"Broken {score:.2f}" if cls == 2 else \
                        f"JYZ {score:.2f}" if cls == 1 else \
                        f"Nora1 {score:.2f}"
                cv2.putText(frame, label, (x1, max(10, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 缩小以节省带宽 (最长边 960)
            h, w = frame.shape[:2]
            if max(h, w) > 960:
                scale = 960 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

            time.sleep(0.04)  # ~25 FPS

    def update_state(self, frame: np.ndarray, dets: List, fps: float = 0.0,
                     verify_busy: bool = False, big_boxes: List = None):
        """主线程调用, 更新共享状态"""
        with self._frame_lock:
            self._current_frame = frame
        self._current_dets = dets
        self._current_big_boxes = big_boxes or []
        self._current_fps = fps
        self._verify_busy = verify_busy

    def start(self):
        """在后台线程启动 Flask"""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[WEB] 仪表盘已启动 → http://{self.host}:{self.port}")

    def _run(self):
        """Flask 运行 (关闭 debug/reloader, 适配生产环境)"""
        try:
            from werkzeug.serving import make_server
            server = make_server(self.host, self.port, self._app, threaded=True)
            server.serve_forever()
        except ImportError:
            self._app.run(host=self.host, port=self.port,
                          debug=False, use_reloader=False, threaded=True)
