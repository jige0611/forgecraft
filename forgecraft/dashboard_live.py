"""
实时进化看板 — WebSocket 推送 + HTTP 轮询兜底

使用:
  python -m forgecraft.dashboard_live

特性:
  - 外部队列 push_stats 写入，polling/tail 读取
  - 进度条 + 预计剩余时间
  - 3 秒自动刷新
  - WebSocket 实时推送 (优先)
  - HTTP /api/status 轮询 (兜底)

依赖: fastapi, uvicorn
"""

import json
import time
import threading
from typing import Dict, List

__all__ = ["LiveDashboard", "push_stats", "get_status", "get_dashboard_state"]

# ═══════════════════════════════════════
# 线程安全共享状态 (进化线程写入, 服务器读取)
# ═══════════════════════════════════════
_STATE_LOCK = threading.Lock()
_SHARED_STATE: Dict = {
    "generation": 0,
    "total_generations": 0,
    "best_fitness": 0.0,
    "mean_fitness": 0.0,
    "population_size": 0,
    "pareto_front": 0,
    "best_parts": 0,
    "start_time": time.time(),
    "start_gen": 0,            # 看板启动时的代数
    "gen_times": [],           # 最近 10 代每代的时间戳
    "history": [],             # [(gen, best, avg), ...]
    "logs": [],                # 最近日志
    "status": "waiting",       # waiting | running | done
    "task_name": "",
    "catalog_name": "",
}
_MAX_LOGS = 50
_MAX_HISTORY = 500


def push_stats(generation: int, best_fitness: float,
               mean_fitness: float, population_size: int,
               pareto_front: int = 0, best_parts: int = 0,
               total_generations: int = 0):
    """进化线程调用：写入共享状态"""
    with _STATE_LOCK:
        now = time.time()
        # 去重: 同一代不重复记录时间戳
        if _SHARED_STATE["generation"] != generation:
            _SHARED_STATE["generation"] = generation
            gt = _SHARED_STATE["gen_times"]
            gt.append(now)
            if len(gt) > 10:
                gt[:] = gt[-10:]
        if total_generations:
            _SHARED_STATE["total_generations"] = total_generations
        _SHARED_STATE["best_fitness"] = round(best_fitness, 4)
        _SHARED_STATE["mean_fitness"] = round(mean_fitness, 4)
        _SHARED_STATE["population_size"] = population_size
        _SHARED_STATE["pareto_front"] = pareto_front
        _SHARED_STATE["best_parts"] = best_parts
        _SHARED_STATE["status"] = "running"
        _SHARED_STATE["history"].append((generation, round(best_fitness, 4), round(mean_fitness, 4)))
        if len(_SHARED_STATE["history"]) > _MAX_HISTORY:
            _SHARED_STATE["history"] = _SHARED_STATE["history"][-_MAX_HISTORY:]


def push_log(msg: str):
    """进化线程调用：写入日志"""
    with _STATE_LOCK:
        _SHARED_STATE["logs"].append(msg)
        if len(_SHARED_STATE["logs"]) > _MAX_LOGS:
            _SHARED_STATE["logs"] = _SHARED_STATE["logs"][-_MAX_LOGS:]


def set_config(total_generations: int, task_name: str = "", catalog_name: str = "",
               current_generation: int = 0):
    with _STATE_LOCK:
        _SHARED_STATE["total_generations"] = total_generations
        _SHARED_STATE["task_name"] = task_name
        _SHARED_STATE["catalog_name"] = catalog_name
        _SHARED_STATE["start_time"] = time.time()
        _SHARED_STATE["start_gen"] = current_generation
        _SHARED_STATE["status"] = "running"


def set_done():
    with _STATE_LOCK:
        _SHARED_STATE["status"] = "done"


def get_status() -> Dict:
    """HTTP 端点调用：返回含 ETA 的完整状态"""
    with _STATE_LOCK:
        s = dict(_SHARED_STATE)
    # 计算 ETA — 用最近 N 代的滑动窗口速率，避免断点恢复导致误判
    gt = s.get("gen_times", [])
    now = time.time()
    # 已用时间 = 实时时钟差（每 3 秒轮询时自动更新）
    elapsed = now - s["start_time"]
    if elapsed < 0:
        elapsed = 0
    s["elapsed_seconds"] = round(elapsed, 1)
    s["elapsed_str"] = _fmt_time(elapsed)
    
    if len(gt) >= 2 and s["generation"] > 0 and s["total_generations"] > 0:
        # 滑动窗口: 用最近 gen_times 的时间差 / 代数差
        first_gt = gt[0]
        last_gt = gt[-1]
        window_gens = max(len(gt) - 1, 1)  # 窗口内代际数
        window_elapsed = last_gt - first_gt
        if window_elapsed > 0 and window_gens > 0:
            sec_per_gen = window_elapsed / window_gens
        else:
            sec_per_gen = 0
        gens_total = s["total_generations"]
        gens_left = gens_total - s["generation"]
        eta_seconds = sec_per_gen * gens_left
        s["eta_seconds"] = round(eta_seconds, 1)
        s["eta_str"] = _fmt_time(eta_seconds)
    else:
        s["eta_seconds"] = 0
        s["eta_str"] = "--"
    s["progress_pct"] = round(s["generation"] / max(s["total_generations"], 1) * 100, 1)
    return s


def get_dashboard_state() -> Dict:
    """兼容旧 API"""
    return get_status()


def _fmt_time(sec: float) -> str:
    if sec < 60:
        return f"{int(sec)}s"
    m = int(sec // 60)
    s = int(sec % 60)
    if m < 60:
        return f"{m}:{s:02d}"
    h = m // 60
    m = m % 60
    return f"{h}:{m:02d}:{s:02d}"


# ═══════════════════════════════════════
# HTML 页面
# ═══════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ForgeCraft 进化看板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #0d1117; color: #c9d1d9; }
.header { background: #161b22; padding: 14px 24px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
.header h1 { font-size: 20px; color: #58a6ff; }
.header .info { font-size: 12px; color: #8b949e; }
.badges { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.badge { font-size: 13px; padding: 4px 12px; border-radius: 12px; }
.badge.ws { background: #238636; color: #fff; }
.badge.ws.off { background: #30363d; }
.badge.running { background: #1f6feb; color: #fff; }
.badge.done { background: #238636; color: #fff; }
.badge.waiting { background: #30363d; color: #8b949e; }
/* 进度条 */
.progress-wrap { grid-column: span 2; }
.progress-bar-outer { height: 6px; background: #21262d; border-radius: 3px; margin-top: 8px; overflow: hidden; }
.progress-bar-inner { height: 100%; background: linear-gradient(90deg, #1f6feb, #3fb950); border-radius: 3px; transition: width 0.5s ease; }
.progress-labels { display: flex; justify-content: space-between; font-size: 12px; color: #8b949e; margin-bottom: 4px; }
.progress-labels .eta { color: #d29922; }
.grid { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; padding: 16px; max-width: 1400px; margin: 0 auto; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
.card h3 { font-size: 14px; color: #8b949e; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
.chart-container { position: relative; height: 280px; }
.stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
.stat { text-align: center; padding: 10px; border-radius: 6px; background: #0d1117; }
.stat .label { font-size: 11px; color: #8b949e; }
.stat .value { font-size: 20px; font-weight: bold; margin-top: 2px; }
.stat .value.green { color: #3fb950; }
.stat .value.blue { color: #58a6ff; }
.stat .value.orange { color: #d29922; }
.stat .value.red { color: #f85149; }
.messages { height: 180px; overflow-y: auto; font-family: 'Cascadia Code', monospace; font-size: 11px; padding: 8px; background: #0d1117; border-radius: 4px; }
.messages .msg { padding: 1px 0; border-bottom: 1px solid #1c2128; color: #8b949e; }
.refresh-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #3fb950; animation: pulse 1s infinite; margin-right: 4px; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>ForgeCraft <span style="font-weight:normal; font-size:14px;">进化实时看板</span></h1>
    <div class="info" id="task-info">--</div>
  </div>
  <div class="badges">
    <span id="refresh-indicator"><span class="refresh-dot"></span></span>
    <span id="conn-badge" class="badge waiting">等待数据</span>
    <span>代数: <b id="gen-num" style="color:#58a6ff;">0</b></span>
    <span>最佳: <b id="best-fit" style="color:#3fb950;">--</b></span>
  </div>
</div>
<div style="padding: 4px 24px; font-size: 11px; color: #8b949e; border-bottom: 1px solid #30363d; text-align: right;" id="last-update">等待首次更新...</div>
<div class="grid">
  <div class="card progress-wrap">
    <div class="progress-labels">
      <span>进度: <b id="prog-txt">0%</b></span>
      <span>已用: <b id="elapsed">--</b></span>
      <span class="eta">预计剩余: <b id="eta">--</b></span>
    </div>
    <div class="progress-bar-outer">
      <div class="progress-bar-inner" id="prog-bar" style="width: 0%;"></div>
    </div>
  </div>
  <div class="card">
    <h3>适应度曲线</h3>
    <div class="chart-container"><canvas id="fitness-chart"></canvas></div>
  </div>
  <div class="card">
    <h3>实时统计</h3>
    <div class="stats-grid">
      <div class="stat"><div class="label">种群大小</div><div id="pop-size" class="value blue">--</div></div>
      <div class="stat"><div class="label">平均适应度</div><div id="avg-fit" class="value">--</div></div>
      <div class="stat"><div class="label">最优零件数</div><div id="best-parts" class="value green">--</div></div>
      <div class="stat"><div class="label">每代耗时</div><div id="gen-time" class="value orange">--</div></div>
    </div>
  </div>
  <div class="card" style="grid-column: span 2;">
    <h3>日志</h3>
    <div id="log-messages" class="messages"><div class="msg">等待进化数据...</div></div>
  </div>
</div>
<script>
// ══════════════════════════════════════════
// Chart 初始化
// ══════════════════════════════════════════
let fitnessChart = null;
let hasInit = false;
let bestData = [], avgData = [], genLabels = [];
let pollTimer = null;

function initChart() {
  if (hasInit) return;
  const ctx = document.getElementById('fitness-chart');
  if (!ctx) return;
  fitnessChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [
      { label: '最佳', borderColor: '#3fb950', data: [], tension: 0.3, pointRadius: 0, borderWidth: 2 },
      { label: '平均', borderColor: '#58a6ff', data: [], tension: 0.3, pointRadius: 0, borderDash: [4,4], borderWidth: 1.5 }
    ]},
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 200 },
      scales: {
        x: { ticks: { color: '#8b949e', maxTicksLimit: 20 }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
      },
      plugins: { legend: { labels: { color: '#8b949e', usePointStyle: true } } }
    }
  });
  hasInit = true;
}

// ══════════════════════════════════════════
// 数据更新
// ══════════════════════════════════════════
let lastGen = 0;
let lastUpdateTime = Date.now();
let updateCount = 0;

function updateUI(data) {
  updateCount++;
  lastUpdateTime = Date.now();
  const secAgo = Math.round((Date.now() - lastUpdateTime) / 1000);
  document.getElementById('last-update').textContent = 
    '更新 #' + updateCount + ' (' + new Date().toLocaleTimeString() + ')';
  
  const gen = data.generation || 0;
  if (gen > lastGen) {
    // 新的代 — 闪烁刷新指示器
    const dot = document.getElementById('refresh-indicator');
    dot.innerHTML = '<span class="refresh-dot"></span>';
    setTimeout(() => { dot.innerHTML = '<span class="refresh-dot"></span>'; }, 300);
    lastGen = gen;
  }

  document.getElementById('gen-num').textContent = gen;
  document.getElementById('best-fit').textContent = (data.best_fitness || 0).toFixed(4);
  document.getElementById('pop-size').textContent = data.population_size || '--';
  document.getElementById('avg-fit').textContent = (data.mean_fitness || 0).toFixed(4);
  document.getElementById('best-parts').textContent = data.best_parts || '--';
  document.getElementById('elapsed').textContent = data.elapsed_str || '--';
  document.getElementById('eta').textContent = data.eta_str || '--';

  // 进度条
  const pct = data.progress_pct || 0;
  document.getElementById('prog-txt').textContent = pct.toFixed(0) + '%';
  document.getElementById('prog-bar').style.width = pct + '%';

  // 每代耗时
  if (data.generation > 0 && data.elapsed_seconds > 0) {
    const secPerGen = data.elapsed_seconds / data.generation;
    document.getElementById('gen-time').textContent = secPerGen.toFixed(1) + 's';
  }

  // 状态标签
  const badge = document.getElementById('conn-badge');
  if (data.status === 'done') {
    badge.textContent = '已完成';
    badge.className = 'badge done';
  } else if (data.status === 'running') {
    badge.textContent = '运行中';
    badge.className = 'badge running';
  } else {
    badge.textContent = '等待中';
    badge.className = 'badge waiting';
  }

  // 任务信息
  const info = [];
  if (data.task_name) info.push('任务: ' + data.task_name);
  if (data.catalog_name) info.push('零件箱: ' + data.catalog_name);
  if (data.total_generations) info.push('目标: ' + data.total_generations + ' 代');
  document.getElementById('task-info').textContent = info.join('  |  ') || '--';

  // 图表
  initChart();
  if (fitnessChart && gen > 0) {
    if (genLabels.length === 0 || genLabels[genLabels.length - 1] !== gen) {
      genLabels.push(gen);
      bestData.push(data.best_fitness || 0);
      avgData.push(data.mean_fitness || 0);
      fitnessChart.data.labels = genLabels;
      fitnessChart.data.datasets[0].data = bestData;
      fitnessChart.data.datasets[1].data = avgData;
      fitnessChart.update('none');
    }
  }

  // 日志
  if (data.logs && data.logs.length > 0) {
    const el = document.getElementById('log-messages');
    el.innerHTML = data.logs.map(l => '<div class="msg">' + l.replace(/</g,'&lt;') + '</div>').join('');
    el.scrollTop = el.scrollHeight;
  }
}

// ══════════════════════════════════════════
// HTTP 轮询 (每 3 秒, 兜底)
// ══════════════════════════════════════════
function pollStatus() {
  fetch('/api/status')
    .then(r => r.json())
    .then(data => {
      updateUI(data);
    })
    .catch(err => {
      document.getElementById('conn-badge').textContent = '等待连接';
      document.getElementById('conn-badge').className = 'badge waiting';
    });
}

// ══════════════════════════════════════════
// WebSocket (优先, 实时推送)
// ══════════════════════════════════════════
let useWS = false;
try {
  const ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onopen = () => { useWS = true; console.log('WS connected'); };
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'stats' || data.type === 'full_status') {
        updateUI(data);
      }
    } catch(e) {}
  };
  ws.onclose = () => { useWS = false; };
  ws.onerror = () => { useWS = false; };
} catch(e) { useWS = false; }

// ══════════════════════════════════════════
// 启动
// ══════════════════════════════════════════
pollStatus();  // 立即拉一次
pollTimer = setInterval(pollStatus, 3000);  // 每 3 秒轮询

// 页面关闭时清理
window.addEventListener('beforeunload', () => {
  if (pollTimer) clearInterval(pollTimer);
});
</script>
</body>
</html>"""


# ═══════════════════════════════════════
# LiveDashboard 服务器类
# ═══════════════════════════════════════

class LiveDashboard:
    """实时进化看板服务端 (FastAPI + WebSocket + HTTP 轮询)"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._clients: List = []
        self._lock = threading.Lock()

    def broadcast(self, data: dict):
        """WebSocket 广播"""
        with self._lock:
            dead = []
            for ws in self._clients:
                try:
                    ws.send(json.dumps(data))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.remove(ws)

    def push_stats(self, generation: int, best_fitness: float,
                   mean_fitness: float, population_size: int,
                   pareto_front: int = 0, best_parts: int = 0):
        """向 WebSocket 客户端推送统计 (同时写入共享状态)"""
        with _STATE_LOCK:
            _SHARED_STATE["generation"] = generation
            _SHARED_STATE["best_fitness"] = round(best_fitness, 4)
            _SHARED_STATE["mean_fitness"] = round(mean_fitness, 4)
            _SHARED_STATE["population_size"] = population_size
            _SHARED_STATE["pareto_front"] = pareto_front
            _SHARED_STATE["best_parts"] = best_parts
            _SHARED_STATE["status"] = "running"
        self.broadcast({
            "type": "stats",
            "generation": generation,
            "best_fitness": round(best_fitness, 4),
            "mean_fitness": round(mean_fitness, 4),
            "population_size": population_size,
            "pareto_front": pareto_front,
            "best_parts": best_parts,
        })

    def push_log(self, msg: str):
        self.broadcast({"type": "log", "msg": msg})

    def _setup_routes(self, app):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse

        @app.get("/")
        async def index():
            return HTMLResponse(DASHBOARD_HTML)

        @app.get("/api/status")
        async def api_status():
            return get_status()

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            with self._lock:
                self._clients.append(websocket)
            try:
                await websocket.send_json({"type": "init"})
                # 推送当前状态
                s = get_status()
                s["type"] = "full_status"
                await websocket.send_json(s)
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                with self._lock:
                    if websocket in self._clients:
                        self._clients.remove(websocket)

    def serve(self):
        """启动 WebSocket 服务器 (阻塞)"""
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        self._setup_routes(app)
        print(f"[Dashboard] 进化看板: http://{self.host}:{self.port}")
        if self.host == "0.0.0.0":
            print(f"[Dashboard] 局域网访问: http://localhost:{self.port}")
        uvicorn.run(app, host=self.host, port=self.port, log_level="warning")


# ── 快捷启动 ──────────────────────────────────────
def serve(port: int = 8080):
    dashboard = LiveDashboard(port=port)
    dashboard.serve()


if __name__ == "__main__":
    serve()
