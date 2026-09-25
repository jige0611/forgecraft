"""结构化关系库 — SQLite 精确查询与聚合

用途:
  - 进化轨迹查询: "第 50-100 代之间 QD-score 平均增长率"
  - 失败模式统计: "哪种修复方法成功率最高?"
  - 元数据聚合: "speed 任务的平均最佳适应度"
  - 参数-性能回归: "population_size 和 qd_score 的关系"

Schema:
  - design_cases:       设计案例主表 (与 SemanticKB 关联)
  - evolution_trajectories: 逐代进化记录
  - failure_patterns:   已知失败模式 + 修复历史
  - parameter_history:  参数调整日志 (for 因果推理)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from forgecraft.logging import get_logger

_logger = get_logger(__name__)

# ── Schema ───────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS design_cases (
    case_id         TEXT PRIMARY KEY,
    task_name       TEXT NOT NULL,
    catalog_name    TEXT NOT NULL,
    objectives_json TEXT,             -- JSON {speed: 2.3, ...}
    behavior_bc_json TEXT,            -- JSON [0.8, 0.3, ...]
    morphology_hash TEXT,
    evo_config_json TEXT,
    generation_count INTEGER,
    wall_time_seconds REAL,
    qd_score_final  REAL,
    coverage_final  REAL,
    best_fitness    REAL,
    best_individual_json TEXT,
    tags            TEXT,             -- JSON ["fast", "stable"]
    notes           TEXT,
    run_id          TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS evolution_trajectories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         TEXT NOT NULL,
    generation      INTEGER NOT NULL,
    qd_score        REAL,
    coverage        REAL,
    best_fitness    REAL,
    population_diversity REAL,
    stagnation_flag INTEGER DEFAULT 0,
    stagnation_duration INTEGER DEFAULT 0,
    active_strategy TEXT,
    emitter_usage_json TEXT,          -- {"improvement": 5, "random": 3, ...}
    num_elites      INTEGER,
    wall_time_s     REAL,
    FOREIGN KEY (case_id) REFERENCES design_cases(case_id),
    UNIQUE(case_id, generation)
);

CREATE TABLE IF NOT EXISTS failure_patterns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_name    TEXT NOT NULL UNIQUE,
    description     TEXT,
    detection_rule  TEXT,
    suggested_actions_json TEXT,       -- JSON [action1, action2, ...]
    success_count   INTEGER DEFAULT 0,
    failure_count   INTEGER DEFAULT 0,
    total_attempts  INTEGER DEFAULT 0,
    avg_recovery_gens REAL,
    source          TEXT DEFAULT 'system'  -- 'system', 'human', 'llm'
);

CREATE TABLE IF NOT EXISTS failure_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         TEXT NOT NULL,
    generation      INTEGER NOT NULL,
    pattern_name    TEXT,
    action_taken    TEXT,
    recovery_success INTEGER DEFAULT 0,  -- 0 = no, 1 = yes
    notes           TEXT,
    FOREIGN KEY (case_id) REFERENCES design_cases(case_id)
);

CREATE TABLE IF NOT EXISTS parameter_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         TEXT NOT NULL,
    generation      INTEGER NOT NULL,
    param_name      TEXT NOT NULL,
    old_value       REAL,
    new_value       REAL,
    reason          TEXT,
    source          TEXT DEFAULT 'system',  -- 'system', 'human', 'llm'
    FOREIGN KEY (case_id) REFERENCES design_cases(case_id)
);

CREATE INDEX IF NOT EXISTS idx_traj_case ON evolution_trajectories(case_id);
CREATE INDEX IF NOT EXISTS idx_traj_gen ON evolution_trajectories(generation);
CREATE INDEX IF NOT EXISTS idx_cases_task ON design_cases(task_name);
CREATE INDEX IF NOT EXISTS idx_failure_pattern ON failure_events(pattern_name);
CREATE INDEX IF NOT EXISTS idx_param_case ON parameter_history(case_id);
"""


class RelationalKB:
    """结构化关系知识库

    基于 SQLite, 支持:
      - 设计案例 CRUD
      - 进化轨迹记录与查询
      - 失败模式学习 (统计成功率)
      - 参数历史追踪 (因果关系推理)
      - 聚合统计查询
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "knowledge.db",
            )
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        _logger.info(f"RelationalKB initialized: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._init_db()
        return self._conn

    # ── 设计案例 ───────────────────────────────────────────

    def insert_design_case(self, case_data: Dict) -> str:
        """插入设计案例记录"""
        conn = self._get_conn()
        case_id = case_data.get("case_id", f"case_{int(time.time()*1000)}")

        conn.execute(
            """INSERT OR REPLACE INTO design_cases
               (case_id, task_name, catalog_name, objectives_json,
                behavior_bc_json, morphology_hash, evo_config_json,
                generation_count, wall_time_seconds, qd_score_final,
                coverage_final, best_fitness, best_individual_json,
                tags, notes, run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case_id,
                case_data.get("task_name", "unknown"),
                case_data.get("catalog_name", "default"),
                json.dumps(case_data.get("objectives", {})),
                json.dumps(case_data.get("behavior_bc", [])),
                case_data.get("morphology_hash", ""),
                json.dumps(case_data.get("evolution_config", {})),
                case_data.get("generation_count", 0),
                case_data.get("wall_time_seconds", 0.0),
                case_data.get("qd_score_final", 0.0),
                case_data.get("coverage_final", 0.0),
                case_data.get("best_fitness", 0.0),
                json.dumps(case_data.get("best_individual", {})),
                json.dumps(case_data.get("tags", [])),
                case_data.get("notes", ""),
                case_data.get("run_id", ""),
            ),
        )
        conn.commit()
        return case_id

    def get_design_case(self, case_id: str) -> Optional[Dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM design_cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_case(row)

    def list_cases_by_task(self, task_name: str, limit: int = 50) -> List[Dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM design_cases WHERE task_name = ? "
            "ORDER BY qd_score_final DESC LIMIT ?",
            (task_name, limit),
        ).fetchall()
        return [self._row_to_case(r) for r in rows]

    def get_aggregate_stats(self, task_name: Optional[str] = None) -> Dict:
        conn = self._get_conn()
        if task_name:
            row = conn.execute(
                """SELECT COUNT(*) as n, AVG(qd_score_final) as avg_qd,
                   MAX(qd_score_final) as max_qd, AVG(coverage_final) as avg_cov,
                   AVG(generation_count) as avg_gen, AVG(wall_time_seconds) as avg_time
                   FROM design_cases WHERE task_name = ?""",
                (task_name,),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT COUNT(*) as n, AVG(qd_score_final) as avg_qd,
                   MAX(qd_score_final) as max_qd, AVG(coverage_final) as avg_cov,
                   AVG(generation_count) as avg_gen, AVG(wall_time_seconds) as avg_time
                   FROM design_cases""",
            ).fetchone()

        return dict(row) if row else {}

    # ── 进化轨迹 ───────────────────────────────────────────

    def record_generation(self, case_id: str, gen: int, metrics: Dict):
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO evolution_trajectories
               (case_id, generation, qd_score, coverage, best_fitness,
                population_diversity, stagnation_flag, stagnation_duration,
                active_strategy, emitter_usage_json, num_elites, wall_time_s)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case_id, gen,
                metrics.get("qd_score", 0.0),
                metrics.get("coverage", 0.0),
                metrics.get("best_fitness", 0.0),
                metrics.get("population_diversity", 0.0),
                int(metrics.get("stagnation_flag", False)),
                metrics.get("stagnation_duration", 0),
                metrics.get("active_strategy", ""),
                json.dumps(metrics.get("emitter_usage", {})),
                metrics.get("num_elites", 0),
                metrics.get("wall_time_s", 0.0),
            ),
        )
        conn.commit()

    def get_trajectory(self, case_id: str) -> List[Dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM evolution_trajectories WHERE case_id = ? ORDER BY generation",
            (case_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_qd_score_history(self, case_id: str) -> List[float]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT qd_score FROM evolution_trajectories WHERE case_id = ? ORDER BY generation",
            (case_id,),
        ).fetchall()
        return [r["qd_score"] for r in rows]

    # ── 失败模式 ────────────────────────────────────────────

    def register_failure_pattern(self, pattern: Dict):
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO failure_patterns
               (pattern_name, description, detection_rule,
                suggested_actions_json, source)
               VALUES (?, ?, ?, ?, ?)""",
            (
                pattern["name"],
                pattern.get("description", ""),
                pattern.get("detection_rule", ""),
                json.dumps(pattern.get("suggested_actions", [])),
                pattern.get("source", "system"),
            ),
        )
        conn.commit()

    def record_failure_event(
        self, case_id: str, generation: int,
        pattern_name: str, action_taken: str,
        recovery_success: bool,
    ):
        """记录一次失败检测与修复尝试"""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO failure_events
               (case_id, generation, pattern_name, action_taken, recovery_success)
               VALUES (?, ?, ?, ?, ?)""",
            (case_id, generation, pattern_name, action_taken, int(recovery_success)),
        )

        # 更新统计
        conn.execute(
            """UPDATE failure_patterns
               SET success_count = success_count + ?,
                   failure_count = failure_count + ?,
                   total_attempts = total_attempts + 1
               WHERE pattern_name = ?""",
            (
                int(recovery_success),
                int(not recovery_success),
                pattern_name,
            ),
        )
        conn.commit()

    def get_failure_stats(self) -> List[Dict]:
        """获取所有失败模式的统计"""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT pattern_name, description, success_count, failure_count,
                      total_attempts,
                      CASE WHEN total_attempts > 0
                           THEN CAST(success_count AS REAL) / total_attempts
                           ELSE 0 END as success_rate
               FROM failure_patterns ORDER BY total_attempts DESC""",
        ).fetchall()
        return [dict(r) for r in rows]

    def get_best_recovery_action(self, pattern_name: str) -> Optional[Dict]:
        """对指定失败模式, 找到成功率最高的修复方案"""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT action_taken, COUNT(*) as cnt,
                      SUM(recovery_success) as successes,
                      CAST(SUM(recovery_success) AS REAL) / COUNT(*) as rate
               FROM failure_events
               WHERE pattern_name = ?
               GROUP BY action_taken
               ORDER BY rate DESC
               LIMIT 1""",
            (pattern_name,),
        ).fetchone()
        return dict(rows) if rows else None

    # ── 参数历史 ────────────────────────────────────────────

    def record_param_change(
        self, case_id: str, generation: int,
        param_name: str, old_value: float, new_value: float,
        reason: str = "", source: str = "llm",
    ):
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO parameter_history
               (case_id, generation, param_name, old_value, new_value, reason, source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (case_id, generation, param_name, old_value, new_value, reason, source),
        )
        conn.commit()

    def get_param_impact(self, param_name: str, task_name: Optional[str] = None) -> List[Dict]:
        """查询某参数的历史调整与效果 (用于因果分析)"""
        conn = self._get_conn()
        if task_name:
            rows = conn.execute(
                """SELECT ph.*, dc.qd_score_final
                   FROM parameter_history ph
                   JOIN design_cases dc ON ph.case_id = dc.case_id
                   WHERE ph.param_name = ? AND dc.task_name = ?
                   ORDER BY ph.generation""",
                (param_name, task_name),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT ph.*, dc.qd_score_final
                   FROM parameter_history ph
                   JOIN design_cases dc ON ph.case_id = dc.case_id
                   WHERE ph.param_name = ?
                   ORDER BY ph.generation""",
                (param_name,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── 帮助方法 ────────────────────────────────────────────

    def _row_to_case(self, row) -> Dict:
        d = dict(row)
        for json_field in ["objectives_json", "behavior_bc_json",
                           "evo_config_json", "best_individual_json", "tags"]:
            if d.get(json_field):
                try:
                    d[json_field.replace("_json", "")] = json.loads(d[json_field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
