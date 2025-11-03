# -*- coding: utf-8 -*-
"""
baseline_manager.py
高级自适应版的“分层融合 + 门控更新”被试级基线管理器
- 以 subject_base 为键（如 shh）管理基线：{med_tb, mad_tb, med_ta, mad_ta, meta}
- 门控更新条件（建议最低要求）：
  1) 信号质量好：肌电/大幅度伪迹比例 < 30%（可传 qc["bad_ratio"]）
  2) 处于清醒/低负荷：可用你的业务规则或概率（如在线负荷<0.3）传 qc["is_lowload"]=True
  3) 候选统计“合理”：与历史分布差距不过大（内部做百分比/倍数约束）
- 分层融合：
  - 若跨天：先做“昨天基线 + 今天短基线”的加权融合，alpha_day较小（如0.2）
  - 再做全局历史EMA低速更新，alpha_ema更小（如0.05）
- 硬边界：每次更新后，中心值最多±20%；MAD 不低于历史MAD的30%且≥1e-6
"""

import os, json, time

DEFAULTS = dict(
    alpha_day = 0.2,   # 跨天融合系数（今日短基线 vs 昨日）
    alpha_ema = 0.05,  # 全历史EMA低速更新
    max_drift = 0.20,  # 单次中心值最大漂移 ±20%
    min_mad_frac = 0.30, # MAD 不低于历史MAD的30%
    mad_floor = 1e-6
)

class BaselineManager:
    def __init__(self, model_dir: str):
        self.dir = os.path.join(model_dir, "baselines")
        os.makedirs(self.dir, exist_ok=True)
        self.index_path = os.path.join(self.dir, "_index.json")
        self.index = self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            try:
                return json.load(open(self.index_path,"r",encoding="utf-8"))
            except:
                pass
        return {}

    def _save_index(self):
        json.dump(self.index, open(self.index_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    def _path(self, subject_base: str) -> str:
        return os.path.join(self.dir, f"{subject_base}.json")

    def get(self, subject_base: str):
        p = self._path(subject_base)
        if os.path.exists(p):
            return json.load(open(p,"r",encoding="utf-8"))
        return None

    def save(self, subject_base: str, stats: dict):
        stats = dict(stats)
        stats.setdefault("meta", {})
        stats["meta"]["ts"] = time.time()
        json.dump(stats, open(self._path(subject_base),"w",encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        self.index[subject_base] = dict(updated_at=stats["meta"]["ts"])
        self._save_index()

    # 门控：需满足低负荷、好信号、统计不过离谱
    def _pass_gate(self, current: dict|None, candidate: dict, qc: dict|None):
        if qc is None: qc={}
        # 低负荷门控（外部给出）
        if not qc.get("is_lowload", True):
            return False, "gate_lowload_false"
        # 伪迹门控（外部给出 bad_ratio）
        if qc.get("bad_ratio", 0.0) > 0.30:
            return False, "gate_bad_ratio_high"
        # 基本数值合理性
        if candidate["mad_tb"] < DEFAULTS["mad_floor"] or candidate["mad_ta"] < DEFAULTS["mad_floor"]:
            return False, "gate_mad_floor"
        if current:
            # 和历史差距不过大（中心值不超过5倍，MAD不超过5倍）
            for k in ["med_tb","med_ta","mad_tb","mad_ta"]:
                if candidate[k] <= 0 or (current[k] > 0 and candidate[k] > 5*current[k]):
                    return False, f"gate_{k}_too_large"
        return True, "ok"

    # 硬边界束缚
    def _clamp(self, old: float, new: float, frac: float):
        lo = old*(1-frac); hi = old*(1+frac)
        if hi < lo: lo,hi = hi,lo
        return max(lo, min(hi, new))

    # 分层融合 + 低速更新 + 硬边界
    def gated_update(self, subject_base: str, candidate: dict, qc: dict|None=None,
                     alpha_day: float|None=None, alpha_ema: float|None=None):
        cur = self.get(subject_base)
        ok, reason = self._pass_gate(cur, candidate, qc)
        if not ok:
            return dict(updated=False, reason=reason, stats=cur or candidate)

        a_day = DEFAULTS["alpha_day"] if alpha_day is None else alpha_day
        a_ema = DEFAULTS["alpha_ema"] if alpha_ema is None else alpha_ema

        if cur is None:
            # 首次建立
            self.save(subject_base, candidate)
            return dict(updated=True, reason="init", stats=candidate)

        # 第1层：跨天融合（today vs yesterday）
        day = {}
        for k in ["med_tb","mad_tb","med_ta","mad_ta"]:
            day[k] = (1-a_day)*cur[k] + a_day*candidate[k]

        # 第2层：EMA低速更新
        fused = {}
        for k in ["med_tb","mad_tb","med_ta","mad_ta"]:
            fused[k] = (1-a_ema)*cur[k] + a_ema*day[k]

        # 硬边界：中心值±20%，MAD不低于历史30%与地板
        fused["med_tb"] = self._clamp(cur["med_tb"], fused["med_tb"], DEFAULTS["max_drift"])
        fused["med_ta"] = self._clamp(cur["med_ta"], fused["med_ta"], DEFAULTS["max_drift"])
        fused["mad_tb"] = max(DEFAULTS["mad_floor"], max(DEFAULTS["min_mad_frac"]*cur["mad_tb"], fused["mad_tb"]))
        fused["mad_ta"] = max(DEFAULTS["mad_floor"], max(DEFAULTS["min_mad_frac"]*cur["mad_ta"], fused["mad_ta"]))
        fused["meta"] = dict(src="gated_update")

        self.save(subject_base, fused)
        return dict(updated=True, reason="fused", stats=fused)
