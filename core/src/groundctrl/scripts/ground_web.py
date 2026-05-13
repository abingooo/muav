#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import xmlrpc.client
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parent.parent
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "groundctrl.yaml"
GAME_PROCESS_PATTERN = (
    r"[r]oslaunch adv adv.launch|[a]dv_ros_adapter|"
    r"[r]oslaunch mpc mpc.launch|[m]pc_ros_adapter"
)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MUAV Ground Web</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --line: #d8dee6;
      --text: #172033;
      --muted: #64748b;
      --ok: #15803d;
      --warn: #b45309;
      --bad: #b91c1c;
      --blue: #1d4ed8;
      --blue-dark: #1e40af;
      --gray: #475569;
      --gray-dark: #334155;
      --red: #dc2626;
      --red-dark: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    header {
      background: #111827;
      color: #ffffff;
      padding: 14px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    header h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    header .sub {
      color: #cbd5e1;
      font-size: 13px;
      margin-top: 2px;
    }
    main {
      max-width: 1360px;
      margin: 0 auto;
      padding: 18px 20px 24px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 14px;
      overflow: hidden;
    }
    .section-title {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background: #fbfcfd;
    }
    .section-title h2 {
      margin: 0;
      font-size: 15px;
      font-weight: 650;
    }
    .section-body {
      padding: 14px;
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    label {
      color: var(--muted);
      font-size: 13px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    select, input {
      height: 34px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 0 10px;
      background: #ffffff;
      color: var(--text);
      font: inherit;
    }
    input[type="number"] {
      width: 76px;
    }
    input[type="checkbox"] {
      height: auto;
      width: auto;
    }
    button {
      min-height: 34px;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 7px 12px;
      color: #ffffff;
      background: var(--blue);
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: var(--blue-dark); }
    button.secondary { background: var(--gray); }
    button.secondary:hover { background: var(--gray-dark); }
    button.warning { background: #b45309; }
    button.warning:hover { background: #92400e; }
    button.danger { background: var(--red); }
    button.danger:hover { background: var(--red-dark); }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1.4fr;
      gap: 14px;
    }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: auto;
    }
    th, td {
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 8px;
      text-align: left;
      white-space: nowrap;
      vertical-align: middle;
    }
    th {
      color: #475569;
      font-size: 12px;
      font-weight: 700;
      background: #f8fafc;
    }
    tr:last-child td { border-bottom: 0; }
    .num {
      font-variant-numeric: tabular-nums;
      text-align: right;
    }
    .badge {
      display: inline-block;
      min-width: 58px;
      text-align: center;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid transparent;
    }
    .ok {
      color: var(--ok);
      background: #dcfce7;
      border-color: #bbf7d0;
    }
    .warn {
      color: var(--warn);
      background: #fef3c7;
      border-color: #fde68a;
    }
    .bad {
      color: var(--bad);
      background: #fee2e2;
      border-color: #fecaca;
    }
    .idle {
      color: #475569;
      background: #e2e8f0;
      border-color: #cbd5e1;
    }
    .muted {
      color: var(--muted);
    }
    .hidden {
      display: none !important;
    }
    .mode-chip {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 9px;
      background: #e0f2fe;
      color: #075985;
      font-size: 12px;
      font-weight: 800;
      border: 1px solid #bae6fd;
      white-space: nowrap;
    }
    .safety {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border: 1px solid #fbbf24;
      background: #fffbeb;
      border-radius: 6px;
      color: #78350f;
      font-weight: 650;
    }
    .actions {
      display: flex;
      align-items: center;
      gap: 9px;
      flex-wrap: wrap;
    }
    .job-list {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .job-pill {
      background: #e2e8f0;
      color: #0f172a;
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      padding: 5px 9px;
      cursor: pointer;
      font-size: 12px;
    }
    .job-pill.active {
      background: #dbeafe;
      border-color: #93c5fd;
      color: #1e3a8a;
    }
    details.log-panel summary {
      cursor: pointer;
      color: var(--gray-dark);
      font-weight: 650;
      margin-bottom: 10px;
      user-select: none;
    }
    pre {
      margin: 0;
      min-height: 220px;
      max-height: 420px;
      overflow: auto;
      padding: 12px;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .note {
      color: var(--muted);
      line-height: 1.5;
      margin: 8px 0 0;
    }
    .stage-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 10px;
    }
    .stage {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #f8fafc;
    }
    .stage-name {
      font-weight: 700;
      margin-bottom: 6px;
    }
    .checklist {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .check-item {
      display: flex;
      justify-content: flex-start;
      align-items: center;
      gap: 10px;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 8px 10px;
      background: #ffffff;
      flex-wrap: wrap;
    }
    .compact-uav {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      margin-right: 10px;
      font-weight: 650;
    }
    .mini {
      display: inline-block;
      min-width: 26px;
      text-align: center;
      border-radius: 999px;
      padding: 2px 6px;
      font-size: 11px;
      font-weight: 800;
      border: 1px solid transparent;
    }
    .flight-state-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .flight-card {
      min-width: 0;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 9px 10px;
      background: #f8fafc;
    }
    .flight-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .flight-name {
      font-weight: 750;
      font-variant-numeric: tabular-nums;
    }
    .state-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 52px;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 800;
      border: 1px solid transparent;
      white-space: nowrap;
    }
    .state-ready { color: #475569; background: #e2e8f0; border-color: #cbd5e1; }
    .state-launch { color: #1d4ed8; background: #dbeafe; border-color: #93c5fd; }
    .state-hover { color: #15803d; background: #dcfce7; border-color: #bbf7d0; }
    .state-game { color: #7c2d12; background: #ffedd5; border-color: #fed7aa; }
    .state-land { color: #b45309; background: #fef3c7; border-color: #fde68a; }
    .flight-meta {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      min-height: 18px;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .game-result-box {
      display: flex;
      align-items: center;
      gap: 7px;
      flex-wrap: wrap;
      margin-top: 10px;
      min-height: 36px;
      border: 1px solid #d8dee6;
      border-radius: 6px;
      padding: 8px 10px;
      background: #ffffff;
    }
    .result-label {
      font-weight: 750;
    }
    .result-chip {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      background: #eef2ff;
      color: #3730a3;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .result-positions {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    @media (max-width: 720px) {
      .stage-grid { grid-template-columns: 1fr; }
      .flight-state-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>MUAV Ground Web</h1>
      <div class="sub" id="masterLine">加载中</div>
    </div>
    <div class="toolbar">
      <label>目标
        <select id="uavTarget" onchange="renderPreflight(latestState); updateControls();">
          <option value="all">all</option>
          <option value="uav0">uav0</option>
          <option value="uav1">uav1</option>
          <option value="uav2">uav2</option>
        </select>
      </label>
      <label>降落超时
        <input id="landTimeout" type="number" min="1" step="1" value="8">
      </label>
      <label><input id="startRviz" type="checkbox"> prep 启动 RViz</label>
      <button class="secondary" onclick="refreshState()">刷新状态</button>
    </div>
  </header>

  <main>
    <section>
      <div class="section-title">
        <h2>本机控制</h2>
        <span id="localSummary" class="muted">等待状态</span>
      </div>
      <div class="section-body actions">
        <button id="btnRoscore" onclick="runAction('start_roscore')">启动本地 roscore</button>
        <button id="btnStopRoscore" class="secondary" onclick="confirmAction('stop_local_roscore', '确认停止网页启动的本地 roscore？')">停止本地 roscore</button>
        <button id="btnPrep" onclick="runAction('prep')">Prep</button>
        <label class="safety"><input id="airspaceSafe" type="checkbox" onchange="updateControls()">确认空域安全</label>
        <button id="btnTakeoff" onclick="requestTakeoff()">起飞</button>
        <button id="btnLand" class="warning" onclick="confirmAction('land', `确认降落 ${currentTarget()}？`)">降落</button>
        <button id="btnStop" class="danger" onclick="confirmAction('stop', `确认停止 ${currentTarget()} 并停止 groundsync？`)">一键停止</button>
      </div>
    </section>

    <section>
      <div class="section-title">
        <h2>Prep 阶段</h2>
        <span class="muted" id="prepSummary">尚未执行</span>
      </div>
      <div class="section-body">
        <div id="prepStages" class="stage-grid"></div>
      </div>
    </section>

    <section>
      <div class="section-title">
        <h2>起飞前检查</h2>
        <span class="muted" id="preflightSummary">等待状态</span>
      </div>
      <div class="section-body">
        <div id="preflightList" class="checklist"></div>
        <p class="note">起飞按钮要求先勾选“确认空域安全”。检查未全绿时仍可二次确认强制起飞，但页面会明确提示风险项。</p>
      </div>
    </section>

    <section>
      <div class="section-title">
        <h2>博弈实验</h2>
        <span class="muted" id="gameSummary">defender_0=uav0 defender_1=uav1 enemy=uav2</span>
      </div>
      <div class="section-body actions">
        <span id="gameModeBadge" class="mode-chip">完整三机博弈</span>
        <div id="fullGameControls" class="actions">
          <label>defender_0
            <select id="roleDefender0" onchange="updateGameSummary(); updateControls();">
              <option value="uav0" selected>uav0</option>
              <option value="uav1">uav1</option>
              <option value="uav2">uav2</option>
            </select>
          </label>
          <label>defender_1
            <select id="roleDefender1" onchange="updateGameSummary(); updateControls();">
              <option value="uav0">uav0</option>
              <option value="uav1" selected>uav1</option>
              <option value="uav2">uav2</option>
            </select>
          </label>
          <label>enemy
            <select id="roleEnemy" onchange="updateGameSummary(); updateControls();">
              <option value="uav0">uav0</option>
              <option value="uav1">uav1</option>
              <option value="uav2" selected>uav2</option>
            </select>
          </label>
        </div>
        <div id="singleGameControls" class="actions hidden">
          <span id="singleGameTarget" class="muted">测试 UAV: uav0</span>
          <label>测试角色
            <select id="singleGameRole" onchange="updateGameSummary(); updateControls();">
              <option value="defender_0" selected>defender_0 / ADV</option>
              <option value="defender_1">defender_1 / ADV</option>
              <option value="enemy">enemy / MPC</option>
            </select>
          </label>
        </div>
        <button id="btnGameStart" onclick="requestGameStart()">启动博弈</button>
        <button id="btnGameStop" class="danger" onclick="requestGameStop()">停止博弈</button>
        <span id="gameHint" class="muted">启动后 defender 跑 ADV，enemy 跑 MPC。</span>
      </div>
    </section>

    <div class="grid">
      <section>
        <div class="section-title">
          <h2>连接状态</h2>
          <span class="muted" id="statusTime">-</span>
        </div>
        <div class="section-body">
          <table>
            <thead>
              <tr>
                <th>项目</th>
                <th>状态</th>
                <th>说明</th>
              </tr>
            </thead>
            <tbody id="linkRows"></tbody>
          </table>
        </div>
      </section>

      <section>
        <div class="section-title">
          <h2>里程计 Watch</h2>
          <span class="muted">OK &lt; 1s, LOST &gt; 3s</span>
        </div>
        <div class="section-body">
          <table>
            <thead>
              <tr>
                <th>UAV</th>
                <th>状态</th>
                <th class="num">AGE</th>
                <th class="num">HZ</th>
                <th class="num">X</th>
                <th class="num">Y</th>
                <th class="num">Z</th>
                <th class="num">VX</th>
                <th class="num">VY</th>
                <th class="num">VZ</th>
              </tr>
            </thead>
            <tbody id="odomRows"></tbody>
          </table>
          <div id="flightStateRows" class="flight-state-grid"></div>
          <div id="gameResultBox" class="game-result-box muted">暂无博弈结束结果</div>
          <p class="note">页面实时读取 ground 本机 ROS master 上同步到的 odom。若显示 LOST，先确认 groundsync 和 UAV 侧 vins/odom 是否正常。</p>
        </div>
      </section>
    </div>

    <section>
      <div class="section-title">
        <h2>任务日志</h2>
        <span class="muted" id="jobSummary">无任务</span>
      </div>
      <div class="section-body">
        <div id="jobList" class="job-list"></div>
        <details id="jobLogPanel" class="log-panel">
          <summary>展开任务日志</summary>
          <pre id="jobLog">暂无日志</pre>
        </details>
      </div>
    </section>
  </main>

  <script>
    let selectedJob = null;
    let latestState = null;
    let latestJobs = [];
    let odomInFlight = false;
    let tickInFlight = false;

    function badge(state) {
      const s = String(state || 'UNKNOWN');
      let cls = 'bad';
      if (s === 'OK' || s === 'RUNNING' || s === 'REACHABLE') cls = 'ok';
      if (s === 'STALE' || s === 'PARTIAL' || s === 'UNKNOWN') cls = 'warn';
      if (s === 'PENDING' || s === 'SKIPPED' || s === 'IDLE') cls = 'idle';
      return `<span class="badge ${cls}">${s}</span>`;
    }

    function mini(label, ok, title) {
      const cls = ok ? 'ok' : 'bad';
      return `<span class="mini ${cls}" title="${title || ''}">${label}</span>`;
    }

    function fmt(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
      return Number(value).toFixed(digits);
    }

    function currentTarget() {
      return document.getElementById('uavTarget').value;
    }

    function targetNames() {
      const target = currentTarget();
      if (target === 'all') return ['uav0', 'uav1', 'uav2'];
      return target.split(',').map(item => item.trim()).filter(Boolean);
    }

    function isFullGameMode() {
      return currentTarget() === 'all';
    }

    function singleGameRole() {
      return document.getElementById('singleGameRole').value;
    }

    function gameKindForRole(role) {
      return role === 'enemy' ? 'MPC' : 'ADV';
    }

    function allUavNames(data) {
      const names = (data && data.odom ? data.odom : []).map(row => row.name).filter(Boolean);
      return names.length ? names : ['uav0', 'uav1', 'uav2'];
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      }[ch]));
    }

    function jobAppliesToUav(job, name) {
      const text = job && job.name ? job.name : '';
      return text.includes(' all') || text.includes(name);
    }

    function runningJob(prefix, name) {
      return latestJobs.some(job => job.state === 'RUNNING' && (job.name || '').startsWith(prefix) && jobAppliesToUav(job, name));
    }

    function recentJob(prefix, name, seconds) {
      const now = Date.now() / 1000;
      return latestJobs.some(job => {
        if (!(job.name || '').startsWith(prefix) || !jobAppliesToUav(job, name)) return false;
        if (!job.ended_at || job.state === 'RUNNING') return false;
        return now - Number(job.ended_at) <= seconds;
      });
    }

    function latestGameRoleByUav() {
      const fallback = gameAssignment();
      const roleByUav = {};
      if (isFullGameMode()) {
        roleByUav[fallback.defender0] = 'defender_0';
        roleByUav[fallback.defender1] = 'defender_1';
        roleByUav[fallback.enemy] = 'enemy';
      } else {
        roleByUav[currentTarget()] = singleGameRole();
      }

      const gameJob = latestJobs.find(job => (job.name || '').startsWith('game start '));
      const match = gameJob && (gameJob.name || '').match(/d0=(uav\d+)\s+d1=(uav\d+)\s+enemy=(uav\d+)/);
      const singleJob = latestJobs.find(job => (job.name || '').startsWith('game single '));
      const singleMatch = singleJob && (singleJob.name || '').match(/game single (uav\d+) role=(defender_0|defender_1|enemy)/);
      if (singleMatch) {
        roleByUav[singleMatch[1]] = singleMatch[2];
      }
      if (!match) return roleByUav;
      return {
        ...roleByUav,
        [match[1]]: 'defender_0',
        [match[2]]: 'defender_1',
        [match[3]]: 'enemy'
      };
    }

    function inferFlightState(name, odom, data) {
      const z = Number(odom && odom.z);
      const odomOk = odom && odom.state === 'OK';
      const zText = Number.isFinite(z) ? `z=${fmt(z, 2)}m` : `odom ${odom && odom.state ? odom.state : 'LOST'}`;
      const gameRunning = data && data.game_running && data.game_running[name];

      if (gameRunning || runningJob('game start ', name) || runningJob('game single ', name) || recentJob('game start ', name, 8) || recentJob('game single ', name, 8)) {
        return { label: '博弈', kind: 'game', detail: 'ADV/MPC 运行中' };
      }
      if (runningJob('land ', name) || runningJob('stop ', name)) {
        return { label: '降落', kind: 'land', detail: zText };
      }
      if ((recentJob('land ', name, 20) || recentJob('stop ', name, 20)) && odomOk && Number.isFinite(z) && z >= 0.25) {
        return { label: '降落', kind: 'land', detail: zText };
      }
      if (runningJob('takeoff ', name) || (recentJob('takeoff ', name, 12) && (!odomOk || !Number.isFinite(z) || z < 0.45))) {
        return { label: '起飞', kind: 'launch', detail: zText };
      }
      if (odomOk && Number.isFinite(z) && z >= 0.35) {
        return { label: '悬停', kind: 'hover', detail: `${zText} hz=${fmt(odom.hz, 1)}` };
      }
      return { label: '待飞', kind: 'ready', detail: zText };
    }

    function renderFlightStates(data) {
      const container = document.getElementById('flightStateRows');
      if (!container) return;
      const odomByName = {};
      for (const item of (data && data.odom ? data.odom : [])) odomByName[item.name] = item;
      const roleByUav = latestGameRoleByUav();
      container.innerHTML = allUavNames(data).map(name => {
        const state = inferFlightState(name, odomByName[name] || {}, data || {});
        const role = state.kind === 'game' && roleByUav[name] ? `${roleByUav[name]} · ` : '';
        return `<div class="flight-card">
          <div class="flight-top">
            <span class="flight-name">${escapeHtml(name)}</span>
            <span class="state-pill state-${state.kind}">${state.label}</span>
          </div>
          <div class="flight-meta">${escapeHtml(role + state.detail)}</div>
        </div>`;
      }).join('');
    }

    function gameOutcomeText(outcome) {
      const labels = {
        defender_win: '防守方胜',
        enemy_win: '敌方胜',
        draw: '平局',
        timeout: '超时',
        unknown: '未知'
      };
      if (!outcome) return '未知结果';
      return labels[outcome] ? `${labels[outcome]} (${outcome})` : outcome;
    }

    function shortRole(role) {
      return { defender_0: 'd0', defender_1: 'd1', enemy: 'enemy' }[role] || role;
    }

    function renderGameResult(result) {
      const box = document.getElementById('gameResultBox');
      if (!box) return;
      if (!result) {
        box.classList.add('muted');
        box.innerHTML = '暂无博弈结束结果';
        return;
      }

      box.classList.remove('muted');
      const chips = [];
      chips.push(`<span class="result-chip">${escapeHtml(gameOutcomeText(result.outcome))}</span>`);
      if (result.node_role) chips.push(`<span class="result-chip">role=${escapeHtml(result.node_role)}</span>`);
      if (result.runtime !== null && result.runtime !== undefined) chips.push(`<span class="result-chip">${fmt(result.runtime, 2)}s</span>`);
      if (result.step_count !== null && result.step_count !== undefined) chips.push(`<span class="result-chip">${escapeHtml(result.step_count)} steps</span>`);
      if (result.hover_roles && result.hover_roles.length) {
        chips.push(`<span class="result-chip">hover=${escapeHtml(result.hover_roles.join(','))}</span>`);
      }

      let positions = '';
      if (result.positions) {
        positions = Object.keys(result.positions).map(role => {
          const values = result.positions[role] || [];
          return `${shortRole(role)}=(${values.map(v => fmt(v, 1)).join(',')})`;
        }).join(' ');
      }
      const source = result.source_uav ? `来自 ${result.source_uav}` : '';
      box.innerHTML = `<span class="result-label">最新博弈</span>${chips.join('')}<span class="result-positions">${escapeHtml([positions, source].filter(Boolean).join(' · '))}</span>`;
    }

    function targetPayload() {
      return {
        uav: currentTarget(),
        start_rviz: document.getElementById('startRviz').checked,
        land_timeout: Number(document.getElementById('landTimeout').value || 8),
        defender0: document.getElementById('roleDefender0').value,
        defender1: document.getElementById('roleDefender1').value,
        enemy: document.getElementById('roleEnemy').value,
        game_role: singleGameRole()
      };
    }

    async function runAction(action) {
      const payload = targetPayload();
      payload.action = action;
      const res = await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!data.ok) {
        alert(data.error || '操作失败');
        return;
      }
      if (data.job_id) selectedJob = data.job_id;
      await refreshState();
      await refreshJobs();
    }

    function confirmAction(action, text) {
      if (window.confirm(text)) runAction(action);
    }

    function gameAssignment() {
      return {
        defender0: document.getElementById('roleDefender0').value,
        defender1: document.getElementById('roleDefender1').value,
        enemy: document.getElementById('roleEnemy').value
      };
    }

    function gameAssignmentError() {
      if (!isFullGameMode()) {
        const role = singleGameRole();
        if (!['defender_0', 'defender_1', 'enemy'].includes(role)) return '单机测试角色无效';
        return '';
      }
      const role = gameAssignment();
      const unique = new Set([role.defender0, role.defender1, role.enemy]);
      if (unique.size !== 3) return '三个角色必须分配给三架不同 UAV';
      return '';
    }

    function updateGameSummary() {
      const fullMode = isFullGameMode();
      document.getElementById('fullGameControls').classList.toggle('hidden', !fullMode);
      document.getElementById('singleGameControls').classList.toggle('hidden', fullMode);
      document.getElementById('gameModeBadge').textContent = fullMode ? '完整三机博弈' : '单机博弈测试';

      if (!fullMode) {
        const target = currentTarget();
        const role = singleGameRole();
        document.getElementById('singleGameTarget').textContent = `测试 UAV: ${target}`;
        document.getElementById('gameSummary').innerHTML = `${target} as ${role}，启动 ${gameKindForRole(role)}`;
        document.getElementById('gameHint').textContent = '单机测试只启动当前目标 UAV 的一个 ADV/MPC 进程，用于验证单机输入输出和跟随。';
        return;
      }

      const role = gameAssignment();
      const err = gameAssignmentError();
      const fleet = `${role.defender0},${role.defender1},${role.enemy}`;
      document.getElementById('gameSummary').innerHTML = err
        ? `${badge('LOST')} ${err}`
        : `defender_0=${role.defender0} defender_1=${role.defender1} enemy=${role.enemy} fleet=${fleet}`;
      document.getElementById('gameHint').textContent = '启动后 defender 跑 ADV，enemy 跑 MPC。';
    }

    function requestGameStart() {
      const err = gameAssignmentError();
      if (err) {
        alert(err);
        return;
      }
      if (!isFullGameMode()) {
        const target = currentTarget();
        const role = singleGameRole();
        const kind = gameKindForRole(role);
        const text = `确认启动单机博弈测试？\n\n${target} 作为 ${role}，运行 ${kind}`;
        confirmAction('game_start', text);
        return;
      }
      const role = gameAssignment();
      const text = `确认启动博弈？\n\ndefender_0=${role.defender0} 运行 ADV\ndefender_1=${role.defender1} 运行 ADV\nenemy=${role.enemy} 运行 MPC`;
      confirmAction('game_start', text);
    }

    function requestGameStop() {
      const target = currentTarget();
      const text = target === 'all'
        ? '确认停止三机 ADV/MPC 博弈进程？'
        : `确认停止 ${target} 的单机 ADV/MPC 测试进程？`;
      confirmAction('game_stop', text);
    }

    function preflightItems(data) {
      if (!data) return [{ label: '状态加载', ok: false, detail: '等待 /api/state' }];
      const names = targetNames();
      const uavByName = {};
      const odomByName = {};
      for (const item of data.uavs || []) uavByName[item.name] = item;
      for (const item of data.odom || []) odomByName[item.name] = item;

      const items = [
        {
          label: '本地 roscore',
          ok: !!data.local_roscore,
          detail: data.master_uri || ''
        },
        {
          label: 'groundsync',
          ok: !!data.groundsync,
          detail: '用于同步三机里程计'
        }
      ];

      for (const name of names) {
        const link = uavByName[name] || {};
        const odom = odomByName[name] || {};
        items.push({
          label: `${name} SSH`,
          ok: !!link.ssh,
          detail: link.ssh_host || ''
        });
        items.push({
          label: `${name} ROS master`,
          ok: !!link.ros_master,
          detail: link.ros_master_uri || ''
        });
        items.push({
          label: `${name} odom`,
          ok: odom.state === 'OK',
          detail: `${odom.state || 'LOST'} age=${fmt(odom.age, 1)}s hz=${fmt(odom.hz, 1)}`
        });
      }
      return items;
    }

    function renderPreflight(data) {
      const items = preflightItems(data);
      const failed = items.filter(item => !item.ok);
      document.getElementById('preflightSummary').innerHTML =
        failed.length === 0 ? `${badge('OK')} 起飞前检查通过` : `${badge('PARTIAL')} ${failed.length} 项未通过`;
      if (!data) {
        document.getElementById('preflightList').innerHTML =
          `<div class="check-item"><span>状态加载</span>${badge('LOST')}<span class="muted">等待 /api/state</span></div>`;
        return failed;
      }

      const names = targetNames();
      const uavByName = {};
      const odomByName = {};
      for (const item of data.uavs || []) uavByName[item.name] = item;
      for (const item of data.odom || []) odomByName[item.name] = item;

      const line1 = `<div class="check-item">
        <strong>基础</strong>
        <span>roscore ${badge(data.local_roscore ? 'OK' : 'LOST')}</span>
        <span>groundsync ${badge(data.groundsync ? 'OK' : 'LOST')}</span>
        <span class="muted">目标 ${currentTarget()}</span>
      </div>`;

      const line2 = `<div class="check-item">
        <strong>UAV</strong>
        ${names.map(name => {
          const link = uavByName[name] || {};
          const odom = odomByName[name] || {};
          return `<span class="compact-uav">
            ${name}
            ${mini('S', !!link.ssh, 'SSH')}
            ${mini('R', !!link.ros_master, 'ROS master')}
            ${mini('O', odom.state === 'OK', `ODOM ${odom.state || 'LOST'} age=${fmt(odom.age, 1)}s hz=${fmt(odom.hz, 1)}`)}
          </span>`;
        }).join('')}
      </div>`;
      document.getElementById('preflightList').innerHTML = line1 + line2;
      return failed;
    }

    function renderPrepStages(jobs) {
      const prep = jobs.find(job => job.name && job.name.startsWith('prep '));
      const container = document.getElementById('prepStages');
      if (!prep || !prep.stages) {
        document.getElementById('prepSummary').textContent = '尚未执行';
        container.innerHTML = ['roscore', 'core', 'vins-sync', 'groundsync'].map(name => (
          `<div class="stage"><div class="stage-name">${name}</div>${badge('PENDING')}<div class="muted">等待 Prep</div></div>`
        )).join('');
        return;
      }

      document.getElementById('prepSummary').innerHTML = `${prep.name} ${badge(prep.state)}`;
      container.innerHTML = prep.stages.map(stage => (
        `<div class="stage">
          <div class="stage-name">${stage.label}</div>
          ${badge(stage.state)}
          <div class="muted">${stage.detail || ''}</div>
        </div>`
      )).join('');
    }

    function runningJobs() {
      return latestJobs.filter(job => job.state === 'RUNNING');
    }

    function hasRunningStop() {
      return runningJobs().some(job => job.name && job.name.startsWith('stop '));
    }

    function hasBlockingPrep() {
      return runningJobs().some(job => job.name && job.name.startsWith('prep ') && job.blocking);
    }

    function hasBlockingControl() {
      return runningJobs().some(job => job.blocking && !(job.name || '').startsWith('prep '));
    }

    function updateControls() {
      const safe = document.getElementById('airspaceSafe').checked;
      const stopRunning = hasRunningStop();
      const prepBlocking = hasBlockingPrep();
      const controlBlocking = hasBlockingControl();
      const anyBlocking = stopRunning || prepBlocking || controlBlocking;

      document.getElementById('btnRoscore').disabled = stopRunning;
      document.getElementById('btnStopRoscore').disabled = stopRunning || anyBlocking;
      document.getElementById('btnPrep').disabled = stopRunning || prepBlocking || controlBlocking;
      document.getElementById('btnTakeoff').disabled = stopRunning || prepBlocking || controlBlocking || !safe;
      document.getElementById('btnLand').disabled = stopRunning || prepBlocking || controlBlocking;
      document.getElementById('btnStop').disabled = stopRunning;
      const gameInvalid = !!gameAssignmentError();
      document.getElementById('btnGameStart').disabled = stopRunning || prepBlocking || controlBlocking || gameInvalid;
      document.getElementById('btnGameStop').disabled = stopRunning;

      const target = currentTarget();
      document.getElementById('btnTakeoff').textContent = target === 'all' ? '起飞 all（三机）' : `起飞 ${target}`;
      document.getElementById('btnLand').textContent = target === 'all' ? '降落 all（三机）' : `降落 ${target}`;
      document.getElementById('btnStop').textContent = target === 'all' ? '一键停止 all' : `一键停止 ${target}`;
      document.getElementById('btnGameStart').textContent = target === 'all' ? '启动三机博弈' : `启动 ${target} 单机测试`;
      document.getElementById('btnGameStop').textContent = target === 'all' ? '停止博弈' : `停止 ${target} 测试`;
      updateGameSummary();
    }

    function requestTakeoff() {
      const safe = document.getElementById('airspaceSafe').checked;
      if (!safe) {
        alert('请先勾选“确认空域安全”。');
        return;
      }
      const failed = renderPreflight(latestState);
      const target = currentTarget();
      let text = `确认起飞 ${target}？`;
      if (target === 'all') text = '确认三架无人机并行起飞？';
      if (failed.length > 0) {
        const details = failed.map(item => `- ${item.label}: ${item.detail}`).join('\n');
        if (!window.confirm(`起飞前检查未全绿：\n${details}\n\n仍然继续起飞 ${target}？`)) return;
      }
      confirmAction('takeoff', text);
    }

    function renderOdom(data) {
      const odomRows = [];
      for (const row of data.odom || []) {
        odomRows.push(
          `<tr>
            <td>${row.name}</td>
            <td>${badge(row.state)}</td>
            <td class="num">${fmt(row.age, 1)}</td>
            <td class="num">${fmt(row.hz, 1)}</td>
            <td class="num">${fmt(row.x)}</td>
            <td class="num">${fmt(row.y)}</td>
            <td class="num">${fmt(row.z)}</td>
            <td class="num">${fmt(row.vx)}</td>
            <td class="num">${fmt(row.vy)}</td>
            <td class="num">${fmt(row.vz)}</td>
          </tr>`
        );
      }
      document.getElementById('odomRows').innerHTML = odomRows.join('');
      renderFlightStates(data);
      renderGameResult(data.game_result);
    }

    async function refreshOdom() {
      if (odomInFlight) return;
      odomInFlight = true;
      try {
        const res = await fetch('/api/odom');
        if (!res.ok) return;
        const data = await res.json();
        latestState = { ...(latestState || {}), ...data };
        renderOdom(latestState);
      } catch (_err) {
        // Keep the last displayed odom if one refresh fails.
      } finally {
        odomInFlight = false;
      }
    }

    async function refreshState() {
      const res = await fetch('/api/state');
      const data = await res.json();
      latestState = data;
      document.getElementById('masterLine').textContent = `ROS_MASTER_URI=${data.master_uri || ''}`;
      document.getElementById('statusTime').textContent = data.time || '-';
      document.getElementById('localSummary').innerHTML =
        `本地 roscore ${badge(data.local_roscore ? 'REACHABLE' : 'LOST')} groundsync ${badge(data.groundsync ? 'RUNNING' : 'LOST')}`;

      const linkRows = [];
      linkRows.push(`<tr><td>local roscore</td><td>${badge(data.local_roscore ? 'REACHABLE' : 'LOST')}</td><td>${data.master_uri || ''}</td></tr>`);
      linkRows.push(`<tr><td>groundsync</td><td>${badge(data.groundsync ? 'RUNNING' : 'LOST')}</td><td>ground_master_sync / ground_sync.launch</td></tr>`);
      for (const item of data.uavs || []) {
        const sshState = item.ssh ? 'REACHABLE' : 'LOST';
        const rosState = item.ros_master ? 'REACHABLE' : 'LOST';
        linkRows.push(`<tr><td>${item.name} SSH</td><td>${badge(sshState)}</td><td>${item.ssh_host || ''}</td></tr>`);
        linkRows.push(`<tr><td>${item.name} ROS</td><td>${badge(rosState)}</td><td>${item.ros_master_uri || ''}</td></tr>`);
      }
      document.getElementById('linkRows').innerHTML = linkRows.join('');

      renderOdom(data);
      renderPreflight(data);
      updateControls();
    }

    async function refreshJobs() {
      const res = await fetch('/api/jobs');
      const data = await res.json();
      const jobs = data.jobs || [];
      latestJobs = jobs;
      if (!selectedJob && jobs.length) selectedJob = jobs[0].id;
      document.getElementById('jobSummary').textContent = jobs.length ? `${jobs.length} 个任务` : '无任务';
      document.getElementById('jobList').innerHTML = jobs.map(job => {
        const active = job.id === selectedJob ? ' active' : '';
        return `<button class="job-pill${active}" onclick="selectJob('${job.id}')">${job.name} ${job.state}</button>`;
      }).join('');
      if (selectedJob) {
        const logRes = await fetch(`/api/job?id=${encodeURIComponent(selectedJob)}`);
        const logData = await logRes.json();
        document.getElementById('jobLog').textContent = logData.log || '暂无日志';
        const pre = document.getElementById('jobLog');
        pre.scrollTop = pre.scrollHeight;
      }
      renderPrepStages(jobs);
      if (latestState) renderFlightStates(latestState);
      updateControls();
    }

    function selectJob(id) {
      selectedJob = id;
      refreshJobs();
    }

    async function tick() {
      if (tickInFlight) return;
      tickInFlight = true;
      try {
        await refreshState();
        await refreshJobs();
      } finally {
        tickInFlight = false;
      }
    }

    tick();
    setInterval(refreshOdom, 100);
    setInterval(tick, 1500);
  </script>
</body>
</html>
"""


def load_config(path):
    path = Path(path or DEFAULT_CONFIG).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ordered_uavs(config):
    return sorted(config["uavs"], key=lambda name: int(config["uavs"][name]["id"]))


def topic(config, topic_name, uav_name):
    uav_conf = config["uavs"][uav_name]
    return config["topics"][topic_name].format(uav=uav_name, id=uav_conf["id"])


def master_reachable(uri, timeout_s=1.0):
    if not uri:
        return False
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_s)
    try:
        proxy = xmlrpc.client.ServerProxy(uri)
        code, _message, _pid = proxy.getPid("/ground_web")
        return int(code) == 1
    except Exception:
        return False
    finally:
        socket.setdefaulttimeout(old_timeout)


def ssh_reachable(ssh_host, timeout_s=3.0):
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=2",
        ssh_host,
        "true",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def process_running(pattern):
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class OdomSlot:
    def __init__(self, name, topic_name):
        self.name = name
        self.topic_name = topic_name
        self.last_msg = None
        self.last_wall = None
        self.samples = deque(maxlen=200)

    def update(self, msg):
        now = time.monotonic()
        self.last_msg = msg
        self.last_wall = now
        self.samples.append(now)

    def snapshot(self, now, warn_age, lost_age, hz_window):
        age = None if self.last_wall is None else now - self.last_wall
        state = "LOST"
        if age is not None:
            if age >= lost_age:
                state = "LOST"
            elif age >= warn_age:
                state = "STALE"
            else:
                state = "OK"

        cutoff = now - hz_window
        while self.samples and self.samples[0] < cutoff:
            self.samples.popleft()
        hz = 0.0
        if len(self.samples) >= 2:
            span = self.samples[-1] - self.samples[0]
            if span > 0.0:
                hz = (len(self.samples) - 1) / span

        values = {
            "x": None,
            "y": None,
            "z": None,
            "vx": None,
            "vy": None,
            "vz": None,
        }
        if self.last_msg is not None:
            pose = self.last_msg.pose.pose
            twist = self.last_msg.twist.twist
            values = {
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z,
                "vx": twist.linear.x,
                "vy": twist.linear.y,
                "vz": twist.linear.z,
            }

        return {
            "name": self.name,
            "state": state,
            "age": age,
            "hz": hz,
            "topic": self.topic_name,
            **values,
        }


class OdomMonitor:
    def __init__(self, config, warn_age=1.0, lost_age=3.0, hz_window=3.0):
        self.config = config
        self.warn_age = warn_age
        self.lost_age = lost_age
        self.hz_window = hz_window
        self.lock = threading.Lock()
        self.started = False
        self.error = ""
        self.last_start_attempt = 0.0
        self.slots = {
            name: OdomSlot(name, topic(config, "odom", name))
            for name in ordered_uavs(config)
        }
        self.subscribers = []

    def ensure_started(self):
        if self.started:
            return
        now = time.monotonic()
        if now - self.last_start_attempt < 2.0:
            return
        self.last_start_attempt = now
        try:
            import rospy
            from nav_msgs.msg import Odometry

            rospy.init_node("ground_web_odom_monitor", anonymous=True, disable_signals=True)
            for slot in self.slots.values():
                self.subscribers.append(
                    rospy.Subscriber(
                        slot.topic_name,
                        Odometry,
                        self._make_cb(slot),
                        queue_size=20,
                    )
                )
            self.started = True
            self.error = ""
        except Exception as exc:
            self.error = str(exc)

    def _make_cb(self, slot):
        def callback(msg):
            with self.lock:
                slot.update(msg)

        return callback

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            rows = [
                slot.snapshot(now, self.warn_age, self.lost_age, self.hz_window)
                for slot in self.slots.values()
            ]
        return rows


class Job:
    def __init__(self, job_id, name, cmd, proc, blocking=True):
        self.id = job_id
        self.name = name
        self.cmd = cmd
        self.proc = proc
        self.blocking = blocking
        self.started_at = time.time()
        self.ended_at = None
        self.returncode = None
        self.log = deque(maxlen=500)
        self.lock = threading.Lock()
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _append(self, line):
        timestamp = time.strftime("%H:%M:%S")
        with self.lock:
            self.log.append(f"[{timestamp}] {line.rstrip()}")

    def _read_output(self):
        self._append("$ " + " ".join(shlex_quote(part) for part in self.cmd))
        try:
            if self.proc.stdout is not None:
                for line in self.proc.stdout:
                    self._append(line)
            self.returncode = self.proc.wait()
        except Exception as exc:
            self._append(f"job reader error: {exc}")
            self.returncode = self.proc.poll()
        self.ended_at = time.time()
        self._append(f"exit={self.returncode}")

    def state(self):
        if self.returncode is None and self.proc.poll() is None:
            return "RUNNING"
        if self.returncode is None:
            self.returncode = self.proc.returncode
        return "OK" if self.returncode == 0 else "FAILED"

    def to_summary(self):
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state(),
            "returncode": self.returncode,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "blocking": self.state() == "RUNNING" and self.blocking,
            "stages": [],
        }

    def log_text(self):
        with self.lock:
            return "\n".join(self.log)


def shlex_quote(text):
    import shlex

    return shlex.quote(str(text))


def ssh_capture(ssh_host, command, timeout_s=4.0):
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=2",
            ssh_host,
            "bash",
            "-lc",
            shlex_quote(command),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0 and not result.stdout:
        raise RuntimeError((result.stderr or f"ssh exit {result.returncode}").strip())
    return result.stdout


def parse_float_token(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_game_result_text(text):
    outcome_match = re.search(r"\boutcome=([A-Za-z0-9_:-]+)", text)
    role_match = re.search(r"\bnode_role=([A-Za-z0-9_:-]+)", text)
    runtime_match = re.search(r"\bruntime=([0-9.]+)s", text)
    steps_match = re.search(r"\bstep_count=([0-9]+)", text)
    hover_match = re.search(r"Published terminal hover for roles=\[([^\]]*)\]", text)

    positions = {}
    positions_match = re.search(
        r"world_positions=(.*?)(?:\.\s*Published terminal hover|Published terminal hover|$)",
        text,
    )
    if positions_match:
        positions_text = positions_match.group(1)
        for match in re.finditer(r"(defender_0|defender_1|enemy)=\[([^\]]+)\]", positions_text):
            values = [
                parse_float_token(token)
                for token in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", match.group(2))
            ]
            values = [value for value in values if value is not None][:3]
            if values:
                positions[match.group(1)] = values

    hover_roles = []
    if hover_match:
        hover_roles = [
            item.strip().strip("'\"")
            for item in hover_match.group(1).split(",")
            if item.strip()
        ]

    if not any((outcome_match, role_match, runtime_match, steps_match, positions, hover_roles)):
        return None

    runtime = parse_float_token(runtime_match.group(1)) if runtime_match else None
    return {
        "outcome": outcome_match.group(1) if outcome_match else None,
        "node_role": role_match.group(1) if role_match else None,
        "runtime": runtime,
        "step_count": int(steps_match.group(1)) if steps_match else None,
        "hover_roles": hover_roles,
        "positions": positions,
        "raw": text[-500:],
    }


def parse_game_result_records(records, source_uav):
    candidates = []
    for index, record in enumerate(records):
        line = record["line"]
        if not re.search(r"\bgame ended:", line, re.IGNORECASE):
            continue
        combined = line
        if "Published terminal hover" not in combined:
            for next_record in records[index + 1 : index + 5]:
                if next_record["file"] == record["file"] and "Published terminal hover" in next_record["line"]:
                    combined = combined + " " + next_record["line"]
                    break
        parsed = parse_game_result_text(combined)
        if not parsed:
            continue
        parsed["source_uav"] = source_uav
        parsed["source_file"] = record["file"]
        parsed["mtime"] = record["mtime"]
        candidates.append((record["mtime"], index, parsed))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


class PipelineJob:
    def __init__(self, job_id, name, steps, cwd, env, nonblocking_stage=None):
        self.id = job_id
        self.name = name
        self.steps = steps
        self.cwd = cwd
        self.env = env
        self.nonblocking_stage = nonblocking_stage
        self.started_at = time.time()
        self.ended_at = None
        self.returncode = None
        self.current_proc = None
        self.current_stage = None
        self.lock = threading.Lock()
        self.log = deque(maxlen=700)
        self.stages = [
            {"label": label, "state": "PENDING", "detail": ""}
            for label, _cmd in steps
        ]
        self.runner = threading.Thread(target=self._run, daemon=True)
        self.runner.start()

    def _append(self, line):
        timestamp = time.strftime("%H:%M:%S")
        with self.lock:
            self.log.append(f"[{timestamp}] {line.rstrip()}")

    def _set_stage(self, index, state, detail=""):
        with self.lock:
            self.stages[index]["state"] = state
            self.stages[index]["detail"] = detail
            self.current_stage = index

    def _run(self):
        for index, (label, cmd) in enumerate(self.steps):
            self._set_stage(index, "RUNNING", " ".join(shlex_quote(part) for part in cmd))
            self._append(f"=== {label} ===")
            self._append("$ " + " ".join(shlex_quote(part) for part in cmd))
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=self.cwd,
                    env=self.env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,
                )
                with self.lock:
                    self.current_proc = proc
                if proc.stdout is not None:
                    for line in proc.stdout:
                        self._append(line)
                rc = proc.wait()
            except Exception as exc:
                self._append(f"stage error: {exc}")
                self._set_stage(index, "FAILED", str(exc))
                self.returncode = 1
                self.ended_at = time.time()
                return

            with self.lock:
                self.current_proc = None
            if rc != 0:
                self._append(f"{label} exit={rc}")
                self._set_stage(index, "FAILED", f"exit={rc}")
                self.returncode = rc
                self.ended_at = time.time()
                return
            self._append(f"{label} exit=0")
            self._set_stage(index, "OK", "exit=0")

        self.returncode = 0
        self.ended_at = time.time()
        with self.lock:
            self.current_stage = None

    def state(self):
        if self.returncode is None:
            return "RUNNING"
        return "OK" if self.returncode == 0 else "FAILED"

    def is_blocking(self):
        if self.state() != "RUNNING":
            return False
        with self.lock:
            if self.current_stage is None:
                return True
            label = self.stages[self.current_stage]["label"]
        return label != self.nonblocking_stage

    def to_summary(self):
        with self.lock:
            stages = [dict(stage) for stage in self.stages]
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state(),
            "returncode": self.returncode,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "blocking": self.is_blocking(),
            "stages": stages,
        }

    def log_text(self):
        with self.lock:
            return "\n".join(self.log)


class GroundWebApp:
    def __init__(self, args):
        self.args = args
        self.config = load_config(args.config)
        self.project_dir = Path(self.config["ground"].get("project", Path.cwd())).expanduser()
        self.shfiles_dir = self.project_dir / "core" / "shfiles"
        self.lock = threading.Lock()
        self.jobs = {}
        self.job_counter = 0
        self.local_roscore_job_id = None
        self.local_roscore_cache = None
        self.local_roscore_cache_at = 0.0
        self.status_cache = None
        self.status_cache_at = 0.0
        self.game_cache = None
        self.game_cache_at = 0.0
        self.odom_monitor = OdomMonitor(
            self.config,
            warn_age=args.warn_age,
            lost_age=args.lost_age,
            hz_window=args.hz_window,
        )

    def env(self):
        env = os.environ.copy()
        ground = self.config["ground"]
        env["ROS_MASTER_URI"] = str(ground["ros_master_uri"])
        env["ROS_IP"] = str(ground["ip"])
        env.pop("ROS_HOSTNAME", None)
        return env

    def script(self, name):
        return str(self.shfiles_dir / name)

    def local_roscore_reachable(self, timeout_s=0.5, ttl=1.0):
        now = time.monotonic()
        if self.local_roscore_cache is not None and ttl > 0.0 and now - self.local_roscore_cache_at < ttl:
            return self.local_roscore_cache
        ok = master_reachable(self.config["ground"]["ros_master_uri"], timeout_s=timeout_s)
        self.local_roscore_cache = ok
        self.local_roscore_cache_at = now
        return ok

    def invalidate_local_roscore_cache(self):
        self.local_roscore_cache = None
        self.local_roscore_cache_at = 0.0

    def validate_uav(self, value):
        if value in (None, "", "all"):
            return "all"
        targets = [item.strip() for item in str(value).split(",") if item.strip()]
        unknown = [name for name in targets if name not in self.config["uavs"]]
        if unknown:
            raise ValueError("unknown UAV target(s): " + ", ".join(unknown))
        return ",".join(targets)

    def validate_game_assignment(self, payload):
        defender0 = self.validate_uav(payload.get("defender0", "uav0"))
        defender1 = self.validate_uav(payload.get("defender1", "uav1"))
        enemy = self.validate_uav(payload.get("enemy", "uav2"))
        roles = (defender0, defender1, enemy)
        if any("," in role or role == "all" for role in roles):
            raise ValueError("game roles must be single UAV names")
        if len(set(roles)) != 3:
            raise ValueError("defender0, defender1, and enemy must be three different UAVs")
        return defender0, defender1, enemy

    def validate_game_role(self, value):
        role = str(value or "defender_0")
        allowed = {"defender_0", "defender_1", "enemy"}
        if role not in allowed:
            raise ValueError("game_role must be one of defender_0, defender_1, enemy")
        return role

    def validate_single_game_target(self, target):
        if target == "all" or "," in target:
            raise ValueError("single game test requires one UAV target")
        return target

    def new_job(self, name, cmd, blocking=True):
        with self.lock:
            self.job_counter += 1
            job_id = str(self.job_counter)
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.project_dir),
            env=self.env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        job = Job(job_id, name, cmd, proc, blocking=blocking)
        with self.lock:
            self.jobs[job_id] = job
            if len(self.jobs) > 30:
                for old_id in sorted(self.jobs, key=lambda item: int(item))[:-30]:
                    del self.jobs[old_id]
        return job

    def new_pipeline_job(self, name, steps, nonblocking_stage=None):
        with self.lock:
            self.job_counter += 1
            job_id = str(self.job_counter)
        job = PipelineJob(
            job_id,
            name,
            steps,
            cwd=str(self.project_dir),
            env=self.env(),
            nonblocking_stage=nonblocking_stage,
        )
        with self.lock:
            self.jobs[job_id] = job
            if len(self.jobs) > 30:
                for old_id in sorted(self.jobs, key=lambda item: int(item))[:-30]:
                    del self.jobs[old_id]
        return job

    def active_blocking_jobs(self):
        with self.lock:
            jobs = list(self.jobs.values())
        return [job for job in jobs if job.to_summary().get("blocking")]

    def reject_if_conflicting(self, action):
        blockers = self.active_blocking_jobs()
        if not blockers:
            return
        if action in ("stop", "game_stop"):
            return
        names = ", ".join(job.name for job in blockers)
        raise ValueError(f"operation blocked by running job(s): {names}")

    def start_local_roscore(self):
        if self.local_roscore_reachable(timeout_s=0.6, ttl=0.0):
            return {"ok": True, "message": "local roscore already reachable"}
        job = self.new_job("local roscore", ["roscore"], blocking=False)
        self.invalidate_local_roscore_cache()
        self.local_roscore_job_id = job.id
        return {"ok": True, "job_id": job.id}

    def stop_local_roscore(self):
        if not self.local_roscore_job_id:
            return {"ok": True, "message": "no web-started local roscore job"}
        with self.lock:
            job = self.jobs.get(str(self.local_roscore_job_id))
        if job is None or job.proc.poll() is not None:
            return {"ok": True, "message": "web-started local roscore is not running"}
        try:
            os.killpg(os.getpgid(job.proc.pid), signal.SIGTERM)
            try:
                job.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(job.proc.pid), signal.SIGKILL)
            job._append("local roscore stop requested from web")
            self.invalidate_local_roscore_cache()
            return {"ok": True, "job_id": job.id}
        except Exception as exc:
            raise ValueError(f"failed to stop web-started local roscore: {exc}")

    def action(self, payload):
        action = payload.get("action")
        target = self.validate_uav(payload.get("uav", "all"))
        land_timeout = max(1.0, safe_float(payload.get("land_timeout"), 8.0))
        self.reject_if_conflicting(action)
        if action == "start_roscore":
            return self.start_local_roscore()
        if action == "stop_local_roscore":
            return self.stop_local_roscore()
        if action == "prep":
            start_rviz = "true" if payload.get("start_rviz") else "false"
            steps = [
                ("roscore", [self.script("groundctl.sh"), "start", "roscore", "--uav", target]),
                ("core", [self.script("groundctl.sh"), "start", "core", "--uav", target]),
                ("vins-sync", [self.script("groundctl.sh"), "start", "vins-sync", "--uav", target]),
                ("groundsync", [self.script("groundsync.sh"), "sync", f"start_rviz:={start_rviz}"]),
            ]
            job = self.new_pipeline_job(f"prep {target}", steps, nonblocking_stage="groundsync")
            return {"ok": True, "job_id": job.id}
        if action == "takeoff":
            cmd = [self.script("groundctl.sh"), "takeoff", "--uav", target]
            job = self.new_job(f"takeoff {target}", cmd)
            return {"ok": True, "job_id": job.id}
        if action == "land":
            timeout_bin = shutil.which("timeout")
            base_cmd = [self.script("groundctl.sh"), "land", "--uav", target, "--connect-timeout", "1"]
            if timeout_bin:
                cmd = [timeout_bin, "--kill-after=2s", f"{land_timeout:.0f}s"] + base_cmd
            else:
                cmd = base_cmd
            job = self.new_job(f"land {target}", cmd)
            return {"ok": True, "job_id": job.id}
        if action == "stop":
            cmd = [
                self.script("groundstop.sh"),
                "--uav",
                target,
                "--land-timeout",
                f"{land_timeout:.0f}",
            ]
            job = self.new_job(f"stop {target}", cmd)
            return {"ok": True, "job_id": job.id}
        if action == "game_start":
            if target != "all":
                single_target = self.validate_single_game_target(target)
                game_role = self.validate_game_role(payload.get("game_role", "defender_0"))
                cmd = [
                    self.script("groundgame.sh"),
                    "single",
                    "--uav",
                    single_target,
                    "--role",
                    game_role,
                ]
                kind = "mpc" if game_role == "enemy" else "adv"
                job = self.new_job(f"game single {single_target} role={game_role} kind={kind}", cmd)
                return {"ok": True, "job_id": job.id}
            defender0, defender1, enemy = self.validate_game_assignment(payload)
            cmd = [
                self.script("groundgame.sh"),
                "start",
                "--defender0",
                defender0,
                "--defender1",
                defender1,
                "--enemy",
                enemy,
            ]
            job = self.new_job(f"game start d0={defender0} d1={defender1} enemy={enemy}", cmd)
            return {"ok": True, "job_id": job.id}
        if action == "game_stop":
            cmd = [self.script("groundgame.sh"), "stop", "--uav", target, "--force", "--jobs", "3"]
            job = self.new_job(f"game stop {target}", cmd)
            return {"ok": True, "job_id": job.id}
        raise ValueError(f"unknown action: {action}")

    def link_status(self):
        now = time.monotonic()
        if self.status_cache is not None and now - self.status_cache_at < self.args.status_ttl:
            return self.status_cache

        rows = []
        with ThreadPoolExecutor(max_workers=max(1, len(self.config["uavs"]) * 2)) as executor:
            futures = {}
            for name, conf in self.config["uavs"].items():
                futures[executor.submit(ssh_reachable, conf["ssh_host"])] = (name, "ssh")
                futures[executor.submit(master_reachable, conf["ros_master_uri"], 1.0)] = (name, "ros_master")
            values = {name: {"name": name, **conf} for name, conf in self.config["uavs"].items()}
            for future in as_completed(futures):
                name, key = futures[future]
                try:
                    values[name][key] = bool(future.result())
                except Exception:
                    values[name][key] = False
        for name in ordered_uavs(self.config):
            rows.append(values[name])

        self.status_cache = rows
        self.status_cache_at = now
        return rows

    def remote_game_probe(self, name, conf):
        log_dir = str(self.config.get("runtime", {}).get("remote_log_dir", "logs/groundctrl")).rstrip("/")
        command = (
            f"cd {shlex_quote(conf['project'])} 2>/dev/null || exit 0; "
            f"PAT={shlex_quote(GAME_PROCESS_PATTERN)}; "
            "printf '__RUNNING__\\n'; "
            'pgrep -u "$(id -u)" -af "$PAT" 2>/dev/null || true; '
            "printf '__GAMELOG__\\n'; "
            f"for f in {shlex_quote(log_dir)}/adv_*.log {shlex_quote(log_dir)}/mpc_*.log; do "
            '[ -f "$f" ] || continue; '
            'mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0); '
            'grep -aE "game ended|Game ended|Published terminal hover" "$f" 2>/dev/null | tail -n 8 | '
            'while IFS= read -r line; do printf "__LINE__%s\\t%s\\t%s\\n" "$mtime" "$f" "$line"; done; '
            "done"
        )
        row = {"name": name, "running": False, "processes": [], "error": "", "result": None}
        try:
            output = ssh_capture(conf["ssh_host"], command, timeout_s=max(3.0, self.args.status_ttl + 1.0))
        except Exception as exc:
            row["error"] = str(exc)
            return row

        mode = None
        records = []
        for line in output.splitlines():
            if line == "__RUNNING__":
                mode = "running"
                continue
            if line == "__GAMELOG__":
                mode = "log"
                continue
            if mode == "running":
                if line.strip():
                    row["processes"].append(line.strip())
                continue
            if line.startswith("__LINE__"):
                payload = line[len("__LINE__") :]
                parts = payload.split("\t", 2)
                if len(parts) != 3:
                    continue
                mtime = parse_float_token(parts[0]) or 0.0
                records.append({"mtime": mtime, "file": parts[1], "line": parts[2]})

        row["running"] = bool(row["processes"])
        row["processes"] = row["processes"][:5]
        row["result"] = parse_game_result_records(records, name)
        return row

    def game_snapshot(self):
        now = time.monotonic()
        if self.game_cache is not None and now - self.game_cache_at < self.args.status_ttl:
            return self.game_cache

        rows_by_name = {}
        with ThreadPoolExecutor(max_workers=max(1, len(self.config["uavs"]))) as executor:
            futures = {
                executor.submit(self.remote_game_probe, name, conf): name
                for name, conf in self.config["uavs"].items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    rows_by_name[name] = future.result()
                except Exception as exc:
                    rows_by_name[name] = {
                        "name": name,
                        "running": False,
                        "processes": [],
                        "error": str(exc),
                        "result": None,
                    }

        rows = [rows_by_name.get(name, {"name": name, "running": False, "processes": [], "error": "", "result": None}) for name in ordered_uavs(self.config)]
        results = [row["result"] for row in rows if row.get("result")]
        latest_result = None
        if results:
            latest_result = max(
                results,
                key=lambda item: (
                    float(item.get("mtime") or 0.0),
                    len(item.get("positions") or {}),
                    item.get("source_uav") or "",
                ),
            )

        snapshot = {
            "running": {row["name"]: bool(row.get("running")) for row in rows},
            "rows": rows,
            "result": latest_result,
        }
        self.game_cache = snapshot
        self.game_cache_at = now
        return snapshot

    def cached_game_snapshot(self):
        if self.game_cache is not None:
            return self.game_cache
        return {
            "running": {name: False for name in ordered_uavs(self.config)},
            "rows": [],
            "result": None,
        }

    def state(self):
        master_uri = self.config["ground"]["ros_master_uri"]
        local_ok = self.local_roscore_reachable(timeout_s=0.5, ttl=0.75)
        if local_ok:
            self.odom_monitor.ensure_started()
        with ThreadPoolExecutor(max_workers=2) as executor:
            link_future = executor.submit(self.link_status)
            game_future = executor.submit(self.game_snapshot)
            uavs = link_future.result()
            game = game_future.result()
        return {
            "time": time.strftime("%H:%M:%S"),
            "master_uri": master_uri,
            "local_roscore": local_ok,
            "groundsync": process_running("ground_sync.launch|ground_master_sync"),
            "uavs": uavs,
            "odom": self.odom_monitor.snapshot(),
            "odom_monitor_error": self.odom_monitor.error,
            "game_running": game["running"],
            "game_status": game["rows"],
            "game_result": game["result"],
        }

    def odom_state(self):
        local_ok = self.local_roscore_reachable(timeout_s=0.2, ttl=1.0)
        if local_ok:
            self.odom_monitor.ensure_started()
        game = self.cached_game_snapshot()
        return {
            "time": time.strftime("%H:%M:%S"),
            "local_roscore": local_ok,
            "odom": self.odom_monitor.snapshot(),
            "odom_monitor_error": self.odom_monitor.error,
            "game_running": game["running"],
            "game_result": game["result"],
        }

    def jobs_summary(self):
        with self.lock:
            jobs = [job.to_summary() for job in self.jobs.values()]
        jobs.sort(key=lambda item: int(item["id"]), reverse=True)
        return {"jobs": jobs}

    def job_log(self, job_id):
        with self.lock:
            job = self.jobs.get(str(job_id))
        if job is None:
            return {"ok": False, "error": "job not found", "log": ""}
        return {"ok": True, "job": job.to_summary(), "log": job.log_text()}


class RequestHandler(BaseHTTPRequestHandler):
    app = None

    def log_message(self, fmt, *args):
        message = fmt % args
        if '"GET /api/odom ' in message:
            return
        sys.stderr.write("[ground_web] " + message + "\n")

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, body):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_html(INDEX_HTML)
            return
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if parsed.path == "/api/state":
            self.send_json(self.app.state())
            return
        if parsed.path == "/api/odom":
            self.send_json(self.app.odom_state())
            return
        if parsed.path == "/api/jobs":
            self.send_json(self.app.jobs_summary())
            return
        if parsed.path == "/api/job":
            params = urllib.parse.parse_qs(parsed.query)
            job_id = (params.get("id") or [""])[0]
            self.send_json(self.app.job_log(job_id))
            return
        self.send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/action":
            self.send_json({"ok": False, "error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw)
            result = self.app.action(payload)
            self.send_json(result)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)


def build_parser():
    parser = argparse.ArgumentParser(description="Local web control panel for MUAV ground station")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="groundctrl.yaml path")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8080, help="HTTP bind port")
    parser.add_argument("--warn-age", type=float, default=1.0)
    parser.add_argument("--lost-age", type=float, default=3.0)
    parser.add_argument("--hz-window", type=float, default=3.0)
    parser.add_argument("--status-ttl", type=float, default=3.0, help="Seconds to cache SSH/ROS link checks")
    return parser


def run_server(args):
    app = GroundWebApp(args)
    RequestHandler.app = app
    last_error = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer((args.host, port), RequestHandler)
            url_host = "localhost" if args.host in ("127.0.0.1", "localhost") else args.host
            print(f"[ground_web] listening on http://{url_host}:{port}", flush=True)
            server.serve_forever()
            return
        except OSError as exc:
            last_error = exc
            if exc.errno not in (48, 98):
                raise
    raise SystemExit(f"could not bind a web port starting at {args.port}: {last_error}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_server(args)


if __name__ == "__main__":
    main()
