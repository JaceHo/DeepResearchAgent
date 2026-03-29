import atexit
import argparse
import asyncio
from collections import deque
import importlib
import json
import os
import socket
import threading
import time
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from src.agent import acp
from src.config import config
from src.environment import ecp
from src.logger import logger
from src.memory import memory_manager
from src.model import model_manager
from src.prompt import prompt_manager
from src.session.types import SessionContext
from src.skill import scp
from src.tool import tcp
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request
from mmengine import DictAction

load_dotenv(verbose=True)

TOOL_MODULES = {
    "bash": "src.tool.default_tools.bash",
    "python_interpreter": "src.tool.default_tools.python_interpreter",
    "done": "src.tool.default_tools.done",
    "todo": "src.tool.workflow_tools.todo",
    "skill_generator": "src.tool.workflow_tools.skill_generator",
    "ai_capability_debate": "src.tool.other_tools.ai_capability_debate",
    "mcp_importer": "src.tool.mcp_tools.importer",
}

ENVIRONMENT_MODULES = {
    "file_system": "src.environment.file_system_environment",
}


DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AgentServ Control Center</title>
    <style>
        :root {
            color-scheme: dark;
            --bg: #0b1020;
            --panel: #121936;
            --panel-2: #1a2348;
            --border: #2a376a;
            --text: #e8ecff;
            --muted: #9cadde;
            --good: #21c17a;
            --bad: #ff6b6b;
            --warn: #ffbf69;
            --accent: #7c9cff;
            --accent-2: #96f2ff;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: linear-gradient(180deg, #0b1020 0%, #101735 100%);
            color: var(--text);
        }

        a {
            color: var(--accent-2);
        }

        .shell {
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px;
        }

        .hero {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            margin-bottom: 20px;
            padding: 24px;
            border: 1px solid var(--border);
            background: rgba(18, 25, 54, 0.88);
            border-radius: 20px;
            backdrop-filter: blur(8px);
        }

        .hero h1 {
            margin: 0 0 8px;
            font-size: 30px;
        }

        .hero p {
            margin: 0;
            color: var(--muted);
            max-width: 900px;
        }

        .actions {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
        }

        button, select, textarea, input {
            font: inherit;
        }

        button {
            border: 1px solid var(--border);
            background: var(--panel-2);
            color: var(--text);
            border-radius: 12px;
            padding: 10px 14px;
            cursor: pointer;
        }

        button.primary {
            background: linear-gradient(135deg, var(--accent), #5b73ff);
            border-color: #6982ff;
        }

        button:hover {
            filter: brightness(1.08);
        }

        .toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--muted);
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 20px;
        }

        .card {
            background: rgba(18, 25, 54, 0.92);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px;
            min-height: 120px;
        }

        .card h3, .panel h2 {
            margin: 0 0 10px;
        }

        .metric {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .muted {
            color: var(--muted);
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid var(--border);
            font-size: 13px;
            white-space: nowrap;
        }

        .ok {
            color: var(--good);
        }

        .bad {
            color: var(--bad);
        }

        .warn {
            color: var(--warn);
        }

        .tabs {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 16px;
        }

        .tab {
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: rgba(26, 35, 72, 0.85);
        }

        .tab.active {
            background: linear-gradient(135deg, rgba(124, 156, 255, 0.35), rgba(150, 242, 255, 0.18));
            border-color: #7c9cff;
        }

        .panel {
            display: none;
            background: rgba(18, 25, 54, 0.92);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 18px;
        }

        .panel.active {
            display: block;
        }

        .two-col {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 16px;
        }

        .stack {
            display: grid;
            gap: 14px;
        }

        .table-wrap {
            overflow: auto;
            border: 1px solid var(--border);
            border-radius: 16px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            text-align: left;
            padding: 12px 14px;
            border-bottom: 1px solid rgba(124, 156, 255, 0.15);
            vertical-align: top;
        }

        th {
            color: var(--muted);
            font-weight: 600;
            background: rgba(11, 16, 32, 0.32);
        }

        pre {
            margin: 0;
            padding: 14px;
            white-space: pre-wrap;
            word-break: break-word;
            background: #091125;
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: auto;
            max-height: 520px;
        }

        textarea, input, select {
            width: 100%;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: #0b1020;
            color: var(--text);
            padding: 12px 14px;
        }

        textarea {
            min-height: 140px;
            resize: vertical;
        }

        .form-grid {
            display: grid;
            gap: 12px;
        }

        .inline {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }

        .chips {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .section-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
        }

        .small {
            font-size: 13px;
        }

        .empty {
            color: var(--muted);
            padding: 18px 0;
        }

        @media (max-width: 1100px) {
            .grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .two-col {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 700px) {
            .shell {
                padding: 14px;
            }

            .hero {
                padding: 18px;
            }

            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="shell">
        <section class="hero">
            <div>
                <h1>AgentServ Control Center</h1>
                <p>Unified dashboard for gateway health, dependency status, agent runtime state, task execution history, logs, and API docs.</p>
            </div>
            <div class="actions">
                <button id="refreshButton" class="primary">Refresh Now</button>
                <label class="toggle">
                    <input id="autoRefresh" type="checkbox" checked>
                    <span>Auto refresh</span>
                </label>
                <div id="lastUpdated" class="pill muted">Waiting for first load</div>
            </div>
        </section>

        <section class="grid">
            <div class="card">
                <div class="small muted">Gateway Health</div>
                <div id="metricHealth" class="metric">...</div>
                <div id="metricHealthMeta" class="muted">Loading</div>
            </div>
            <div class="card">
                <div class="small muted">Selected Agent</div>
                <div id="metricAgent" class="metric">...</div>
                <div id="metricAgentMeta" class="muted">Loading</div>
            </div>
            <div class="card">
                <div class="small muted">Dependencies Up</div>
                <div id="metricDeps" class="metric">...</div>
                <div id="metricDepsMeta" class="muted">Loading</div>
            </div>
            <div class="card">
                <div class="small muted">Tasks Tracked</div>
                <div id="metricTasks" class="metric">...</div>
                <div id="metricTasksMeta" class="muted">Loading</div>
            </div>
        </section>

        <div class="tabs">
            <button class="tab active" data-tab="overview">Overview</button>
            <button class="tab" data-tab="components">Components</button>
            <button class="tab" data-tab="run">Run Task</button>
            <button class="tab" data-tab="tasks">Tasks</button>
            <button class="tab" data-tab="logs">Logs</button>
            <button class="tab" data-tab="docs">Docs</button>
        </div>

        <section id="tab-overview" class="panel active">
            <div class="two-col">
                <div class="stack">
                    <div>
                        <div class="section-title">
                            <h2>Runtime Summary</h2>
                            <div id="overviewPills" class="chips"></div>
                        </div>
                        <div id="overviewSummary" class="table-wrap"></div>
                    </div>
                    <div>
                        <h2>Underlying Runtime</h2>
                        <pre id="overviewStatusJson">Loading...</pre>
                    </div>
                </div>
                <div class="stack">
                    <div>
                        <h2>Current Task</h2>
                        <pre id="currentTaskPane">No active task.</pre>
                    </div>
                    <div>
                        <h2>Quick Access</h2>
                        <div id="quickLinks" class="stack"></div>
                    </div>
                </div>
            </div>
        </section>

        <section id="tab-components" class="panel">
            <div class="section-title">
                <h2>Service and Component Status</h2>
                <div class="muted small">Gateway, dependencies, agents, tools, memories, and skills</div>
            </div>
            <div id="componentTables" class="stack"></div>
        </section>

        <section id="tab-run" class="panel">
            <div class="two-col">
                <div>
                    <h2>Run Agent Task</h2>
                    <div class="form-grid">
                        <div>
                            <label for="taskInput" class="small muted">Task</label>
                            <textarea id="taskInput">Use the done tool immediately with result exactly hello.</textarea>
                        </div>
                        <div>
                            <label for="agentSelect" class="small muted">Agent</label>
                            <select id="agentSelect"></select>
                        </div>
                        <div>
                            <label for="filesInput" class="small muted">Files</label>
                            <input id="filesInput" placeholder='["/absolute/path/file.txt"]'>
                        </div>
                        <div class="inline">
                            <button id="runTaskButton" class="primary">Run Task</button>
                            <div id="runTaskStatus" class="pill muted">Idle</div>
                        </div>
                    </div>
                </div>
                <div>
                    <h2>Realtime Task Trace</h2>
                    <div class="stack">
                        <div>
                            <h3>Input</h3>
                            <pre id="runInputPane">No task submitted yet.</pre>
                        </div>
                        <div>
                            <h3>Runtime Status</h3>
                            <pre id="runTracePane">No active dashboard task.</pre>
                        </div>
                        <div>
                            <h3>Verbose Running Log</h3>
                            <pre id="runVerbosePane">No verbose runtime log yet.</pre>
                        </div>
                        <div>
                            <h3>Final Output</h3>
                            <pre id="runResultPane">No task run from dashboard yet.</pre>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <section id="tab-tasks" class="panel">
            <div class="section-title">
                <h2>Task Status</h2>
                <button id="refreshTasksButton">Refresh Tasks</button>
            </div>
            <div class="stack">
                <div>
                    <h3>Active</h3>
                    <pre id="activeTaskPane">No active task.</pre>
                </div>
                <div>
                    <h3>History</h3>
                    <div id="taskHistoryTable" class="table-wrap"></div>
                </div>
            </div>
        </section>

        <section id="tab-logs" class="panel">
            <div class="section-title">
                <h2>Logs</h2>
                <div class="inline">
                    <select id="logSelect"></select>
                    <input id="logLinesInput" type="number" min="20" max="1000" value="200">
                    <button id="refreshLogsButton">Refresh Log</button>
                </div>
            </div>
            <div class="stack">
                <pre id="logMetaPane">Loading log sources...</pre>
                <pre id="logContentPane">Select a log source.</pre>
            </div>
        </section>

        <section id="tab-docs" class="panel">
            <div class="section-title">
                <h2>Docs and Management Guide</h2>
                <button id="refreshDocsButton">Refresh Docs</button>
            </div>
            <div class="stack">
                <div id="docsOverview" class="stack"></div>
                <div id="docsEndpoints" class="table-wrap"></div>
                <div id="docsServices" class="table-wrap"></div>
                <pre id="docsCommands"></pre>
            </div>
        </section>
    </div>

    <script>
        const state = {
            dashboard: null,
            tasks: null,
            docs: null,
            selectedLog: null,
            autoRefreshHandle: null,
            currentRunTaskId: null,
            taskPollHandle: null
        };

        function pretty(value) {
            return JSON.stringify(value, null, 2);
        }

        async function getJson(url, options = undefined) {
            const response = await fetch(url, options);
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message || response.statusText || "Request failed");
            }
            return data;
        }

        function setText(id, value) {
            document.getElementById(id).textContent = value;
        }

        function buildTable(headers, rows) {
            if (!rows.length) {
                return '<div class="empty">No data available.</div>';
            }
            const head = headers.map((header) => `<th>${header}</th>`).join("");
            const body = rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("");
            return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
        }

        function escapeHtml(value) {
            return String(value ?? "")
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;")
                .replaceAll("'", "&#39;");
        }

        function renderOverview(data) {
            const status = data.status;
            const health = data.health;
            const tasks = data.tasks;
            const deps = Object.entries(health.dependencies || {});
            const upCount = deps.filter(([, item]) => item.ok).length;
            const pills = [];
            pills.push(`<span class="pill ${health.ok ? "ok" : "bad"}">${health.ok ? "Healthy" : "Unhealthy"}</span>`);
            pills.push(`<span class="pill">Uptime ${status.uptime_s}s</span>`);
            pills.push(`<span class="pill">Tools ${status.tools.length}</span>`);
            pills.push(`<span class="pill">Agents ${status.agents.length}</span>`);
            document.getElementById("overviewPills").innerHTML = pills.join("");

            const summaryRows = [
                ["Gateway", "agentserv"],
                ["Selected Agent", escapeHtml(status.selected_agent)],
                ["Config", escapeHtml(status.config_path)],
                ["Workdir", escapeHtml(status.workdir)],
                ["MCP URL", escapeHtml(status.mcp_url || "disabled")],
                ["Dependencies Up", `${upCount}/${deps.length}`],
                ["Active Task", tasks.active ? escapeHtml(tasks.active.task_id) : "none"]
            ];
            document.getElementById("overviewSummary").innerHTML = buildTable(["Item", "Value"], summaryRows);
            document.getElementById("overviewStatusJson").textContent = pretty(status);
            document.getElementById("currentTaskPane").textContent = tasks.active ? pretty(tasks.active) : "No active task.";

            const quickLinks = [
                { label: "Health JSON", href: "/health" },
                { label: "Status JSON", href: "/api/status" },
                { label: "Tasks JSON", href: "/api/tasks" },
                { label: "Docs JSON", href: "/api/docs" },
                { label: "Dashboard JSON", href: "/api/dashboard" }
            ];
            document.getElementById("quickLinks").innerHTML = quickLinks
                .map((item) => `<a class="pill" href="${item.href}" target="_blank" rel="noreferrer">${item.label}</a>`)
                .join("");
        }

        function renderMetrics(data) {
            const health = data.health;
            const status = data.status;
            const tasks = data.tasks;
            const deps = Object.values(health.dependencies || {});
            const upCount = deps.filter((item) => item.ok).length;

            setText("metricHealth", health.ok ? "Healthy" : "Degraded");
            setText("metricHealthMeta", `Initialized: ${status.initialized}`);
            setText("metricAgent", status.selected_agent || "n/a");
            setText("metricAgentMeta", `${status.agents.length} agent(s) registered`);
            setText("metricDeps", `${upCount}/${deps.length}`);
            setText("metricDepsMeta", "Dependencies responding");
            setText("metricTasks", String(tasks.summary.total));
            setText("metricTasksMeta", tasks.active ? `Active: ${tasks.active.task_id}` : "No active task");
        }

        function renderComponents(data) {
            const status = data.status;
            const health = data.health;
            const dependencyRows = Object.entries(health.dependencies || {}).map(([name, item]) => [
                escapeHtml(name),
                item.ok ? '<span class="ok">up</span>' : '<span class="bad">down</span>',
                escapeHtml(item.url || ""),
                `<pre>${escapeHtml(pretty(item.ok ? item.data : { error: item.error }))}</pre>`
            ]);

            const runtimeRows = [
                ["Selected Agent", escapeHtml(status.selected_agent)],
                ["Imported MCP Tools", escapeHtml((status.imported_mcp_tools || []).join(", ") || "none")],
                ["Memories", escapeHtml((status.memories || []).join(", ") || "none")],
                ["Skills", escapeHtml((status.skills || []).join(", ") || "none")],
                ["Config", escapeHtml(status.config_path)],
                ["Workdir", escapeHtml(status.workdir)]
            ];

            const agentRows = (status.agents || []).map((name) => [escapeHtml(name)]);
            const toolRows = (status.tools || []).map((name) => [escapeHtml(name)]);
            const logsRows = (data.logs.sources || []).map((item) => [
                escapeHtml(item.name),
                item.exists ? '<span class="ok">present</span>' : '<span class="warn">missing</span>',
                escapeHtml(item.path),
                escapeHtml(String(item.size_bytes ?? 0))
            ]);

            document.getElementById("componentTables").innerHTML = [
                `<div><h3>Dependencies</h3><div class="table-wrap">${buildTable(["Name", "Status", "URL", "Details"], dependencyRows)}</div></div>`,
                `<div><h3>Runtime</h3><div class="table-wrap">${buildTable(["Item", "Value"], runtimeRows)}</div></div>`,
                `<div class="two-col"><div><h3>Agents</h3><div class="table-wrap">${buildTable(["Agent"], agentRows)}</div></div><div><h3>Tools</h3><div class="table-wrap">${buildTable(["Tool"], toolRows)}</div></div></div>`,
                `<div><h3>Log Sources</h3><div class="table-wrap">${buildTable(["Name", "Availability", "Path", "Bytes"], logsRows)}</div></div>`
            ].join("");
        }

        function renderTasks(data) {
            document.getElementById("activeTaskPane").textContent = data.active ? pretty(data.active) : "No active task.";
            const rows = (data.history || []).map((task) => [
                escapeHtml(task.task_id),
                escapeHtml(task.status),
                escapeHtml(task.agent_name),
                escapeHtml(task.started_at || ""),
                escapeHtml(String(task.duration_s ?? "")),
                `<pre>${escapeHtml(task.task)}</pre>`,
                `<pre>${escapeHtml(pretty(task.result || { message: task.message, error: task.error }))}</pre>`
            ]);
            document.getElementById("taskHistoryTable").innerHTML = buildTable(
                ["Task ID", "Status", "Agent", "Started", "Duration(s)", "Task", "Result"],
                rows
            );
        }

        function renderRunTaskDetail(detail) {
            const task = detail.task;
            const log = detail.log || {};
            state.currentRunTaskId = task.task_id;
            setText("runTaskStatus", task.status || "unknown");
            document.getElementById("runInputPane").textContent = pretty(task.input || {
                task: task.task,
                files: task.files || [],
                agent_name: task.agent_name
            });
            document.getElementById("runTracePane").textContent = pretty({
                task_id: task.task_id,
                status: task.status,
                submitted_at: task.submitted_at,
                started_at: task.started_at,
                finished_at: task.finished_at,
                duration_s: task.duration_s,
                agent_name: task.agent_name,
                message: task.message,
                events: task.events || []
            });
            document.getElementById("runVerbosePane").textContent = log.content || "No verbose runtime log yet.";
            if (["completed", "failed", "incomplete"].includes(task.status)) {
                document.getElementById("runResultPane").textContent = pretty(
                    task.result || {
                        success: task.success,
                        message: task.message,
                        error: task.error
                    }
                );
            } else {
                document.getElementById("runResultPane").textContent = "Task still running...";
            }
        }

        async function refreshTaskDetail(taskId) {
            const detail = await getJson(`/api/tasks/${encodeURIComponent(taskId)}`);
            renderRunTaskDetail(detail);
            return detail;
        }

        function stopTaskPolling() {
            if (state.taskPollHandle) {
                clearInterval(state.taskPollHandle);
                state.taskPollHandle = null;
            }
        }

        function startTaskPolling(taskId) {
            state.currentRunTaskId = taskId;
            stopTaskPolling();

            async function pollOnce() {
                try {
                    const detail = await refreshTaskDetail(taskId);
                    await refreshTasks();
                    if (["completed", "failed", "incomplete"].includes(detail.task.status)) {
                        stopTaskPolling();
                        await refreshDashboard();
                    }
                } catch (error) {
                    setText("runTaskStatus", `poll failed: ${error.message}`);
                    stopTaskPolling();
                }
            }

            pollOnce();
            state.taskPollHandle = setInterval(pollOnce, 1000);
        }

        function renderLogSources(data) {
            const select = document.getElementById("logSelect");
            const sources = data.logs.sources || [];
            select.innerHTML = sources.map((item) => `<option value="${item.name}">${item.name}</option>`).join("");
            if (!state.selectedLog && sources.length) {
                state.selectedLog = sources[0].name;
            }
            if (state.selectedLog) {
                select.value = state.selectedLog;
            }
            document.getElementById("logMetaPane").textContent = pretty(sources);
        }

        function renderDocs(data) {
            const docs = data.docs;
            document.getElementById("docsOverview").innerHTML = [
                `<div class="card"><div class="small muted">Dashboard</div><div class="metric">${escapeHtml(docs.title)}</div><div class="muted">${escapeHtml(docs.summary)}</div></div>`,
                `<div class="card"><div class="small muted">How To Use</div><div class="muted">${escapeHtml(docs.how_to_use)}</div></div>`
            ].join("");

            const endpointRows = (docs.endpoints || []).map((item) => [
                escapeHtml(item.method),
                `<a href="${item.path}" target="_blank" rel="noreferrer">${escapeHtml(item.path)}</a>`,
                escapeHtml(item.purpose)
            ]);
            document.getElementById("docsEndpoints").innerHTML = buildTable(["Method", "Path", "Purpose"], endpointRows);

            const serviceRows = (docs.services || []).map((item) => [
                escapeHtml(item.name),
                escapeHtml(item.url || "internal"),
                escapeHtml(item.purpose),
                escapeHtml(item.manage)
            ]);
            document.getElementById("docsServices").innerHTML = buildTable(["Service", "URL", "Purpose", "Manage"], serviceRows);

            const commands = (docs.commands || []).map((item) => `${item.name}\\n${item.command}`).join("\\n\\n");
            document.getElementById("docsCommands").textContent = commands || "No command docs.";
        }

        async function refreshDashboard() {
            const data = await getJson("/api/dashboard");
            state.dashboard = data;
            renderMetrics(data);
            renderOverview(data);
            renderComponents(data);
            renderLogSources(data);
            renderDocs(data);

            const agentSelect = document.getElementById("agentSelect");
            const selectedAgent = document.getElementById("agentSelect").value || data.status.selected_agent;
            agentSelect.innerHTML = (data.status.agents || []).map((agent) => `<option value="${agent}">${agent}</option>`).join("");
            if ((data.status.agents || []).includes(selectedAgent)) {
                agentSelect.value = selectedAgent;
            }
            setText("lastUpdated", `Last updated ${new Date().toLocaleTimeString()}`);
        }

        async function refreshTasks() {
            const data = await getJson("/api/tasks");
            state.tasks = data;
            renderTasks(data);
            if (data.active && data.active.task_id) {
                if (state.currentRunTaskId !== data.active.task_id) {
                    startTaskPolling(data.active.task_id);
                }
            } else if (!state.currentRunTaskId && (data.history || []).length) {
                try {
                    await refreshTaskDetail(data.history[0].task_id);
                } catch (error) {
                    document.getElementById("runTracePane").textContent = String(error);
                }
            }
        }

        async function refreshLog() {
            const lines = Number(document.getElementById("logLinesInput").value || "200");
            const selected = document.getElementById("logSelect").value || state.selectedLog;
            if (!selected) {
                document.getElementById("logContentPane").textContent = "No log source available.";
                return;
            }
            state.selectedLog = selected;
            const data = await getJson(`/api/logs?name=${encodeURIComponent(selected)}&lines=${encodeURIComponent(lines)}`);
            document.getElementById("logMetaPane").textContent = pretty({
                name: data.name,
                path: data.path,
                exists: data.exists,
                line_count: data.line_count,
                size_bytes: data.size_bytes
            });
            document.getElementById("logContentPane").textContent = data.content || "";
        }

        async function refreshDocs() {
            const data = await getJson("/api/docs");
            state.docs = data;
            renderDocs({ docs: data });
        }

        async function submitTask() {
            const task = document.getElementById("taskInput").value.trim();
            const agentName = document.getElementById("agentSelect").value || null;
            const filesRaw = document.getElementById("filesInput").value.trim();
            let files = [];
            if (!task) {
                setText("runTaskStatus", "Task required");
                return;
            }
            if (filesRaw) {
                try {
                    files = JSON.parse(filesRaw);
                    if (!Array.isArray(files)) {
                        throw new Error("files must be an array");
                    }
                } catch (error) {
                    setText("runTaskStatus", error.message);
                    return;
                }
            }

            setText("runTaskStatus", "Submitting");
            document.getElementById("runInputPane").textContent = pretty({ task, files, agent_name: agentName });
            document.getElementById("runTracePane").textContent = "Submitting task to agentserv...";
            document.getElementById("runVerbosePane").textContent = "Waiting for runtime log...";
            document.getElementById("runResultPane").textContent = "Waiting for final output...";

            try {
                const data = await getJson("/api/run/async", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ task, files, agent_name: agentName })
                });
                setText("runTaskStatus", data.success ? "Queued" : "Rejected");
                document.getElementById("runTracePane").textContent = pretty(data.task || data);
                if (data.task && data.task.task_id) {
                    startTaskPolling(data.task.task_id);
                }
                await refreshDashboard();
                await refreshTasks();
            } catch (error) {
                setText("runTaskStatus", "Failed");
                document.getElementById("runTracePane").textContent = String(error);
                document.getElementById("runResultPane").textContent = String(error);
                await refreshTasks();
            }
        }

        function setupTabs() {
            document.querySelectorAll(".tab").forEach((button) => {
                button.addEventListener("click", () => {
                    const tab = button.dataset.tab;
                    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
                    document.querySelectorAll(".panel").forEach((item) => item.classList.remove("active"));
                    button.classList.add("active");
                    document.getElementById(`tab-${tab}`).classList.add("active");
                });
            });
        }

        function setupRefresh() {
            async function refreshAll() {
                await refreshDashboard();
                await refreshTasks();
                await refreshLog();
            }

            document.getElementById("refreshButton").addEventListener("click", refreshAll);
            document.getElementById("refreshTasksButton").addEventListener("click", refreshTasks);
            document.getElementById("refreshLogsButton").addEventListener("click", refreshLog);
            document.getElementById("refreshDocsButton").addEventListener("click", refreshDocs);
            document.getElementById("runTaskButton").addEventListener("click", submitTask);
            document.getElementById("logSelect").addEventListener("change", refreshLog);

            const checkbox = document.getElementById("autoRefresh");
            function updateAutoRefresh() {
                if (state.autoRefreshHandle) {
                    clearInterval(state.autoRefreshHandle);
                    state.autoRefreshHandle = null;
                }
                if (checkbox.checked) {
                    state.autoRefreshHandle = setInterval(async () => {
                        try {
                            await refreshAll();
                        } catch (error) {
                            setText("lastUpdated", `Refresh failed: ${error.message}`);
                        }
                    }, 5000);
                }
            }

            checkbox.addEventListener("change", updateAutoRefresh);
            updateAutoRefresh();
            refreshAll().catch((error) => {
                setText("lastUpdated", `Initial load failed: ${error.message}`);
                document.getElementById("overviewStatusJson").textContent = String(error);
            });
        }

        setupTabs();
        setupRefresh();
    </script>
</body>
</html>
"""


def load_runtime_modules(module_map: Dict[str, str], names: List[str]) -> None:
    for name in names:
        module_path = module_map.get(name)
        if module_path:
            importlib.import_module(module_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="agentserv")
    parser.add_argument(
        "--config",
        default=os.getenv(
            "AGENTSERV_CONFIG",
            os.path.join(os.path.dirname(__file__), "configs", "tool_calling_agent.py"),
        ),
    )
    parser.add_argument("--host", default=os.getenv("AGENTSERV_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AGENTSERV_PORT", "4002")))
    parser.add_argument("--agent-name", default=os.getenv("AGENTSERV_AGENT_NAME", "tool_calling"))
    parser.add_argument("--mcp-url", default=os.getenv("AGENTSERV_MCP_URL", "http://127.0.0.1:4001/hub/mcp"))
    parser.add_argument("--memory-url", default=os.getenv("AGENTSERV_MEMORY_URL", "http://127.0.0.1:18800/stats"))
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config values in key=value format",
    )
    return parser.parse_args()


class AgentServRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.started_at = time.time()
        self.task_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.initialized = False
        self.imported_mcp_tools: List[str] = []
        self.selected_agent_name = args.agent_name
        self.init_error: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._shutdown_started = False
        self.current_task: Optional[Dict[str, Any]] = None
        self.task_history: List[Dict[str, Any]] = []
        self.max_task_history = 50

    def start_loop(self) -> None:
        if self._loop is not None and self._loop_thread is not None and self._loop_thread.is_alive():
            return

        ready = threading.Event()

        def runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            ready.set()
            loop.run_forever()
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

        self._loop_thread = threading.Thread(target=runner, name="agentserv-runtime", daemon=True)
        self._loop_thread.start()
        ready.wait()

    def run_async(self, coro: asyncio.Future) -> Any:
        self.start_loop()
        if self._loop is None:
            raise RuntimeError("Runtime event loop is not available")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def initialize_sync(self) -> None:
        self.run_async(self.initialize())

    def status_sync(self) -> Dict[str, Any]:
        return self.run_async(self.status())

    def _runtime_log_path(self) -> Optional[str]:
        for source in self.log_sources():
            if source["name"] == "agent_runtime":
                return source["path"]
        return None

    def access_hosts(self) -> List[str]:
        hosts = ["127.0.0.1"]
        try:
            candidates = sorted({addr[4][0] for addr in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)})
            for candidate in candidates:
                if candidate not in hosts and candidate != "0.0.0.0":
                    hosts.append(candidate)
        except Exception:
            pass
        configured_host = self.args.host
        if configured_host not in hosts and configured_host != "0.0.0.0":
            hosts.insert(0, configured_host)
        return hosts

    def access_urls(self) -> List[str]:
        port = self.args.port
        return [f"http://{host}:{port}" for host in self.access_hosts()]

    def _timestamp(self, value: Optional[float] = None) -> str:
        actual = time.time() if value is None else value
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(actual))

    def _append_task_event(self, task_id: str, message: str, level: str = "info") -> None:
        event = {
            "timestamp": self._timestamp(),
            "level": level,
            "message": message,
        }
        with self.state_lock:
            for task_ref in [self.current_task, *self.task_history]:
                if task_ref is not None and task_ref.get("task_id") == task_id:
                    events = list(task_ref.get("events") or [])
                    events.append(event)
                    task_ref["events"] = events[-100:]

    def _create_task_record(
        self,
        task: str,
        files: Optional[List[str]] = None,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected_agent = agent_name or self.selected_agent_name
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        runtime_log_path = self._runtime_log_path()
        log_start_offset = 0
        if runtime_log_path and os.path.exists(runtime_log_path):
            log_start_offset = os.path.getsize(runtime_log_path)
        return {
            "task_id": task_id,
            "task": task,
            "files": files or [],
            "agent_name": selected_agent,
            "status": "queued",
            "submitted_at": self._timestamp(),
            "started_at": None,
            "finished_at": None,
            "duration_s": None,
            "success": None,
            "message": "Task queued",
            "result": None,
            "error": None,
            "input": {
                "task": task,
                "files": files or [],
                "agent_name": selected_agent,
            },
            "events": [
                {
                    "timestamp": self._timestamp(),
                    "level": "info",
                    "message": f"Queued task for agent {selected_agent}",
                }
            ],
            "log_path": runtime_log_path,
            "log_start_offset": log_start_offset,
        }

    def _register_task_record(self, task_record: Dict[str, Any]) -> None:
        with self.state_lock:
            self.current_task = dict(task_record)
            self.task_history.insert(0, dict(task_record))
            if len(self.task_history) > self.max_task_history:
                self.task_history = self.task_history[:self.max_task_history]

    def _update_task_tracking(self, task_id: str, **updates: Any) -> None:
        with self.state_lock:
            if self.current_task is not None and self.current_task.get("task_id") == task_id:
                self.current_task.update(updates)
            for item in self.task_history:
                if item.get("task_id") == task_id:
                    item.update(updates)
                    break

    def task_snapshot(self) -> Dict[str, Any]:
        with self.state_lock:
            active = dict(self.current_task) if self.current_task else None
            history = [dict(item) for item in self.task_history]
        return {
            "active": active,
            "history": history,
            "summary": {
                "total": len(history),
                "running": 1 if active else 0,
                "completed": sum(1 for item in history if item.get("status") == "completed"),
                "failed": sum(1 for item in history if item.get("status") == "failed"),
                "incomplete": sum(1 for item in history if item.get("status") == "incomplete"),
            },
        }

    def _read_task_log_excerpt(
        self,
        path: Optional[str],
        start_offset: int = 0,
        max_bytes: int = 120000,
    ) -> Dict[str, Any]:
        if not path:
            return {
                "path": None,
                "exists": False,
                "size_bytes": 0,
                "start_offset": start_offset,
                "end_offset": start_offset,
                "truncated": False,
                "content": "",
            }
        if not os.path.exists(path):
            return {
                "path": path,
                "exists": False,
                "size_bytes": 0,
                "start_offset": start_offset,
                "end_offset": start_offset,
                "truncated": False,
                "content": "",
            }

        size_bytes = os.path.getsize(path)
        safe_start = max(0, min(start_offset, size_bytes))
        read_start = safe_start
        truncated = False
        if size_bytes - safe_start > max_bytes:
            read_start = max(safe_start, size_bytes - max_bytes)
            truncated = True

        with open(path, "rb") as handle:
            handle.seek(read_start)
            content = handle.read().decode("utf-8", errors="replace")

        return {
            "path": path,
            "exists": True,
            "size_bytes": size_bytes,
            "start_offset": safe_start,
            "end_offset": size_bytes,
            "truncated": truncated,
            "content": content,
        }

    def task_detail(self, task_id: str) -> Dict[str, Any]:
        with self.state_lock:
            task_record = None
            if self.current_task is not None and self.current_task.get("task_id") == task_id:
                task_record = dict(self.current_task)
            else:
                for item in self.task_history:
                    if item.get("task_id") == task_id:
                        task_record = dict(item)
                        break
        if task_record is None:
            raise ValueError(f"Task not found: {task_id}")

        return {
            "task": task_record,
            "log": self._read_task_log_excerpt(
                task_record.get("log_path"),
                start_offset=int(task_record.get("log_start_offset") or 0),
            ),
        }

    def _execute_task_record(self, task_record: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task_record["task_id"]
        task = task_record["task"]
        files = task_record.get("files") or []
        agent_name = task_record["agent_name"]

        with self.task_lock:
            started_at = time.time()
            self._update_task_tracking(
                task_id,
                status="running",
                started_at=self._timestamp(started_at),
                message="Task running",
            )
            self._append_task_event(task_id, f"Started task with agent {agent_name}")
            self._append_task_event(task_id, f"Input task: {task}")
            if files:
                self._append_task_event(task_id, f"Input files: {files}")

            try:
                response = self.run_async(self.run_task(task=task, files=files, agent_name=agent_name))
                finished_at = time.time()
                final_status = "completed" if response.get("success") else "incomplete"
                updates = {
                    "status": final_status,
                    "finished_at": self._timestamp(finished_at),
                    "duration_s": round(finished_at - started_at, 3),
                    "success": response.get("success"),
                    "message": response.get("message"),
                    "result": response,
                }
                self._update_task_tracking(task_id, **updates)
                self._append_task_event(task_id, f"Task finished with status {final_status}")
                output = dict(response)
                output["task_id"] = task_id
                return output
            except Exception as exc:
                finished_at = time.time()
                updates = {
                    "status": "failed",
                    "finished_at": self._timestamp(finished_at),
                    "duration_s": round(finished_at - started_at, 3),
                    "success": False,
                    "message": str(exc),
                    "error": str(exc),
                }
                self._update_task_tracking(task_id, **updates)
                self._append_task_event(task_id, f"Task failed: {exc}", level="error")
                raise
            finally:
                with self.state_lock:
                    if self.current_task is not None and self.current_task.get("task_id") == task_id:
                        self.current_task = None

    def start_task_async(
        self,
        task: str,
        files: Optional[List[str]] = None,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.state_lock:
            if self.current_task is not None:
                raise RuntimeError(f"Task already running: {self.current_task.get('task_id')}")
        task_record = self._create_task_record(task=task, files=files, agent_name=agent_name)
        self._register_task_record(task_record)

        def worker() -> None:
            try:
                self._execute_task_record(task_record)
            except Exception:
                pass

        threading.Thread(
            target=worker,
            name=f"agentserv-task-{task_record['task_id']}",
            daemon=True,
        ).start()
        return dict(task_record)

    def run_task_sync(
        self,
        task: str,
        files: Optional[List[str]] = None,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        task_record = self._create_task_record(task=task, files=files, agent_name=agent_name)
        self._register_task_record(task_record)
        return self._execute_task_record(task_record)

    async def shutdown(self) -> None:
        cleanup_steps = [
            ("agent", getattr(acp, "cleanup", None)),
            ("environment", getattr(ecp, "cleanup", None)),
            ("tool", getattr(tcp, "cleanup", None)),
            ("memory", getattr(memory_manager, "cleanup", None)),
            ("prompt", getattr(prompt_manager, "cleanup", None)),
            ("skill", getattr(scp, "cleanup", None)),
        ]
        for _, cleanup in cleanup_steps:
            if cleanup is None:
                continue
            try:
                await cleanup()
            except Exception as exc:
                logger.error(f"| Shutdown cleanup failed: {exc}")
        self.initialized = False

    def stop(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        try:
            if self._loop is not None and not self._loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(self.shutdown(), self._loop)
                future.result(timeout=30)
        except Exception as exc:
            logger.error(f"| Runtime shutdown failed: {exc}")
        finally:
            if self._loop is not None and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=5)
            self._loop = None
            self._loop_thread = None

    async def initialize(self) -> None:
        config.initialize(config_path=self.args.config, args=self.args)
        logger.initialize(config=config)

        await model_manager.initialize()
        await prompt_manager.initialize()
        await memory_manager.initialize(memory_names=config.memory_names)

        tool_names = list(getattr(config, "tool_names", []))
        if "mcp_importer" not in tool_names:
            tool_names.append("mcp_importer")
        load_runtime_modules(TOOL_MODULES, tool_names)
        await tcp.initialize(tool_names=tool_names)

        if self.args.mcp_url:
            importer = await tcp.get("mcp_importer")
            if importer is not None:
                result = await importer(
                    connections={
                        "toolserv": {
                            "transport": "http",
                            "url": self.args.mcp_url,
                        }
                    },
                    name_prefix="mcp",
                    override=True,
                )
                if not result.success:
                    raise RuntimeError(result.message)
                self.imported_mcp_tools = ["mcp_toolserv"]

        skill_names = getattr(config, "skill_names", None)
        await scp.initialize(skill_names=skill_names)
        load_runtime_modules(ENVIRONMENT_MODULES, list(getattr(config, "env_names", [])))
        await ecp.initialize(config.env_names)
        await acp.initialize(agent_names=config.agent_names)

        available_agents = await acp.list()
        if self.selected_agent_name not in available_agents:
            if available_agents:
                self.selected_agent_name = available_agents[0]
            else:
                raise RuntimeError("No agents were initialized")

        self.initialized = True

    async def run_task(
        self,
        task: str,
        files: Optional[List[str]] = None,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected_agent = agent_name or self.selected_agent_name
        agent = await acp.get(selected_agent)
        if agent is None:
            raise RuntimeError(f"Agent not found: {selected_agent}")

        response = await agent(task=task, files=files or [], ctx=SessionContext())
        return response.model_dump()

    def _fetch_json(self, url: str, timeout: float = 3.0) -> Dict[str, Any]:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
        return json.loads(payload)

    def dependency_status(self) -> Dict[str, Any]:
        checks = {
            "aiserv": "http://127.0.0.1:4000/health",
            "toolserv": "http://127.0.0.1:4001/api/status",
        }
        if self.args.memory_url:
            checks["agentmem"] = self.args.memory_url

        result: Dict[str, Any] = {}
        for name, url in checks.items():
            try:
                result[name] = {
                    "ok": True,
                    "url": url,
                    "data": self._fetch_json(url),
                }
            except Exception as exc:
                result[name] = {
                    "ok": False,
                    "url": url,
                    "error": str(exc),
                }
        return result

    def health_payload(self) -> Dict[str, Any]:
        dependencies = self.dependency_status()
        ok = self.initialized and all(item.get("ok") for item in dependencies.values())
        return {
            "ok": ok,
            "initialized": self.initialized,
            "selected_agent": self.selected_agent_name,
            "init_error": self.init_error,
            "dependencies": dependencies,
        }

    def log_sources(self) -> List[Dict[str, Any]]:
        project_root = os.path.dirname(__file__)
        configured_workdir = getattr(config, "workdir", None)
        configured_log_path = getattr(config, "log_path", None)
        resolved_workdir = None
        if configured_workdir:
            resolved_workdir = configured_workdir if os.path.isabs(configured_workdir) else os.path.join(project_root, configured_workdir)

        candidates = {
            "gateway_stdout": os.path.join(project_root, "logs", "agentserv.log"),
            "gateway_stderr": os.path.join(project_root, "logs", "agentserv.error.log"),
        }

        if resolved_workdir and configured_log_path:
            candidates["agent_runtime"] = (
                configured_log_path
                if os.path.isabs(configured_log_path)
                else os.path.join(resolved_workdir, configured_log_path)
            )

        sources: List[Dict[str, Any]] = []
        for name, path in candidates.items():
            exists = os.path.exists(path)
            size_bytes = os.path.getsize(path) if exists else 0
            sources.append(
                {
                    "name": name,
                    "path": path,
                    "exists": exists,
                    "size_bytes": size_bytes,
                }
            )
        return sources

    def read_log(self, name: str, lines: int = 200) -> Dict[str, Any]:
        sources = {item["name"]: item for item in self.log_sources()}
        if name not in sources:
            raise ValueError(f"Unknown log source: {name}")
        source = sources[name]
        path = source["path"]
        if not source["exists"]:
            return {
                "name": name,
                "path": path,
                "exists": False,
                "size_bytes": 0,
                "line_count": 0,
                "content": "",
            }
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            tail_lines = list(deque(handle, maxlen=max(1, lines)))
        return {
            "name": name,
            "path": path,
            "exists": True,
            "size_bytes": source["size_bytes"],
            "line_count": len(tail_lines),
            "content": "".join(tail_lines),
        }

    def docs_payload(self) -> Dict[str, Any]:
        port = self.args.port
        access_urls = self.access_urls()
        primary_url = access_urls[0]
        return {
            "title": "AgentServ Control Center",
            "summary": "Single-page dashboard for viewing gateway health, runtime components, task execution, logs, and management docs.",
            "how_to_use": "Open the root page for the dashboard, use the Run Task tab to execute the selected agent, Tasks for history, Logs for tailing gateway and agent logs, and Docs for service endpoints and operational commands.",
            "access_urls": access_urls,
            "endpoints": [
                {"method": "GET", "path": "/", "purpose": "Control UI dashboard"},
                {"method": "GET", "path": "/health", "purpose": "Gateway health and dependency checks"},
                {"method": "GET", "path": "/api/status", "purpose": "Gateway runtime inventory and dependency summary"},
                {"method": "GET", "path": "/api/tools", "purpose": "Selected agent and available tools"},
                {"method": "POST", "path": "/api/run", "purpose": "Run a task through agentserv"},
                {"method": "POST", "path": "/api/run/async", "purpose": "Start a task and poll live task state separately"},
                {"method": "GET", "path": "/api/tasks", "purpose": "Active task and recent task history"},
                {"method": "GET", "path": "/api/tasks/<task_id>", "purpose": "Detailed task state plus verbose runtime log excerpt"},
                {"method": "GET", "path": "/api/logs", "purpose": "Tail a named log source"},
                {"method": "GET", "path": "/api/docs", "purpose": "Management docs for services and routes"},
                {"method": "GET", "path": "/api/dashboard", "purpose": "Aggregated payload used by the control UI"},
            ],
            "services": [
                {
                    "name": "agentserv",
                    "url": primary_url,
                    "purpose": "Gateway that exposes the control API and delegates tasks to the selected agent.",
                    "manage": f"Launch with ./agentserv.sh start, bind on all interfaces by default, and access it via {', '.join(access_urls)}.",
                },
                {
                    "name": "aiserv",
                    "url": "http://127.0.0.1:4000/health",
                    "purpose": "Model-serving dependency used by configured providers and local proxy routes.",
                    "manage": "Check via ./agentserv.sh health or the dashboard health panel.",
                },
                {
                    "name": "toolserv",
                    "url": self.args.mcp_url,
                    "purpose": "External MCP-backed tool service imported through mcp_importer.",
                    "manage": "Verify imported MCP tools in the Components tab and the toolserv status endpoint.",
                },
                {
                    "name": "agentmem",
                    "url": self.args.memory_url,
                    "purpose": "Memory backend used for summaries, insights, and task event storage.",
                    "manage": "Check store stats from the dashboard dependencies section.",
                },
            ],
            "commands": [
                {"name": "Open Dashboard", "command": f"open {primary_url}/"},
                {"name": "Open LAN Dashboard", "command": f"open http://192.168.31.105:{port}/"},
                {"name": "Gateway Health", "command": "cd /Users/jace/code/agentserv && ./agentserv.sh health"},
                {"name": "Gateway Status", "command": "cd /Users/jace/code/agentserv && ./agentserv.sh status"},
                {"name": "Start AgentServ", "command": "cd /Users/jace/code/agentserv && ./agentserv.sh start"},
                {"name": "Start Foreground", "command": "cd /Users/jace/code/agentserv && ./agentserv.sh start --foreground"},
                {"name": "Stop AgentServ", "command": "cd /Users/jace/code/agentserv && ./agentserv.sh stop"},
                {"name": "Restart AgentServ", "command": "cd /Users/jace/code/agentserv && ./agentserv.sh restart"},
            ],
        }

    def dashboard_payload(self) -> Dict[str, Any]:
        return {
            "service": "agentserv",
            "health": self.health_payload(),
            "status": self.status_sync(),
            "tasks": self.task_snapshot(),
            "logs": {"sources": self.log_sources()},
            "docs": self.docs_payload(),
        }

    async def status(self) -> Dict[str, Any]:
        tools = await tcp.list()
        agents = await acp.list()
        memories = await memory_manager.list()
        skills = await scp.list()
        return {
            "service": "agentserv",
            "initialized": self.initialized,
            "config_path": self.args.config,
            "workdir": getattr(config, "workdir", None),
            "log_path": getattr(config, "log_path", None),
            "selected_agent": self.selected_agent_name,
            "agents": agents,
            "tools": tools,
            "memories": memories,
            "skills": skills,
            "mcp_url": self.args.mcp_url,
            "imported_mcp_tools": self.imported_mcp_tools,
            "uptime_s": round(time.time() - self.started_at, 3),
            "dependencies": self.dependency_status(),
        }


def create_app(runtime: AgentServRuntime) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.get("/health")
    def health():
        payload = runtime.health_payload()
        ok = payload["ok"]
        status_code = 200 if ok else 503
        return jsonify(payload), status_code

    @app.get("/api/status")
    def status():
        return jsonify(runtime.status_sync())

    @app.get("/api/tools")
    def tools():
        status_payload = runtime.status_sync()
        return jsonify(
            {
                "selected_agent": status_payload["selected_agent"],
                "tools": status_payload["tools"],
                "imported_mcp_tools": status_payload["imported_mcp_tools"],
            }
        )

    @app.get("/api/tasks")
    def tasks():
        return jsonify(runtime.task_snapshot())

    @app.get("/api/tasks/<task_id>")
    def task_detail(task_id: str):
        try:
            return jsonify(runtime.task_detail(task_id))
        except ValueError as exc:
            return jsonify({"success": False, "message": str(exc)}), 404

    @app.get("/api/logs")
    def logs():
        name = request.args.get("name")
        if not name:
            return jsonify({"sources": runtime.log_sources()})
        try:
            lines = int(request.args.get("lines", "200"))
        except ValueError:
            return jsonify({"success": False, "message": "lines must be an integer"}), 400
        try:
            return jsonify(runtime.read_log(name=name, lines=max(1, min(lines, 1000))))
        except ValueError as exc:
            return jsonify({"success": False, "message": str(exc)}), 404

    @app.get("/api/docs")
    def docs():
        return jsonify(runtime.docs_payload())

    @app.get("/api/dashboard")
    def dashboard():
        return jsonify(runtime.dashboard_payload())

    @app.post("/api/run/async")
    def run_task_async():
        payload = request.get_json(silent=True) or {}
        task = payload.get("task")
        files = payload.get("files") or []
        agent_name = payload.get("agent_name")

        if not isinstance(task, str) or not task.strip():
            return jsonify({"success": False, "message": "task is required"}), 400
        if not isinstance(files, list):
            return jsonify({"success": False, "message": "files must be a list"}), 400

        try:
            task_record = runtime.start_task_async(task=task, files=files, agent_name=agent_name)
            return jsonify({"success": True, "task": task_record}), 202
        except Exception as exc:
            return jsonify({"success": False, "message": str(exc)}), 409

    @app.post("/api/run")
    def run_task():
        payload = request.get_json(silent=True) or {}
        task = payload.get("task")
        files = payload.get("files") or []
        agent_name = payload.get("agent_name")

        if not isinstance(task, str) or not task.strip():
            return jsonify({"success": False, "message": "task is required"}), 400
        if not isinstance(files, list):
            return jsonify({"success": False, "message": "files must be a list"}), 400

        with runtime.task_lock:
            try:
                response = runtime.run_task_sync(task=task, files=files, agent_name=agent_name)
                return jsonify(response)
            except Exception as exc:
                return jsonify({"success": False, "message": str(exc)}), 500

    return app


def main() -> None:
    args = parse_args()
    runtime = AgentServRuntime(args)
    atexit.register(runtime.stop)
    try:
        runtime.initialize_sync()
    except Exception as exc:
        runtime.init_error = str(exc)
        runtime.stop()
        raise

    app = create_app(runtime)
    try:
        app.run(host=args.host, port=args.port)
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
