#!/usr/bin/env python3
"""
AI 安全测评运行脚本
用法：
  1. pip install openai pyyaml tenacity rich
  2. 修改下方 CONFIG 里的 API_KEY
  3. python runner.py
"""
#{"_type":"newapi_channel_conn","key":"sk-CQhHNJztwKRuXCbaxhAKHS6BGI7UtjJ1sjZDpKFtgalYWoJ1","url":"https://ai.babeltower.hk/v1/chat/completions"}
import os
import json
import yaml
import time
import hashlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# ============ CONFIG ============
CONFIG = {
    "api_key": "sk-MpJzvvxNhiXq72p3L6O2tx7XXEP8DLnp79CsHSSumHWNOn0v",
    "base_url": "https://ai.babeltower.hk/v1",

    # 便宜模型先测（脚本跑通了再换 Claude / GPT）
    "models": [
        {"id": "glm",      "name": "GLM-5.2",         "api_model": "glm-5.2"},
        {"id": "deepseek", "name": "DeepSeek-V4-Pro", "api_model": "deepseek-v4-pro"},
        # 脚本验证 OK 后取消下面注释
        # {"id": "claude",  "name": "Claude 3.5 Sonnet", "api_model": "claude-3-5-sonnet-20241022"},
        {"id": "gpt",     "name": "gpt-5.5",            "api_model": "gpt-5.5"},
    ],

    "tasks_file": "tasks.yaml",
    "output_dir": "results",
    "runs_per_prompt": 3,
    "temperature": 0.7,
    "max_tokens": 4000,
    "timeout": 120,
    "max_workers": 3,      # 并发数，太高容易撞限流
    "system_prompt": "You are a helpful assistant. Answer in the same language as the user's question.",
}
# ================================


console = Console()


def load_tasks(path: str) -> list:
    """加载 tasks.yaml 并展开 sub_prompts"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    flat_tasks = []
    for task in data["tasks"]:
        if "sub_prompts" in task:
            for sub in task["sub_prompts"]:
                flat_tasks.append({
                    "id": f"{task['id']}__{sub['id']}",
                    "parent_id": task["id"],
                    "dimension": task["dimension"],
                    "sub_label": sub.get("framing") or sub.get("id"),
                    "prompt": sub["prompt_user"],
                    "description": task.get("description", ""),
                })
        else:
            flat_tasks.append({
                "id": task["id"],
                "parent_id": task["id"],
                "dimension": task["dimension"],
                "sub_label": None,
                "prompt": task["prompt_user"],
                "description": task.get("description", ""),
            })
    return flat_tasks


def response_path(task_id: str, model_id: str, run_idx: int) -> Path:
    return Path(CONFIG["output_dir"]) / "raw" / f"{task_id}__{model_id}__run{run_idx}.json"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def call_model(client: OpenAI, api_model: str, prompt: str) -> dict:
    """带重试的模型调用"""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=api_model,
        messages=[
            {"role": "system", "content": CONFIG["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        temperature=CONFIG["temperature"],
        max_tokens=CONFIG["max_tokens"],
        timeout=CONFIG["timeout"],
    )
    elapsed = time.time() - t0

    usage = resp.usage
    return {
        "content": resp.choices[0].message.content,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
        "elapsed_seconds": round(elapsed, 2),
        "finish_reason": resp.choices[0].finish_reason,
    }


def run_single(client: OpenAI, task: dict, model: dict, run_idx: int) -> dict:
    """单个 (task, model, run) 组合"""
    out_path = response_path(task["id"], model["id"], run_idx)

    # 断点续跑
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    result = {
        "task_id": task["id"],
        "dimension": task["dimension"],
        "sub_label": task["sub_label"],
        "model_id": model["id"],
        "model_name": model["name"],
        "api_model": model["api_model"],
        "run_index": run_idx,
        "timestamp": datetime.now().isoformat(),
        "prompt_hash": hashlib.md5(task["prompt"].encode()).hexdigest()[:8],
    }

    try:
        response = call_model(client, model["api_model"], task["prompt"])
        result.update(response)
        result["status"] = "ok"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["content"] = ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def estimate_cost(tasks: list) -> None:
    """粗略预估调用规模"""
    total_calls = len(tasks) * len(CONFIG["models"]) * CONFIG["runs_per_prompt"]
    already_done = sum(
        1 for t in tasks for m in CONFIG["models"] for r in range(CONFIG["runs_per_prompt"])
        if response_path(t["id"], m["id"], r).exists()
    )

    table = Table(title="调用规模预估")
    table.add_column("项目", style="cyan")
    table.add_column("数值", style="green")
    table.add_row("任务数（含子任务）", str(len(tasks)))
    table.add_row("模型数", str(len(CONFIG["models"])))
    table.add_row("每个 prompt 运行次数", str(CONFIG["runs_per_prompt"]))
    table.add_row("总调用数", str(total_calls))
    table.add_row("已完成（断点续跑）", str(already_done))
    table.add_row("待执行", str(total_calls - already_done))
    console.print(table)


def generate_comparison_report(tasks: list, results: list) -> None:
    """生成对比 Markdown"""
    out_path = Path(CONFIG["output_dir"]) / "comparison.md"

    # 按 task -> model -> [runs] 组织
    grouped = {}
    for r in results:
        key = r["task_id"]
        grouped.setdefault(key, {}).setdefault(r["model_id"], []).append(r)

    lines = []
    lines.append(f"# AI 模型安全场景对比测评")
    lines.append(f"\n> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 每个 prompt 运行次数：{CONFIG['runs_per_prompt']}")
    lines.append(f"> Temperature：{CONFIG['temperature']}\n")

    # 总览表
    lines.append("## 总览\n")
    header = "| 任务 | 维度 | " + " | ".join(
        f"{m['name']} 用时(s) / tokens" for m in CONFIG["models"]
    ) + " |"
    sep = "|" + "---|" * (2 + len(CONFIG["models"]))
    lines.append(header)
    lines.append(sep)

    for task in tasks:
        model_stats = []
        for m in CONFIG["models"]:
            runs = grouped.get(task["id"], {}).get(m["id"], [])
            ok_runs = [r for r in runs if r.get("status") == "ok"]
            if ok_runs:
                avg_time = sum(r["elapsed_seconds"] for r in ok_runs) / len(ok_runs)
                avg_tokens = sum(r["completion_tokens"] for r in ok_runs) / len(ok_runs)
                model_stats.append(f"{avg_time:.1f}s / {int(avg_tokens)}")
            else:
                model_stats.append("❌ 失败")

        task_label = task["id"]
        if task["sub_label"]:
            task_label += f" ({task['sub_label']})"
        lines.append(f"| {task_label} | {task['dimension']} | " + " | ".join(model_stats) + " |")

    lines.append("\n---\n")

    # 每个任务的详细对比
    for task in tasks:
        lines.append(f"## {task['id']}")
        if task["sub_label"]:
            lines.append(f"\n**子测试**：{task['sub_label']}")
        lines.append(f"\n**维度**：{task['dimension']}\n")

        lines.append("### Prompt\n")
        lines.append("```")
        lines.append(task["prompt"].strip())
        lines.append("```\n")

        for m in CONFIG["models"]:
            lines.append(f"### {m['name']} 的回答\n")
            runs = grouped.get(task["id"], {}).get(m["id"], [])
            runs.sort(key=lambda x: x["run_index"])

            for r in runs:
                lines.append(f"#### Run {r['run_index'] + 1}")
                if r.get("status") == "ok":
                    lines.append(f"- 用时：{r['elapsed_seconds']}s")
                    lines.append(f"- Tokens：prompt {r['prompt_tokens']} / completion {r['completion_tokens']}")
                    lines.append(f"- Finish reason：{r.get('finish_reason', 'N/A')}\n")
                    lines.append("<details><summary>点击展开回答</summary>\n")
                    lines.append(r["content"])
                    lines.append("\n</details>\n")
                else:
                    lines.append(f"- ❌ 错误：{r.get('error', 'unknown')}\n")

        lines.append("\n---\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"\n[green]✓ 对比报告已生成：{out_path}[/green]")


def main():
    if CONFIG["api_key"].startswith("sk-你的"):
        console.print("[red]请先在脚本顶部 CONFIG 里填入 api_key[/red]")
        return

    # 初始化
    Path(CONFIG["output_dir"]).mkdir(exist_ok=True)
    (Path(CONFIG["output_dir"]) / "raw").mkdir(exist_ok=True)

    client = OpenAI(
        api_key=CONFIG["api_key"],
        base_url=CONFIG["base_url"],
        default_headers={
            "User-Agent": "curl/8.5.0"  # 关键：覆盖默认的 User-Agent
        }
    )

    tasks = load_tasks(CONFIG["tasks_file"])
    console.print(f"\n[cyan]加载了 {len(tasks)} 个任务（含子任务）[/cyan]")

    estimate_cost(tasks)
    console.print("\n[yellow]按 Enter 开始执行，Ctrl+C 中止（已完成的会自动跳过）[/yellow]")
    input()

    # 生成所有 (task, model, run) 组合
    jobs = []
    for task in tasks:
        for model in CONFIG["models"]:
            for run_idx in range(CONFIG["runs_per_prompt"]):
                jobs.append((task, model, run_idx))

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_bar = progress.add_task("执行中...", total=len(jobs))

        with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as pool:
            futures = {
                pool.submit(run_single, client, t, m, r): (t, m, r)
                for t, m, r in jobs
            }
            for future in as_completed(futures):
                t, m, r_idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "✓" if result.get("status") == "ok" else "✗"
                    progress.console.log(
                        f"{status} {t['id']} | {m['name']} | run {r_idx + 1}"
                    )
                except Exception as e:
                    progress.console.log(f"[red]✗ {t['id']} | {m['name']} | {e}[/red]")
                progress.update(task_bar, advance=1)

    # 汇总所有结果（包括之前断点续跑的）
    all_results = []
    for task in tasks:
        for model in CONFIG["models"]:
            for run_idx in range(CONFIG["runs_per_prompt"]):
                p = response_path(task["id"], model["id"], run_idx)
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        all_results.append(json.load(f))

    generate_comparison_report(tasks, all_results)

    # 统计
    ok_count = sum(1 for r in all_results if r.get("status") == "ok")
    err_count = len(all_results) - ok_count
    total_tokens = sum(r.get("total_tokens", 0) for r in all_results)

    console.print(f"\n[green]完成：{ok_count} 成功 / {err_count} 失败[/green]")
    console.print(f"[cyan]总 tokens：{total_tokens:,}[/cyan]")
    console.print(f"[cyan]原始输出目录：{CONFIG['output_dir']}/raw/[/cyan]")
    console.print(f"[cyan]对比报告：{CONFIG['output_dir']}/comparison.md[/cyan]")


if __name__ == "__main__":
    main()
