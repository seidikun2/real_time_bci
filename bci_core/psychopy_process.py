# -*- coding: utf-8 -*-
"""Gerenciamento do experimento PsychoPy como subprocesso do pipeline BCI.

Objetivos:
- não depender de abrir o Builder/Runner manualmente;
- compilar o .psyexp pela própria API/CLI do PsychoPy;
- executar um processo PsychoPy novo a cada bloco;
- permitir encerrar/forçar encerramento do processo se necessário;
- manter o .psyexp como fonte editável do experimento.

A compilação usa o mesmo módulo chamado internamente pelo PsychoPy Builder/Runner:
    python -m psychopy.scripts.psyexpCompile experimento.psyexp -o experimento_autorun.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _section(raw: dict) -> dict[str, Any]:
    value = raw.get("psychopy", {}) or {}
    return value if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "sim", "s"}
    return bool(value)


def _resolve_project_path(value: str | os.PathLike | None, project_dir: Path) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return p if p.is_absolute() else (project_dir / p)


def _python_has_psychopy(executable: Path) -> bool:
    try:
        result = subprocess.run(
            [str(executable), "-c", "import psychopy; print(psychopy.__version__)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        return result.returncode == 0
    except Exception:
        return False


def find_psychopy_python(raw: dict, project_dir: Path) -> Path:
    """Resolve o Python que contém a instalação do PsychoPy.

    Ordem:
    1) psychopy.python_executable no config.yaml;
    2) Python que está rodando main.py, se importar psychopy;
    3) instalações Standalone comuns no Windows;
    4) executáveis encontrados no PATH.
    """
    pcfg = _section(raw)

    explicit = _resolve_project_path(pcfg.get("python_executable"), project_dir)
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(
                f"psychopy.python_executable não existe: {explicit}\n"
                "Aponte para o python.exe da instalação do PsychoPy."
            )
        if not _python_has_psychopy(explicit):
            raise RuntimeError(
                f"O executável configurado não consegue importar psychopy: {explicit}"
            )
        return explicit

    current = Path(sys.executable)
    if _python_has_psychopy(current):
        return current

    candidates: list[Path] = []
    if os.name == "nt":
        env = os.environ
        roots = [
            Path(r"C:\Program Files\PsychoPy"),
            Path(r"C:\Program Files\PsychoPy3"),
        ]
        local = env.get("LOCALAPPDATA")
        if local:
            roots += [
                Path(local) / "PsychoPy",
                Path(local) / "Programs" / "PsychoPy",
            ]
        for root in roots:
            candidates.extend([root / "python.exe", root / "pythonw.exe"])

    for cmd in ("python", "python3"):
        found = shutil.which(cmd)
        if found:
            candidates.append(Path(found))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        if _python_has_psychopy(candidate):
            return candidate

    raise RuntimeError(
        "Não encontrei automaticamente um Python com PsychoPy instalado.\n"
        "No config.yaml, defina por exemplo:\n"
        "psychopy:\n"
        "  python_executable: 'C:/Program Files/PsychoPy/python.exe'"
    )


def patch_generated_script(script_path: Path, subject_id: str, session_id: int, raw: dict) -> None:
    """Ajusta somente o script GERADO, nunca o .psyexp fonte.

    - opcionalmente remove o diálogo inicial;
    - injeta subject/session do config via variáveis de ambiente.

    O uso de variáveis de ambiente mantém o script compilado reutilizável entre blocos.
    """
    pcfg = _section(raw)
    text = script_path.read_text(encoding="utf-8-sig")

    if _as_bool(pcfg.get("inject_session_info"), True):
        participant_re = re.compile(r"(?m)^(\s*)'participant'\s*:\s*.*?,\s*$")
        session_re = re.compile(r"(?m)^(\s*)'session'\s*:\s*.*?,\s*$")

        text, n_part = participant_re.subn(
            lambda m: (
                f"{m.group(1)}'participant': "
                f"os.environ.get(\"BCI_SUBJECT_ID\", {str(subject_id)!r}),"
            ),
            text,
            count=1,
        )
        text, n_sess = session_re.subn(
            lambda m: (
                f"{m.group(1)}'session': "
                f"os.environ.get(\"BCI_SESSION_ID\", {str(session_id)!r}),"
            ),
            text,
            count=1,
        )
        if n_part == 0 or n_sess == 0:
            print(
                "[psychopy] Aviso: não consegui substituir participant/session no script compilado; "
                "o experimento ainda pode rodar com os valores definidos no .psyexp."
            )

    if _as_bool(pcfg.get("hide_info_dialog"), True):
        # Builder 2025.x gera exatamente esta chamada no bloco __main__.
        # A regex tolera espaços e pequenas diferenças de formatação.
        pattern = re.compile(
            r"(?m)^(\s*)expInfo\s*=\s*showExpInfoDlg\(expInfo\s*=\s*expInfo\)\s*$"
        )
        text, n = pattern.subn(r"\1# diálogo inicial suprimido pelo main.py (autorun)", text, count=1)
        if n == 0 and "diálogo inicial suprimido pelo main.py (autorun)" not in text:
            print(
                "[psychopy] Aviso: não encontrei a chamada showExpInfoDlg no script compilado. "
                "Talvez o diálogo já esteja desabilitado no .psyexp."
            )

    script_path.write_text(text, encoding="utf-8")


def compile_psyexp(cfg, raw: dict, project_dir: Path) -> tuple[Path, Path]:
    """Compila o .psyexp para um script Python executável e retorna (python, script)."""
    pcfg = _section(raw)
    python_exe = find_psychopy_python(raw, project_dir)

    experiment_file = _resolve_project_path(
        pcfg.get("experiment_file", "Graz_UpperLimb.psyexp"), project_dir
    )
    if experiment_file is None or not experiment_file.exists():
        raise FileNotFoundError(
            f"Arquivo .psyexp não encontrado: {experiment_file}\n"
            "Ajuste psychopy.experiment_file no config.yaml."
        )

    generated_cfg = pcfg.get("generated_script")
    if generated_cfg:
        generated = _resolve_project_path(generated_cfg, project_dir)
    else:
        generated = experiment_file.with_name(experiment_file.stem + "_autorun.py")
    assert generated is not None

    compile_if_changed = _as_bool(pcfg.get("compile_if_changed"), True)
    must_compile = (
        not generated.exists()
        or not compile_if_changed
        or generated.stat().st_mtime < experiment_file.stat().st_mtime
    )

    if must_compile:
        cmd = [
            str(python_exe),
            "-m",
            "psychopy.scripts.psyexpCompile",
            str(experiment_file),
            "-o",
            str(generated),
        ]
        version = pcfg.get("compile_version")
        if version not in (None, "", "null", "None"):
            cmd += ["-v", str(version)]

        print(f"[psychopy] Compilando {experiment_file.name} -> {generated.name}")
        result = subprocess.run(
            cmd,
            cwd=str(experiment_file.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode != 0 or not generated.exists():
            raise RuntimeError(
                f"Falha ao compilar o .psyexp (exit={result.returncode})."
            )

        patch_generated_script(
            generated,
            subject_id=cfg.experiment.subject_id,
            session_id=cfg.experiment.session_id,
            raw=raw,
        )
    else:
        # Mesmo em cache, garantimos que versões antigas do autorun recebam os ajustes.
        patch_generated_script(
            generated,
            subject_id=cfg.experiment.subject_id,
            session_id=cfg.experiment.session_id,
            raw=raw,
        )
        print(f"[psychopy] .psyexp inalterado; reutilizando {generated.name}")

    return python_exe, generated


def resolve_run_target(cfg, raw: dict, project_dir: Path) -> tuple[Path, Path]:
    """Resolve o script a executar, preferindo o .psyexp compilado automaticamente."""
    pcfg = _section(raw)
    mode = str(pcfg.get("mode", "psyexp")).strip().lower()
    fallback = _resolve_project_path(pcfg.get("fallback_script", "Graz_UpperLimb_lastrun.py"), project_dir)

    if mode not in {"psyexp", "script", "auto"}:
        raise ValueError("psychopy.mode deve ser: psyexp, script ou auto")

    if mode in {"psyexp", "auto"}:
        try:
            return compile_psyexp(cfg, raw, project_dir)
        except Exception as exc:
            if mode == "psyexp" or fallback is None or not fallback.exists():
                raise
            print(f"[psychopy] Compilação automática falhou: {type(exc).__name__}: {exc}")
            print(f"[psychopy] Usando fallback: {fallback}")

    if fallback is None or not fallback.exists():
        raise FileNotFoundError(
            f"Script PsychoPy de fallback não encontrado: {fallback}"
        )
    return find_psychopy_python(raw, project_dir), fallback


def start_psychopy(cfg, raw: dict, project_dir: Path) -> subprocess.Popen | None:
    """Inicia um processo PsychoPy novo para o bloco atual."""
    pcfg = _section(raw)
    if not _as_bool(pcfg.get("enabled"), True):
        print("[psychopy] Autostart desabilitado; aguardando PsychoPy iniciado manualmente.")
        return None

    python_exe, script = resolve_run_target(cfg, raw, project_dir)

    env = os.environ.copy()
    env["BCI_SUBJECT_ID"] = str(cfg.experiment.subject_id)
    env["BCI_SESSION_ID"] = str(cfg.experiment.session_id)
    env["BCI_SESSION_TYPE"] = str(cfg.experiment.session_type)

    cmd = [str(python_exe), str(script)]
    print(f"[psychopy] Iniciando novo processo: {script.name}")

    kwargs: dict[str, Any] = {
        "cwd": str(script.parent),
        "env": env,
    }

    if os.name == "nt":
        flags = 0
        if _as_bool(pcfg.get("new_console"), False):
            flags |= getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        else:
            flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    return subprocess.Popen(cmd, **kwargs)


def stop_psychopy(proc: subprocess.Popen | None, raw: dict) -> None:
    """Espera saída normal; se necessário, termina e por fim mata a árvore do processo."""
    if proc is None or proc.poll() is not None:
        return

    pcfg = _section(raw)
    grace_s = float(pcfg.get("exit_grace_s", 5.0) or 5.0)

    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        print(f"[psychopy] Processo não fechou em {grace_s:.1f}s; solicitando encerramento.")

    try:
        proc.terminate()
        proc.wait(timeout=2.0)
        return
    except Exception:
        pass

    print("[psychopy] Encerramento normal falhou; forçando término do processo.")
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def timeout_value(raw: dict, key: str, default: float | None) -> float | None:
    value = _section(raw).get(key, default)
    if value in (None, "", "null", "None", False):
        return None
    return float(value)
