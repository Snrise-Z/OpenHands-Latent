"""UDockerRuntime: OpenHands 0.62.0 runtime backed by udocker (rootless, no dockerd).

Runs each shell command inside a udocker container created from
sandbox_config.base_container_image (the official SWE-bench instance image),
and performs file operations directly on the container rootfs via path
translation. Designed for the SWE-bench eval harness on hosts where the
docker daemon is unavailable (e.g. rented GPU containers).

Storage is split on purpose: image layers/repos are read from the shared
disk (already holds all 500 Verified images), while extracted containers go
to the local data disk. The shared disk has a 200k inode hard cap and a
single extracted container costs tens of thousands of inodes.

State persistence: env exports and cwd are persisted between commands via
state files inside the container, because `bash -c` sessions do not share
env.

Select with: RUNTIME=udocker_runtime.UDockerRuntime (module must be on
PYTHONPATH).
"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from openhands.core.logger import openhands_logger as logger
from openhands.events.observation import CmdOutputObservation
from openhands.runtime.impl.cli.cli_runtime import CLIRuntime

# 注意:不要用 UDOCKER_BIN 这个名字 —— udocker 自己把它读作「工具目录」
# (proot/patchelf 所在处),指成可执行文件会让 F3 模式找不到 patchelf。
UDOCKER_EXE = os.environ.get('UDOCKER_EXE', '/usr/local/bin/udocker')
# topdir holds config + containers (local disk); layers/repos live on the
# shared disk where the pre-pulled Verified images already are.
UDOCKER_DIR = os.environ.get('UDOCKER_DIR', '/root/autodl-tmp/.udocker')
UDOCKER_STORE = os.environ.get('UDOCKER_STORE', '/root/autodl-fs/udocker_store')
UDOCKER_REPOS = os.environ.get('UDOCKER_REPOS', f'{UDOCKER_STORE}/repos')
UDOCKER_LAYERS = os.environ.get('UDOCKER_LAYERS', f'{UDOCKER_STORE}/layers')
UDOCKER_CONTAINERS = os.environ.get('UDOCKER_CONTAINERS', f'{UDOCKER_DIR}/containers')
UDOCKER_EXECMODE = os.environ.get('UDOCKER_EXECMODE', 'F3')
# Container templates: extract (create + setup) each image ONCE into
# UDOCKER_TEMPLATE_DIR/<image>/src, then clone per task with `cp -a
# --reflink` (XFS/btrfs reflink: metadata-only, no data copy). Container
# creation is otherwise a 2.5 GB write per task, which saturates the local
# RAID with 60+ concurrent tasks (18 % iowait, 15-20 min startup stalls).
# Must live on the same filesystem as UDOCKER_CONTAINERS. Empty = disabled.
UDOCKER_TEMPLATE_DIR = os.environ.get('UDOCKER_TEMPLATE_DIR', '')
# proot (P1/P2 fallbacks and some udocker internals) requires an
# exec-permitted temp dir; the default /dev/shm mount here is noexec.
UDOCKER_TMP = os.environ.get('UDOCKER_TMP', '/root/autodl-tmp/.proot-tmp')
UDOCKER_REGISTRIES = [
    r for r in os.environ.get(
        'UDOCKER_REGISTRIES', 'https://docker.xuanyuan.run,https://registry-1.docker.io'
    ).split(',') if r
]
# udocker shares the host network namespace, so a host-local proxy is
# reachable from inside the container at the same address.
CONTAINER_PROXY = os.environ.get('CONTAINER_PROXY', 'http://127.0.0.1:7890')
NO_PROXY = os.environ.get('CONTAINER_NO_PROXY', 'localhost,127.0.0.1,::1')


def _udocker_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k != 'UDOCKER_BIN'}
    return {
        **env,
        'UDOCKER_DIR': UDOCKER_DIR,
        'UDOCKER_REPOS': UDOCKER_REPOS,
        'UDOCKER_LAYERS': UDOCKER_LAYERS,
        'UDOCKER_CONTAINERS': UDOCKER_CONTAINERS,
        'UDOCKER_TMP': UDOCKER_TMP,
        'TMPDIR': UDOCKER_TMP,
        'PROOT_TMP_DIR': UDOCKER_TMP,
    }


def _udocker(*args: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    os.makedirs(UDOCKER_TMP, exist_ok=True)
    os.makedirs(UDOCKER_CONTAINERS, exist_ok=True)
    return subprocess.run(
        [UDOCKER_EXE, '--allow-root', *args],
        capture_output=True, text=True, timeout=timeout, env=_udocker_env(),
    )


def _template_key(image_ref: str) -> str:
    return image_ref.replace('/', '__').replace(':', '__')


def _ensure_template(image_ref: str) -> Path | None:
    """Return the template dir for image_ref (building it under a lock if
    needed), or None if templates are disabled / build failed."""
    if not UDOCKER_TEMPLATE_DIR:
        return None
    base = Path(UDOCKER_TEMPLATE_DIR)
    base.mkdir(parents=True, exist_ok=True)
    tdir = base / _template_key(image_ref)
    ready = tdir / '.ready'
    if ready.exists() and (tdir / 'src' / 'ROOT').exists():
        return tdir
    lock_path = base / f'.{_template_key(image_ref)}.lock'
    with open(lock_path, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if ready.exists() and (tdir / 'src' / 'ROOT').exists():
                return tdir
            tname = f'tmpl_{uuid.uuid4().hex[:12]}'
            out = _udocker('create', f'--name={tname}', image_ref)
            if out.returncode != 0 or 'Error' in out.stderr:
                logger.warning(f'[udocker] template create failed for {image_ref}: {out.stderr[-200:]}')
                return None
            st = _udocker('setup', '--force', f'--execmode={UDOCKER_EXECMODE}', tname)
            link = Path(UDOCKER_CONTAINERS) / tname
            cid_dir = link.resolve() if link.is_symlink() else link
            if not (cid_dir / 'ROOT').exists():
                logger.warning(f'[udocker] template rootfs missing for {image_ref}: {st.stderr[-200:]}')
                return None
            if tdir.exists():
                shutil.rmtree(tdir, ignore_errors=True)
            tdir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(cid_dir), str(tdir / 'src'))     # same filesystem: rename
            if link.is_symlink():
                link.unlink()
            ready.write_text(image_ref)
            logger.info(f'[udocker] built template for {image_ref} at {tdir}')
            return tdir
        except Exception as e:
            logger.warning(f'[udocker] template build error for {image_ref}: {e}')
            return None
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _clone_from_template(tdir: Path, name: str) -> bool:
    """cp -a --reflink the template into a fresh container id and register the name symlink."""
    new_id = str(uuid.uuid4())
    dst = Path(UDOCKER_CONTAINERS) / new_id
    r = subprocess.run(['cp', '-a', '--reflink=always', str(tdir / 'src'), str(dst)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        shutil.rmtree(dst, ignore_errors=True)
        logger.warning(f'[udocker] reflink clone failed: {r.stderr[-200:]}')
        return False
    link = Path(UDOCKER_CONTAINERS) / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(new_id)   # udocker convention: containers/<name> -> <id> (relative)
    return True


class UDockerRuntime(CLIRuntime):
    """CLIRuntime whose shell commands execute inside a udocker container."""

    def __init__(self, config, event_stream, *args, sid='default', **kwargs):
        self._image = config.sandbox.base_container_image
        self._container_name = f'oh_{sid}'.replace('-', '_')[:60]
        self._container_root: Path | None = None
        self._current_dir = '/'
        super().__init__(config, event_stream, *args, sid=sid, **kwargs)

    # -- container lifecycle ------------------------------------------------

    async def connect(self):
        self._ensure_container()
        await super().connect()
        # CLIRuntime pins its workspace under a host temp dir; commands run in
        # the container instead. SWE-bench images check out the repo at
        # /testbed, which is where the agent should start.
        start_dir = os.environ.get('UDOCKER_START_DIR', '/testbed')
        self._current_dir = start_dir
        try:
            cwd_host = Path(self._sandbox_to_host('/tmp/.oh_cwd'))
            cwd_host.parent.mkdir(parents=True, exist_ok=True)
            cwd_host.write_text(start_dir)
        except OSError as e:
            logger.warning(f'[udocker] could not seed cwd file: {e}')

    def _pull_if_missing(self, image_ref: str):
        if image_ref in _udocker('images').stdout:
            return
        last = None
        for attempt in range(4):
            reg = UDOCKER_REGISTRIES[attempt % len(UDOCKER_REGISTRIES)]
            logger.info(f'[udocker] pulling {image_ref} via {reg} '
                        f'(attempt {attempt + 1})')
            last = _udocker('pull', f'--registry={reg}', image_ref, timeout=7200)
            if image_ref in _udocker('images').stdout:
                return
            time.sleep(15 * (attempt + 1))
        raise RuntimeError(f'udocker pull failed: {(last.stderr if last else "")[-500:]}')

    def _ensure_container(self):
        image = self._image
        assert image, 'sandbox.base_container_image must be set'
        image_ref = image.removeprefix('docker.io/')
        self._pull_if_missing(image_ref)
        ps = _udocker('ps')
        if self._container_name not in ps.stdout:
            tdir = _ensure_template(image_ref)
            cloned = bool(tdir) and _clone_from_template(tdir, self._container_name)
            if cloned:
                logger.info(f'[udocker] cloned container {self._container_name} from template (reflink)')
                out = subprocess.CompletedProcess(args=[], returncode=0, stdout='', stderr='')
            else:
                out = _udocker('create', f'--name={self._container_name}', image_ref)
            if out.returncode != 0 or 'Error' in out.stderr:
                logger.warning(f'[udocker] create failed for {image_ref}, '
                               f're-pulling: {out.stderr[-200:]}')
                _udocker('rmi', image_ref, timeout=600)
                self._pull_if_missing(image_ref)
                out = _udocker('create', f'--name={self._container_name}', image_ref)
            if out.returncode != 0:
                raise RuntimeError(f'udocker create failed: {out.stderr[-500:]}')
        # ALWAYS enforce execmode (idempotent): a container left at udocker's
        # default P1 without an exec-permitted TMPDIR fails every command
        # instantly, which the agent experiences as a broken environment.
        setup = _udocker('setup', '--force', f'--execmode={UDOCKER_EXECMODE}',
                         self._container_name)
        mode_file = Path(UDOCKER_CONTAINERS, self._container_name, 'execmode')
        by_name = Path(UDOCKER_CONTAINERS, self._container_name, 'ROOT')
        if by_name.exists():
            self._container_root = by_name
        else:
            for c in Path(UDOCKER_CONTAINERS).glob('*'):
                nf = c / 'container.name'
                if nf.exists() and self._container_name in nf.read_text():
                    self._container_root = c / 'ROOT'
                    mode_file = c / 'execmode'
                    break
        if self._container_root is None or not self._container_root.exists():
            raise RuntimeError(f'cannot resolve rootfs for {self._container_name}')
        if not mode_file.exists() or UDOCKER_EXECMODE not in mode_file.read_text():
            raise RuntimeError(
                f'udocker setup --execmode={UDOCKER_EXECMODE} did not stick '
                f'for {self._container_name}: {setup.stderr[-200:]}')
        logger.info(f'[udocker] container {self._container_name} root={self._container_root}')

    def close(self):
        super().close()
        # Each container ROOT is several GB and tens of thousands of inodes;
        # a 500-instance sweep would fill the disk. Images (layers) are kept
        # for reuse; only the extracted container is removed.
        if os.environ.get('UDOCKER_KEEP_CONTAINERS', '').lower() not in ('1', 'true'):
            try:
                _udocker('rm', self._container_name, timeout=900)
                logger.info(f'[udocker] removed container {self._container_name}')
            except Exception as e:
                logger.warning(f'[udocker] container cleanup failed: {e}')

    # -- path translation for file ops -------------------------------------

    def _sandbox_to_host(self, path: str) -> str:
        p = str(path)
        if self._container_root is None:
            return p
        if p.startswith(str(self._container_root)):
            return p
        if os.path.isabs(p):
            return str(self._container_root) + p
        return p

    def _sanitize_filename(self, filename: str) -> str:
        # CLIRuntime restricts file ops to its workspace; the agent operates
        # on container paths, so translate instead of restricting. Relative
        # paths resolve against the shell's persisted container cwd.
        p = str(filename)
        if not os.path.isabs(p):
            p = os.path.join(self._current_dir or '/', p)
        return self._sandbox_to_host(p)

    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        import shutil
        dest = Path(self._sandbox_to_host(sandbox_dest))
        dest.mkdir(parents=True, exist_ok=True)
        src = Path(host_src)
        if src.is_dir() or recursive:
            shutil.copytree(src, dest / src.name, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest / src.name)

    def copy_from(self, path: str):
        import tempfile
        import zipfile
        host_path = Path(self._sandbox_to_host(path))
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            zip_path = f.name
        with zipfile.ZipFile(zip_path, 'w') as zf:
            if host_path.is_dir():
                for sub in host_path.rglob('*'):
                    if sub.is_file():
                        zf.write(sub, sub.relative_to(host_path))
            elif host_path.is_file():
                zf.write(host_path, host_path.name)
        return Path(zip_path)

    # -- command execution ---------------------------------------------------

    def _execute_shell_command(self, command: str, timeout: float) -> CmdOutputObservation:
        state = '/tmp/.oh_env_state'
        cwd_file = '/tmp/.oh_cwd'
        # Persist env+cwd across bash -c invocations via files inside the
        # container rootfs (which persists between udocker runs).
        wrapped = (
            f'[ -f {state} ] && source {state} >/dev/null 2>&1; '
            f'[ -f {cwd_file} ] && cd "$(cat {cwd_file})" 2>/dev/null; '
            f'{command}\n'
            f'__oh_rc=$?; export -p > {state} 2>/dev/null; '
            f'pwd > {cwd_file}; exit $__oh_rc'
        )
        envs = [
            f'--env=SWE_INSTANCE_ID={os.environ.get("SWE_INSTANCE_ID", "")}',
        ]
        if CONTAINER_PROXY:
            for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
                envs.append(f'--env={k}={CONTAINER_PROXY}')
            envs.append(f'--env=no_proxy={NO_PROXY}')
            envs.append(f'--env=NO_PROXY={NO_PROXY}')
        full_cmd = [UDOCKER_EXE, '--allow-root', 'run', *envs,
                    self._container_name, '/bin/bash', '-c', wrapped]
        start = time.monotonic()
        try:
            # Merge stderr into stdout: tracebacks, pytest failures, and
            # compiler errors all arrive on stderr, and an agent that cannot
            # see them observes a silent empty result for every failure.
            proc = subprocess.run(
                full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=timeout if timeout else 3600,
                env=_udocker_env(), start_new_session=True,
            )
            raw = proc.stdout
            exit_code = proc.returncode
        except subprocess.TimeoutExpired as e:
            raw = (e.stdout or b'').decode(errors='replace') if isinstance(e.stdout, bytes) else (e.stdout or '')
            exit_code = -1
        # Strip udocker's banner: box lines ('# ... #', '* ... *'), the
        # 'STARTING <id>' line inside the box, and 'executing:' trailer.
        # With stderr merged, udocker/proot's own diagnostics now land in the
        # stream too and must not be mistaken for program output.
        def _is_banner(ln: str) -> bool:
            s = ln.strip()
            if not s:
                return False
            if s.startswith(('#', '*')) and s.endswith(('#', '*')):
                return True
            return s.startswith(('executing:', 'proot info:', 'proot error:',
                                 'Info: ', 'Warning: forcing',
                                 'fatal error: see'))

        lines = [ln for ln in raw.split('\n') if not _is_banner(ln)]
        content = '\n'.join(lines).strip('\n')
        cwd = self._current_dir or '/'
        try:
            cwd_host = Path(self._sandbox_to_host(cwd_file))
            if cwd_host.exists():
                cwd = cwd_host.read_text().strip() or cwd
        except OSError:
            pass
        self._current_dir = cwd
        logger.debug(f'[udocker] cmd rc={exit_code} wall={time.monotonic()-start:.1f}s: {command[:120]}')
        return CmdOutputObservation(
            command=command, content=content, exit_code=exit_code,
            metadata={'working_dir': cwd},
        )
