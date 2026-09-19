#!/bin/bash
# 启动 S2V 分镜控制台 V2（默认端口 6010）
# 本脚本会自动拉起本地 Qwen3-TTS 服务（6009），用户只需运行一次即可。
# AutoDL 公网入口是 6008（nginx → 6010），不要把 TTS 再绑回 6008。
cd "$(dirname "$0")"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "已创建 .env，请填写 DEEPSEEK_API_KEY 后重新运行"
  exit 1
fi

# 让 shell 也能读取 .env 变量
set -a
source .env
set +a

export PYTHONUNBUFFERED=1

PORT="${CONSOLE_PORT:-6010}"
TTS_URL="${TTS_URL:-http://127.0.0.1:6009}"

# 从 TTS_URL 提取 host/port，兼容 .env 里任意格式
TTS_HOST=$(python3 -c "from urllib.parse import urlparse; print(urlparse('$TTS_URL').hostname or '127.0.0.1')")
TTS_PORT=$(python3 -c "from urllib.parse import urlparse; print(urlparse('$TTS_URL').port or 6009)")

# 检查地址是否为本机（只在本机时才自动启动 TTS）
is_local_tts() {
  [[ "$TTS_HOST" == "127.0.0.1" || "$TTS_HOST" == "localhost" || "$TTS_HOST" == "0.0.0.0" || "$TTS_HOST" == "::1" ]]
}

# 通用：根据端口号释放被占用的端口（基于 /proc/net/tcp 查找 socket inode -> pid）
release_port_by_num() {
  local target_port="$1"
  local port_hex inode pids pid fd

  port_hex=$(printf '%04X' "$target_port")
  inode=$(awk -v p="$port_hex" '$2 ~ ":"p"$" {print $10; exit}' /proc/net/tcp)
  if [[ -z "$inode" ]]; then
    return 0
  fi

  pids=""
  for fd in /proc/[0-9]*/fd/[0-9]*; do
    if [[ -L "$fd" ]] && [[ "$(readlink "$fd" 2>/dev/null)" == "socket:[$inode]" ]]; then
      pid=$(echo "$fd" | cut -d'/' -f3)
      if [[ "$pid" != "$$" ]] && [[ "$pid" != "$PPID" ]]; then
        pids="$pids $pid"
      fi
    fi
  done
  pids=$(echo "$pids" | tr ' ' '\n' | awk 'NF && !seen[$0]++' | tr '\n' ' ')

  if [[ -n "$pids" ]]; then
    echo "端口 $target_port 已被进程占用: $pids，正在释放..."
    kill $pids 2>/dev/null || true
    sleep 2
    local still_alive=""
    for pid in $pids; do
      if kill -0 "$pid" 2>/dev/null; then
        still_alive="$still_alive $pid"
      fi
    done
    if [[ -n "$still_alive" ]]; then
      echo "进程未退出，正在强制结束..."
      kill -9 $still_alive 2>/dev/null || true
      sleep 1
    fi
  fi
}

# 释放控制台端口（保持原有逻辑）
release_console_port() {
  release_port_by_num "$PORT"
}

# 检测 TTS 是否可用
tts_health() {
  curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$TTS_URL/" 2>/dev/null
}

# 自动拉起本地 TTS 服务
start_local_tts() {
  local tts_dir="/root/ComfyUI/ttsvoice"
  if [[ ! -f "$tts_dir/start_autodl.sh" ]]; then
    echo "未找到 TTS 启动脚本 $tts_dir/start_autodl.sh，跳过自动启动 TTS。"
    return 1
  fi

  echo "TTS 服务未响应，自动启动中...（端口 $TTS_PORT）"
  release_port_by_num "$TTS_PORT"

  cd "$tts_dir"
  nohup ./start_autodl.sh > /tmp/s2v_tts.log 2>&1 &
  local tts_pid=$!
  cd - >/dev/null

  echo "TTS 进程已启动 pid=$tts_pid，等待就绪..."
  local waited=0
  while [[ $waited -lt 180 ]]; do
    sleep 3
    if [[ "$(tts_health)" == "200" ]]; then
      echo "TTS 服务已就绪：$TTS_URL"
      return 0
    fi
    ((waited += 3))
    echo "  等待 TTS 就绪... ${waited}s"
  done
  echo "TTS 服务在 180 秒内未就绪，请检查 /tmp/s2v_tts.log"
  return 1
}

# 如果配置的是本机 TTS，确保它在运行
# 避免 6008 上的旧 TTS 和 6009 双开占显存（6008 留给 nginx 公网入口）
if [[ "$TTS_PORT" != "6008" ]] && ss -lptn 2>/dev/null | grep -q ':6008' && ! ss -lptn | grep ':6008' | grep -q nginx; then
  echo "旧 TTS 仍占用 6008，先释放以免双开..."
  release_port_by_num 6008
fi

if is_local_tts; then
  if [[ "$(tts_health)" != "200" ]]; then
    start_local_tts
  else
    echo "TTS 服务已运行：$TTS_URL"
  fi
fi

# AutoDL 公网入口：6008 → 本机 6010。TTS 必须在 6009，不要占用 6008。
setup_public_nginx() {
  local conf="/root/ComfyUI/s2v_console_v2/nginx-console.conf"
  if ! command -v nginx >/dev/null 2>&1; then
    echo "未安装 nginx，跳过公网反代（控制台仍走 :$PORT）"
    return 0
  fi
  mkdir -p /etc/nginx/sites-available /etc/nginx/sites-enabled
  ln -sfn "$conf" /etc/nginx/sites-available/s2v-console
  ln -sfn /etc/nginx/sites-available/s2v-console /etc/nginx/sites-enabled/s2v-console
  rm -f /etc/nginx/sites-enabled/default
  # 6008 专给 nginx；若旧 TTS 仍占着则释放（TTS 已迁到 6009）
  if ! ss -lptn 2>/dev/null | grep ':6008' | grep -q nginx; then
    echo "释放 6008 给公网 nginx..."
    release_port_by_num 6008
  fi
  if nginx -t >/tmp/s2v_nginx_test.log 2>&1; then
    nginx -s reload 2>/dev/null || nginx
    echo "公网入口已就绪：nginx :6008 → 127.0.0.1:$PORT"
  else
    echo "nginx 配置检查失败，见 /tmp/s2v_nginx_test.log"
  fi
}

setup_public_nginx
release_console_port
exec python3 -m uvicorn app:app --host "${CONSOLE_HOST:-0.0.0.0}" --port "$PORT"
