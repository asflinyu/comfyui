#!/bin/bash
cd /root/ComfyUI/ttsvoice

PORT=6009

kill_port_occupier() {
    # 1. find inode of the listening socket on $PORT
    local inode
    inode=$(awk -v port="$PORT" '
        function hex2dec(h){ r=0; n=length(h); for(i=1;i<=n;i++){ c=substr(h,i,1); r=r*16; if(c>="0"&&c<="9") r+=c-0; else if(c>="a"&&c<="f") r+=c-87; else if(c>="A"&&c<="F") r+=c-55 } return r }
        $2 ~ /:[0-9A-Fa-f]+$/ { split($2,a,":"); if(hex2dec(a[2])==port && $4=="0A") print $10 }
    ' /proc/net/tcp 2>/dev/null | head -1)

    if [ -z "$inode" ]; then
        return
    fi

    # 2. map inode -> pid(s) via /proc/*/fd (fd link is at depth 3)
    local pids
    pids=$(find /proc -maxdepth 3 -type l -path '/proc/[0-9]*/fd/[0-9]*' 2>/dev/null \
        | while read fd; do
            if readlink "$fd" 2>/dev/null | grep -qE "^socket:\[$inode\]$"; then
                echo "$(echo "$fd" | cut -d'/' -f3)"
            fi
        done | sort -u | tr '\n' ' ')

    if [ -n "$pids" ]; then
        echo "[start] port $PORT already in use by pid(s): $pids, killing..."
        kill -9 $pids 2>/dev/null || true
        sleep 2
    fi
}

kill_port_occupier

export QWEN_TTS_CKPT=/root/ComfyUI/ttsvoice/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
export QWEN_TTS_IP=0.0.0.0
export QWEN_TTS_PORT=$PORT
exec .venv/bin/python studio.py
