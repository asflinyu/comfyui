# S2V 分镜控制台 V2

ComfyUI 上的分镜 / 生图 / I2V / S2V 网页控制台。面向学习与自用部署。

**部署与公网访问（AutoDL 镜像）请先看：[部署说明.md](./部署说明.md)**

## 本地启动

```bash
cd /root/ComfyUI/s2v_console_v2
cp -n .env.example .env   # 填写 LLM Key（可选）
./start.sh
```

- 本机入口：http://127.0.0.1:6010/static/home.html
- AutoDL 公网：自定义服务 **6008**（nginx 反代到 6010）
- ComfyUI：6006
- 配音 TTS：6009（`start.sh` 会自动拉起）

## 入口功能

- 快速版分镜：Excel → 出图 + 720p 视频
- 分镜控制台 V2：可走 LLM 写分镜
- 生图、I2V、F2V
- 数字人：视频数字人、精品 14B 口播（S2V）
