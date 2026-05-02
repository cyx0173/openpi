import jax.profiler

print("📸 正在连接服务器端口 9999 进行抓取...")

# 抓取 2000 毫秒 (2秒) 的数据
# 这 2 秒内，服务器发生的所有 GPU 计算（包括你埋点的 named_scope）都会被录下来
jax.profiler.trace(
    "/tmp/jax_trace_result",  # 保存路径
    duration_ms=10000,         # 录制时长
    profiler_port=8000,       # 连接到 serve_policy.py 开的端口
    create_perfetto_link=False
)

print("✅ 抓取完成！请把 /tmp/jax_trace_result 下载下来用 Perfetto 打开。")