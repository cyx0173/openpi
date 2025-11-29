GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . 
(openpi) chengyuxuan@eva1-chengyuxuan:~/vla/openpi$ python scripts/serve_policy.py --env DROID --port 8000 
Traceback (most recent call last):
  File "/home/chengyuxuan/vla/openpi/scripts/serve_policy.py", line 8, in <module>
    from openpi.policies import policy as _policy
  File "/home/chengyuxuan/vla/openpi/src/openpi/policies/policy.py", line 17, in <module>
    from openpi.models import model as _model
  File "/home/chengyuxuan/vla/openpi/src/openpi/models/model.py", line 20, in <module>
    from openpi.models_pytorch import pi0_pytorch
  File "/home/chengyuxuan/vla/openpi/src/openpi/models_pytorch/pi0_pytorch.py", line 10, in <module>
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
  File "/home/chengyuxuan/vla/openpi/src/openpi/models_pytorch/gemma_pytorch.py", line 3, in <module>
    import pytest
ModuleNotFoundError: No module named 'pytest'
(openpi) chengyuxuan@eva1-chengyuxuan:~/vla/openpi$ pip install pytest pandas

uv run scripts/serve_policy.py --env DROID --port 8000
uv run examples/simple_client/main.py --env DROID --num_steps 20
运行真实的数据集
python examples/simple_client/real_main.py --env DROID --port 8000 --dataset-repo-id your_hf_username/my_droid_dataset 

转发文件：python3 -m http.server 8008 