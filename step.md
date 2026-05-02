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

python scripts/serve_policy.py --env DROID --port 8000
uv run examples/simple_client/main.py --env DROID --num_steps 20
运行真实的数据集
python examples/simple_client/real_main.py --env DROID --port 8000 --dataset-repo-id your_hf_username/my_droid_dataset 
运行pytorch版本
uv run scripts/serve_policy.py policy:checkpoint     --policy.config=pi05_droid     --policy.dir=checkpoints/pi05_droid_pytorch 
转发文件：python3 -m http.server 8008 

现在可以梳理清楚这个
python scripts/serve_policy.py --env DROID --port 8000
uv run examples/simple_client/main.py --env DROID --num_steps 20
是在做什么了 1：serve_policy.py的main构建的是一个policy和神经网络的组建 2：examples/simple_client/main.py是通过client来调用这个serve_policy.py里面的policy
步骤是首先 
1）传入的env是 DROID 但是没有传入policy 所以是create_policy -> case Default(): ->
return create_default_policy(args.env, default_prompt=args.default_prompt)
2）在这个内部 调用了
def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
3）_config.get_config 是进入到 src/openpi/training/config.py 
def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
这个获取的是_CONFIGS_DICT里面的TrainConfig 从而获得是参数配置
 TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    )action_horizon=15
action_horizon=15这个意思就是一次预测 未来十五步的动作 所以最后的action.shape = (15,8)

4）create_trained_policy 将所有的参数配置丢进去 我们开始来组建diffusion policy 
  return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )

这里便是定义了一个完整的系统策略，其中这个model 是中间那些神经网络的节点层 然后transforms和output_transforms其实定义的是输入和输出的数据流 
1：output_transforms=[...] 定义了如何把那堆小数变回机器人能执行的指令。
模型输出：那张 (15, 8) 的归一化 Tensor。
Unnormalize (反归一化)：
这是列表里的核心步骤。
代码会拿出之前的“尺子”（norm_stats），执行公式：真实值 = 模型输出 * 方差 + 均值。
例如：模型输出 0.5 -> 乘方差加均值 -> 变回 1.57 (即 90度)。
最终输出：
一个包含真实物理单位（弧度、开合度）的动作指令，发给 ALOHA 或 DROID 机器人的底层控制器执行。|

2：transforms=[...] 这一串列表，定义了数据如何一步步变成 Tensor。
原始输入：
一张图片（比如 224x224 的 RGB 像素矩阵）。
一段文字（比如 "Put the apple in the bowl"）。
当前的机器人状态（比如当前手臂在哪里）。
中间处理 (data_transforms / model_transforms)：
图片：不是简单的转灰度，通常是 Resize（缩放） -> ToTensor（转浮点数） -> Patchify/Tokenize（切成小块，变成一个个 Token 向量）。
文字：通过 Tokenizer 变成整数列表（Token IDs），再变成 Embedding 向量。
状态：Normalize（这一步最重要）。把真实的电机角度（例如 0.5弧度, 1.2弧度）通过减去均值、除以方差，变成类似 -0.1, 0.3 这样的标准正态分布数值。
最终给模型的：
就是一堆 Tensor（张量）。模型根本不知道这些数字原先是图片还是文字，它只负责算数。
所以这里其实是实例化了Policy这个类 然后我们可以调用Policy.infer来处理输入