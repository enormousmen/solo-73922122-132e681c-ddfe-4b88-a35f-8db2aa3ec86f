# Resumable Training API（可恢复单作业训练服务）

FastAPI + CPU 版 PyTorch 实现的**单作业训练服务**。训练工程师在作业被抢占（暂停、
进程强退、容器被杀）后，可以从最近成功的检查点继续训练；在**同环境、同种子、同
配置**下，恢复后的逐批损失与最终参数与**不中断训练逐位一致**。

- 模型：两层全连接网络 `Linear → ReLU → Dropout → Linear`
- 优化器：Adam；每轮（epoch）打乱全部样本后按批训练
- 数据：仓库内置固定分类数据集 `data/classification.csv`（600 样本 / 2 特征 / 3 类）
- 默认配置：`config/default.json`（可在启动时按字段覆盖）

## 目录结构

```
app/
  main.py        # FastAPI 入口（/docs 自动生成）
  trainer.py     # 单作业状态机 + 确定性训练循环 + 暂停/续训
  checkpoint.py  # 原子检查点读写（tmp + fsync + rename）
  config.py      # 默认配置加载与校验
  dataset.py     # 固定 CSV 数据集加载
  model.py       # 两层 FC + Dropout
config/default.json      # 默认训练配置
data/classification.csv  # 固定分类数据集（scripts/generate_data.py 生成）
scripts/generate_data.py # 数据集生成脚本（固定种子，仅作溯源）
scripts/verify_repro.py  # 端到端验证：不中断对照 / 暂停续训 / 强退恢复 等
Dockerfile, docker-compose.yml, requirements.txt
```

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 服务: http://localhost:8000  交互式文档: http://localhost:8000/docs
```

检查点保存在命名卷 `trainer-runs` 中，容器被杀或重启后数据不丢。

### 本地运行（开发）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 环境变量: RUNS_DIR(默认 runs/)  CONFIG_PATH(默认 config/default.json)
```

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务信息 |
| GET | `/config` | 默认配置与运行时信息 |
| POST | `/train/start` | 启动新训练。Body: `{"force": false, "config": {覆盖字段}}` |
| POST | `/train/pause` | 暂停。**训练停止且完整检查点持久化后才返回 200** |
| POST | `/train/resume` | 从暂停态/最近检查点续训（并发只放行一个训练者） |
| GET | `/train/status` | 状态与进度（epoch/batch/百分比/检查点健康度） |
| GET | `/train/losses` | 逐批损失（支持 `since_step`、`limit`） |
| GET | `/train/parameters` | 下载当前模型参数（.pt，取自最近持久化检查点） |
| GET | `/train/checkpoint` | 下载完整检查点（模型+优化器+RNG+历史+数据身份） |
| GET | `/train/archived` | 被强制新开归档的旧检查点列表（含原因与文件路径） |
| GET | `/dataset/status` | 当前绑定数据身份 vs 磁盘 CSV 的比对结果 |

交互式文档（Swagger UI）在 `/docs`，可直接在浏览器里调通上述接口。

### curl 示例

```bash
# 启动（覆盖部分配置；已有运行需 force=true 或先续训）
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 8, "batch_size": 32, "lr": 0.01}}'

# 查询进度
curl -s localhost:8000/train/status | jq '.state, .completed_batches, .total_batches'

# 逐批损失
curl -s 'localhost:8000/train/losses?limit=5' | jq '.losses[]'

# 暂停（阻塞至训练停止且检查点落盘才返回）
curl -X POST localhost:8000/train/pause

# 续训
curl -X POST localhost:8000/train/resume

# 下载参数 / 完整检查点
curl -OJ localhost:8000/train/parameters     # model_parameters.pt
curl -OJ localhost:8000/train/checkpoint     # checkpoint.pt
```

状态机：`idle → running ⇄ pausing → paused → running … → finished`（异常为
`failed`）。`paused` 表示训练已停、可续训；`checkpoint_dirty=true` 表示内存进度
领先于磁盘（上次保存失败），此时重启进程会回退到磁盘上的旧检查点。

## 检查点与训练数据绑定（换数据后不会静默续训）

**问题**：训练工程师把默认 CSV 换成新数据后重启服务，旧检查点仍会“续训”，但
接着算的是新样本——损失历史与参数不再对应同一份训练数据。

**修复**：每个检查点显式记录它所训练样本的身份，续训前重新核对：

- 身份 = `sha256`（特征列名 + 行数/列数 + 按**文件行序**的 float32 特征 +
  int64 标签）。因此**数值变化**或**行序变化**都会改变身份；仅 CSV 排版变化
  （如 `1.0` vs `1.00`）不改变身份。算法常量：`sha256:canonical-float32-rows-v1`。
- 新运行启动时把 CSV **一次性读入内存**并记录身份，训练线程只用这份内存快照。
  **运行中修改 CSV 不会改变已加载数据**，之后写出的检查点仍记录原身份。
- 重启后比较“检查点身份”与“当前路径下的 CSV”：
  - 完全一致 → 正常 `paused`，`POST /train/resume` 可续训（同环境逐位一致）；
  - 数值变化/行序变化 → `dataset_changed`，续训返回 **409**；
  - 原文件丢失 → `dataset_unavailable`，续训 409；
  - 旧格式检查点（无数据身份）→ `legacy_checkpoint`，**同样拒绝续训**；
  - 检查点文件损坏 → `checkpoint_unreadable`，续训 409。
- 拒绝时：**旧检查点原样保留**在 `runs/checkpoint.pt`，不训练、不覆盖；原因可查询：
  `GET /train/status` 的 `data_conflict` / `checkpoint_issue` / `dataset` 字段，
  以及 `GET /dataset/status`。
- **恢复原数据并重启**服务后（swap 回原 CSV，`docker compose restart trainer`），
  身份重新一致，`resume` 继续成功且与不中断训练逐位一致。
- 确实要换数据：`POST /train/start {"force": true}`。会先把旧检查点**移动**到
  `runs/archive/checkpoint.<run>.<时间戳>.pt` 并写一份 JSON 清单（原因、原数据
  身份、进度），**不会删除**；随后用**当前默认 CSV** 新开训练。归档可用
  `GET /train/archived` 查询。若当前 CSV 不可读，force 新开同样返回 400，
  旧检查点不动。
- 非 force 的 `POST /train/start` 在存在不可续训的旧作业时返回 409 并给出原因，
  避免“以为新开、实际覆盖旧点”。

`GET /train/status` 里相关字段示例（换了 CSV 并重启后）：

```json
{
  "state": "paused",
  "ready_to_resume": false,
  "dataset": {
    "state": "changed",
    "bound":   {"fingerprint": "sha256:48f47645cae10cb7..."},
    "current": {"fingerprint": "sha256:42ae2fbf05084679..."},
    "matches_bound": false
  },
  "data_conflict": {
    "type": "dataset_changed",
    "reason": "the CSV at the configured path does not match the samples this checkpoint was trained on (contents or row order changed).",
    "hint": "Restore the original CSV ... or POST /train/start with {\"force\": true} ...",
    "old_checkpoint_kept": true
  }
}
```

`dataset.state` 取值：`current`（一致）、`changed`、`missing`、`invalid`（存在但
解析失败）、`unverified`（旧格式检查点无身份）、`unbound`（从未训练）。

## 更换数据后的演示

### 自动化（本地/CI，推荐）

```bash
python3 scripts/demo_dataset_change.py
```

用临时目录与真实 uvicorn 进程完整演示（不改动仓库文件）：

1. 在 CSV 副本上训练并暂停；
2. 把 CSV 数值换成新数据并重启 → `dataset_changed`、`resume` 409、旧
   `checkpoint.pt` 仍在、归档列表为空；
3. `force=true` 新开 → 旧点归档（带原因）→ 新运行在当前数据上跑完；
4. 换回原数据、重启 → `resume` 200 并跑完。

### Docker Compose 手动演示

```bash
docker compose up -d --build
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 4, "batch_sleep_seconds": 0.2}}'
sleep 2 && curl -X POST localhost:8000/train/pause

# 换数据：修改宿主机上的 ./data/classification.csv（compose 以 bind mount 挂载）
# 例如把 feature_1 全部乘以 0.5，然后重启服务
docker compose restart trainer
curl -s localhost:8000/train/status | jq '{state, ready_to_resume,
  ds: .dataset.state, why: .data_conflict.type}'
#   -> paused, false, "changed", "dataset_changed"
curl -i -X POST localhost:8000/train/resume      # -> 409，原因在 detail
ls -l data/                                          # 原文件仍可换回

# 强制新开（旧检查点归档、不删除）
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"force": true, "config": {"epochs": 2}}'
curl -s localhost:8000/train/archived | jq '.archived[0] | {reason_type, run_id}'

# 恢复原数据后仍可继续原作业：换回 CSV、重启、resume
```

> compose 挂载了 `./data` 与 `./config`，因此换 CSV / 改默认配置无需重建镜像。

## 可复现性设计（为什么恢复后逐位一致）

1. **单线程 + 确定性算子**：`torch.set_num_threads(1)` 并开启
   `torch.use_deterministic_algorithms(True)`，同一构建下浮点结果逐位确定。
2. **单一 RNG 流**：新运行先 `torch.manual_seed(seed)`，模型初始化与 Dropout
   都从该流取数；检查点保存 `torch.get_rng_state()`，恢复时原样写回。
3. **每轮打乱可重推**：第 e 轮的样本排列是 `(seed, e)` 的纯函数
   （`random.Random(f"{seed}:{epoch}")`），重启后无需存储即可重建当轮顺序。
4. **批边界检查点**：每个检查点记录 `epoch`、`next_batch`、模型与 Adam 状态、
   RNG 状态、完整损失历史、解析后的配置、`completed` 标志，以及**训练样本的
   数据身份**（见上一节）——即"准备执行第 k 批"的完整现场。恢复 = 先核对数据
   身份，再还原现场并继续同一确定性序列。
5. **原子写**：先写 `.checkpoint.pt.tmp.<uuid>` → fsync → `os.replace` 原子改名
   → 目录 fsync。保存中被强退，磁盘上只会是**完整的旧点或完整的新点**；
   启动时清理残留的临时文件。保存失败不触碰旧文件，明确报错，可重试。

> 注意：逐位一致的前提是**同环境**（同一 torch 构建/CPU 指令集、单线程）。跨
> 不同 CPU 架构或 torch 版本不保证逐位一致，但同机同镜像（如本 compose）成立。

## 不中断对照与退出重启示例

### 自动化端到端验证（推荐）

```bash
python3 scripts/verify_repro.py
```

脚本会真实拉起 uvicorn 服务进程，依次验证（详见脚本 docstring）：
A 不中断基线；B 暂停+续训；C 强退+重启+续训；D 并发续训只放行一个；
E1 保存失败→报错→旧点保留→重试成功；E2 保存失败后强退→回退旧点恢复；
F 慢保存不挡查询；G 保存进行中强退→恢复完整旧/新点；H 基础接口与 /docs。
B–E、G 的逐批损失与最终参数均与各自不中断基线**逐位一致**才判通过。

> 本环境的 seccomp 策略禁止向其他进程发信号，脚本用 `ENABLE_TEST_CRASH_ENDPOINT=1`
> 开启的测试专用 `POST /test/crash`（`os._exit`，立即死亡、无任何清理）模拟强退；
> 恢复语义与 SIGKILL 等价。该端点默认关闭，切勿在生产开启。

### Docker 手动退出重启示例

```bash
docker compose up -d --build
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 8}}'

docker kill resumable-trainer          # 模拟抢占/强退（真实 SIGKILL）
docker compose start                   # 重启容器（检查点在命名卷中）
curl -s localhost:8000/train/status    # state=paused, recovered_after_restart=true
curl -X POST localhost:8000/train/resume
curl -s localhost:8000/train/status    # 直至 state=finished
```

另开一个终端可对同一数据集做一次不中断对照（`force=true` 重跑同配置），用
`GET /train/losses` 与 `GET /train/parameters` 对比两边逐批损失与参数完全一致。

## 实测结果

（以下为 `python3 scripts/verify_repro.py` 在本仓库环境下的真实输出摘要；
完整 66 项全部 PASS，详见脚本输出）

```
    == A: uninterrupted baseline ==
      [PASS] A baseline finished - 95 batches

    == B: pause + resume ==
      [PASS] B1 pause returns only after checkpoint persisted
      [PASS] B2 resume accepted
        losses: 95 entries, exact match
        params: all tensors bit-identical
      [PASS] B3 pause+resume == uninterrupted

    == C: hard crash + restart recovers from latest checkpoint ==
      [PASS] C2 restart recovers paused state from checkpoint
      [PASS] C4 kill+restart == uninterrupted

    == D/E/F/G ==  concurrent resume、保存失败重试、慢保存不挡查询、
                    保存中强退恢复、旧点回退 —— 全部 PASS，恢复结果逐位一致

    == I: checkpoint is bound to the samples it was trained on ==
      [PASS] I0 uninterrupted baseline on a dataset copy - 57 batches
      [PASS] I1 fresh run records a dataset identity - 600 samples
      [PASS] I2a CSV edit during a run does not change loaded data
      [PASS] I2b startup after the edit still reports the original identity
      [PASS] I3a paused run has a complete checkpoint
      [PASS] I3b0 GET /dataset/status reports the change
      [PASS] I3b restart detects changed dataset - dataset_changed
      [PASS] I3c resume refused with a queryable reason
      [PASS] I3d old checkpoint left untouched
      [PASS] I3e non-force start also refused
      [PASS] I4 row reorder alone is a different dataset
      [PASS] I4b resume refused after row reorder
      [PASS] I5a missing original file is reported - dataset_unavailable
      [PASS] I5b resume refused while the file is missing
      [PASS] I5c new run refused while the CSV is missing
      [PASS] I5d force start without a readable CSV is refused
      [PASS] I5e old checkpoint still kept untouched
      [PASS] I5f archived list is queryable
      [PASS] I5g re-verification on resume notices restored data (no restart needed)
      [PASS] I6a restoring the original data allows resume without a restart
      [PASS] I6b restored-data resume is bit-exact
      [PASS] I7a force start accepted on the current CSV
      [PASS] I7b new run is bound to the new data
      [PASS] I7c old checkpoint archived with a reason (not deleted)
      [PASS] I7d forced run finishes on the new data
      [PASS] I8a legacy checkpoint is not silently resumed - legacy_checkpoint
      [PASS] I8b legacy resume refused with a reason
      [PASS] I8c force start archives the legacy checkpoint
      [PASS] I9a unreadable checkpoint is reported, not resumed
      [PASS] I9b resume with an unreadable checkpoint -> 409
      [PASS] I9c non-force start refused
      [PASS] I9d force start archives it and starts a new run
      [PASS] I9e archived reason is checkpoint_unreadable

    ================ SUMMARY ================
      66/66 checks passed
```

`python3 scripts/demo_dataset_change.py` 的真实输出摘要：

```
    [1] start on the original CSV, then pause
        pause -> 200, state=paused, fingerprint=sha256:48f47645cae10cb7...
    [2] replace the default CSV (all features scaled by 0.5) and restart the service
        state=paused ready_to_resume=False
        dataset.state=changed old=sha256:48f47645cae10cb7... new=sha256:42ae2fbf05084679...
        data_conflict.type=dataset_changed
        POST /train/resume -> 409
    [3] POST /train/start {"force": true} on the new CSV
        force start -> 200, new run finished: 38 batches
        archived: checkpoint.<run>.<ts>.pt (reason=dataset_changed)
    [4] fresh runs dir: pause on the original CSV, restart, resume
        after restart: state=paused dataset.state=current ready_to_resume=True
        resume -> 200, finished: 38 batches
```

## 限制与未完成事项

- 单作业、单进程、单 worker：多副本/多 worker 部署会破坏单训练者语义（需引入
  外部锁与共享存储，未实现）。
- 逐位一致仅保证同环境同构建（见上节）；未做跨机器/跨 torch 版本对齐。
- 活动检查点只保留最新一份（原子替换保证新旧二选一）；被 force 新开替换掉的
  旧点会归档到 `runs/archive/`（不删除），但归档没有自动清理策略，长期使用需
  自行管理磁盘。
- `force=true` 新开会先把旧检查点移到 `runs/archive/`：若新运行在写出第一个
  检查点前崩溃，活动 `checkpoint.pt` 不存在（状态回到 `idle`），只能重新开始；
  旧运行仍在归档中，可手动取回（这是刻意语义，避免被替换的旧运行在重启后被
  悄悄恢复）。
- 数据身份只覆盖**解析后的训练张量**（特征列名、行序、float32 值、int64 标签）。
  标签列名固定为 `label`；若改动列名/新增特征列会改变身份并拒绝续训。CSV 中
  与训练无关的排版变化（空白、数值写法）不会改变身份，这是有意为之。
- 运行中修改 CSV 只影响状态查询里显示的“磁盘现状”（`dataset.state=changed`），
  不影响已加载数据，也不影响写出的检查点身份；该运行结束后下次启动仍会以
  磁盘现状做校验。
- `GET /train/parameters` 取自最近**已持久化**检查点，运行中可能落后最多
  `checkpoint_every` 批；未提供"当前内存参数"下载（避免与训练线程竞争）。
- 本环境无 docker，镜像构建与 compose 未实际执行；Dockerfile/compose 与本地实
  测使用完全相同的启动命令与依赖版本，但未在真实 Docker 中跑过验证脚本。
- 暂停等待上限为 `pause_timeout_seconds`（默认 30s），超时返回 504，训练仍会
  在批边界停下，可轮询状态确认。

## 数据绑定修复的完成度（如实说明）

**已实现**

- 检查点写入数据身份（特征列名 + 行序 + float32 值 + int64 标签，`sha256`）；
  启动/运行中均记录；旧检查点无此字段。
- 启动时**一次读取** CSV，样本与身份来自同一份字节；训练线程只用内存快照。
- 重启后核对身份：一致才可续训；`dataset_changed` / `dataset_unavailable` /
  `legacy_checkpoint` / `checkpoint_unreadable` 一律 409，旧检查点原样保留。
- 可查询原因：`GET /train/status`（`dataset` / `data_conflict` /
  `checkpoint_issue` / `archived_checkpoints`）、`GET /dataset/status`、
  `GET /train/archived`。
- 恢复原 CSV 后（甚至无需重启，`resume` 会重新核对）可继续，且逐位一致。
- `force=true` 新开：旧点移入 `runs/archive/` + JSON 清单，不删除；新运行绑定
  当前 CSV。CSV 不可读时 force 也拒绝（400），旧点不动。
- `GET /train/parameters` 在检查点损坏时返回 409 而不是 500/静默。
- compose 增加 `./data`、`./config` bind mount，换 CSV 无需重建镜像。

**已验证**（本环境，Python 3.11.2 / torch 2.15.0+cpu）

- `scripts/verify_repro.py`：**66/66 PASS**，其中 A–H 共 33 项为原有暂停/续训/
  强退/保存失败/慢保存/并发续训与逐位一致性回归；I 系列 33 项覆盖上述数据绑定
  全部场景（含运行中改 CSV、行序变化、文件丢失、旧格式、损坏检查点、force 归档、
  恢复后逐位一致）。
- `scripts/demo_dataset_change.py`：换数据 → 409 → force 归档 → 原数据恢复续训，
  全流程通过。

**未完成 / 未验证**

- **本环境没有 Docker**：Dockerfile 与 compose 未实际构建运行；`/docs` 的可用性
  由本地 uvicorn 验证（H2）。compose 的换数据演示步骤是文档化的操作，未在真实
  Docker 中执行。
- 未实现跨进程/多副本锁；归档无自动清理；不保证跨 CPU/torch 版本逐位一致。
- 数据身份基于解析后的训练张量，不覆盖 CSV 里与训练无关的额外列内容（这些列
  本来就不参与训练）。
