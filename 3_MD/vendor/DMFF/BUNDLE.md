# Bundled DMFF Runtime

这个目录是为了 `3_MD` 单独打包运行而复制的 DMFF runtime 副本。

来源：

- `/Users/jeremychen/Desktop/Project/project_h2o_etoh/DMFF`

当前复制内容：

- `dmff/` Python 包源码
- `LICENSE`
- `README.upstream.md`

这个副本用于 `3_MD/CMD_H2O` 的 MD client 和测试脚本导入，不包含上游仓库的 `.git`、`tests/`、`docs/`、`examples/` 或构建产物。

本地补丁还包括：

- `dmff/settings.py` 改为同时兼容 `from jax import config` 和旧版 `from jax.config import config`
- `dmff/optimize.py` 与 `dmff/api/paramset.py` 改为使用 `jax.tree_util.tree_map`

这样可以兼容较新的 `jax==0.7.x` 环境。

如果训练端的本地 DMFF 再有修改，更新方式是重新同步：

```bash
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
  /Users/jeremychen/Desktop/Project/project_h2o_etoh/DMFF/dmff \
  /Users/jeremychen/Desktop/Project/project_h2o_etoh/3_MD/vendor/DMFF/
cp /Users/jeremychen/Desktop/Project/project_h2o_etoh/DMFF/LICENSE \
  /Users/jeremychen/Desktop/Project/project_h2o_etoh/3_MD/vendor/DMFF/LICENSE
cp /Users/jeremychen/Desktop/Project/project_h2o_etoh/DMFF/README.md \
  /Users/jeremychen/Desktop/Project/project_h2o_etoh/3_MD/vendor/DMFF/README.upstream.md
```
