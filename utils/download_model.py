from modelscope.hub.snapshot_download import snapshot_download


# 从modelscope下载
def download_from_modelscope(model_id,
                             model_version,
                             cache_dir):
    model_dir = snapshot_download(model_id,
                                  cache_dir=cache_dir,
                                  revision=model_version)

download_from_modelscope('PollyZhao/bert-base-chinese',
                         'master',
                         '/Users/ethan/Documents/projects/models')

