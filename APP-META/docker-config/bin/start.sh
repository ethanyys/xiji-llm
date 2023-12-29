#!/bin/bash --login

echo "hello"

cd /home/admin/$APP_NAME/target/$APP_NAME/

# 多进程启动有cuda共享问题
# gunicorn -w 1 -b 0.0.0.0:9003 serving.app:flash_app --access-logfile logs/api-access.log --log-config ./conf/log.cfg --preload

main
