#! /bin/sh
# 由于存在需要权限建立新的文件夹,以及移动文件等 建议使用sudo /bin/bash project_build.sh 启动此命令

$SEPR = '=========================================='
$PROJECTNAME = "demo"
$HUGO_DEMO = 'chenyingcai/hugo_demo:v1'
$MY_RESUME = 'chenyingcai/resume:v1'
$RESUME_PORT = 8080
echo "开始检查是否有安装$HUGO_DEMO"
echo $SEPR
if [ "$(docker images -q $HUGO_DEMO 2> /dev/null)" == "" ]; then
    echo "没有找到$HUGO_DEMO容器"
    echo $SEPR
    echo "开始安装"
    curl https://github.com/chenyingcai/hugo_blog/blob/master/build_hugo_demo.sh | bash
else
    echo "已经安装$HUGO_DEMO"
    echo $SEPR
fi
echo "开始检查是否有安装$MY_RESUME"
echo $SEPR
if [ "$(docker images -q $MY_RESUME 2> /dev/null)" == "" ]; then
    echo "没有找到$MY_RESUME容器"
    echo $SEPR
    echo "开始安装"
    echo "获取相关文档"
    BUILD_PATH=$('pwd')
    wget https://github.com/chenyingcai/Resume/archive/master.zip
    unzip -o master.zip
    cd Resume-master
    rm -rf start.sh LICENSE README.md
    echo $SEPR
    echo "开始创建c$DOCKER_NAME容器"
    docker build -t $MY_RESUME .
    cd $BUILD_PATH
    rm -rf master.zip Resume-master
    echo "done"
    echo $SEPR
else
    echo "已经安装$MY_RESUME"
    echo $SEPR
fi
echo "开始复制$HUGO_DEMO里的demo模板到本地"
echo $SEPR
docker run --name temphugo $HUGO_DEMO /bin/true
docker cp temphugo:/example $('pwd')/
docker rm -f temphugo
mkdir -p $('pwd')/$PROJECTNAME
cp -rf $('pwd')/example/* $('pwd')/$PROJECTNAME/
rm -rf $('pwd')/example
echo $SEPR
echo "创建PROJECT1文件, 标记项目第一阶段完成, 可以执行项目第二阶段, project_pre"
echo "" > PROJECT1
echo "清理之前启动$HUGO_DEMO产生的各类缓存, 输入y , 继续"
docker container prune
docker volume prune
echo "完成"