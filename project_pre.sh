#!/bin/sh
# 由于存在需要权限建立新的文件夹,以及移动文件等 建议使用sudo /bin/bash project_pre.sh 启动此命令
SEPR="=========================================="
PROJECTNAME="demo"
HUGO_DEMO="chenyingcai/hugo_demo:v1"
MY_RESUME="chenyingcai/resume:v1"
BLOG_PORT=8000
RESUME_PORT=8080
if [ -f $("pwd")/PROJECT2 ]; then
    alias hugopre="docker run -it --rm -p $BLOG_PORT:1313 -v $('pwd')/$PROJECTNAME/:/hugo/ $HUGO_DEMO hugo server --baseURL=localhost:$BLOG_PORT --bind=0.0.0.0 --appendPort=false"
    alias copyresume="sudo rm -rf $('pwd')/resume/static/* && docker exec -it resume generate && cp -rf $('pwd')/resume/static/* $('pwd')/$PROJECTNAME/static/resume/"
    echo "之后每次要预览时, 运行就再一次运行这个脚本, 或者hugopre"
    hugopre
    echo "若修改过resume内容,请执行copyresumesudo rm -rf $('pwd')/resume/static/* && docker exec -it resume generate && cp -rf $('pwd')/resume/static/* $('pwd')/$PROJECTNAME/static/resume/"
    echo """修改好后,若要发布blog, 请执行项目第三阶段:
    sudo curl https://raw.githubusercontent.com/chenyingcai/blog_project/master/project_publish.sh | bash"""
elif [ -f $('pwd')/PROJECT1 ]; then
    echo -e "发现PROJECT1文件\n 进行项目第二阶段"
    echo "创建resume项目"
    mkdir -p $('pwd')/resume
    docker run -d --name resume_tmp $MY_RESUME
    if [ ! -d "$('pwd')/resume/config" ]; then
      docker cp resume_tmp:/usr/html/user/config $('pwd')/resume/config
    fi
    if [ ! -d "$('pwd')/resume/pages" ]; then
      docker cp resume_tmp:/usr/html/user/pages $('pwd')/resume/pages
    fi
    if [ ! -d "$('pwd')/resume/themes" ]; then
      docker cp resume_tmp:/usr/html/user/themes $('pwd')/resume/themes
    fi
    docker rm -f resume_tmp resume >/dev/null 2>&1
    docker run -d --name resume -p $RESUME_PORT:80 \
        -v $('pwd')/resume/themes:/usr/html/user/themes \
        -v $('pwd')/resume/pages:/usr/html/user/pages \
        -v $('pwd')/resume/config/:/usr/html/user/config/ \
        -v $('pwd')/resume/static/:/usr/html/static \
        --restart=always $MY_RESUME
    echo "run the ngnix"
    docker exec -it resume run
    echo "Done"
    echo "generate the initial html file"
    docker exec -it resume generate
    echo "Done"
    echo $SEPR
    echo "有的时候我们启动resume容器会出现一些意外, 若打开localhost:$RESUME_PORT无显示"
    echo "重新执行docker exec -it resume run和docker exec -it resume generate"
    echo $SEPR
    echo "resume项目可以通过localhost:$RESUME_PORT实时预览"
    echo "现在我们开始将resume项目产生的简历与博客连接"
    echo $SEPR    
    echo "我们将resume项目的静态文件夹static里面的内容复制到blog的静态文件夹$('pwd')/$PROJECTNAME/static下的resume目录中"
    echo "由于hugo blog 在发布时, 会直接将$('pwd')/$PROJECTNAME/static下的所有内容复制到博客的根目录中"
    echo "这样我们就可通过[博客的baseURL]/resume/ 查看我们简历了"
    mkdir -p $('pwd')/$PROJECTNAME/static/resume
    cp -rf $('pwd')/resume/static/* $('pwd')/$PROJECTNAME/static/resume/
    echo "创建alias"
    alias hugopre="docker run -it --rm -p $BLOG_PORT:1313 -v $('pwd')/$PROJECTNAME/:/hugo/ $HUGO_DEMO hugo server --baseURL=localhost:$BLOG_PORT --bind=0.0.0.0 --appendPort=false"
    echo "之后每次要预览时, 运行就再一次运行这个脚本, 或者hugopre"
    echo "若修改过resume内容,请执行copyresumesudo rm -rf $('pwd')/resume/static/* && docker exec -it resume generate && cp -rf $('pwd')/resume/static/* $('pwd')/$PROJECTNAME/static/resume/"
    alias copyresume="sudo rm -rf $('pwd')/resume/static/* && docker exec -it resume generate && cp -rf $('pwd')/resume/static/* $('pwd')/$PROJECTNAME/static/resume/"
    hugopre
    rm -rf PROJECT1
    echo "" > PROJECT2
else
    echo "WARNING: 未完成项目第一阶段"
    echo "请先执行项目第一阶段: sudo curl https://raw.githubusercontent.com/chenyingcai/blog_project/master/project_build.sh | bash"
fi
