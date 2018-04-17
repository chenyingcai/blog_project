#! /bin/sh
# 由于存在需要权限建立新的文件夹,以及移动文件等 建议使用sudo /bin/bash project_publish.sh 启动此命令
PROJECTNAME = "demo"
GITPAGE="chenyingcai.github.io"
HUGO_DEMO = 'chenyingcai/hugo_demo:v1'

if [[ -f $("pwd")/PROJECT3 ]]; then
    $MAIN_ADDRESS=$('pwd')
    copyresume
    docker run --it --rm -p $BLOG_PORT:1313 -v $('pwd')/$PROJECTNAME/:/hugo/ $HUGO_DEMO hugo -d $GITPAGE/
    cd $('pwd')/$PROJECTNAME/$GITPAGE
    git add .
    git commit -m "update"
    git push origin master
    cd $MAIN_ADDRESS
fi

if [[ -f $("pwd")/PROJECT2 ]]; then
    $MAIN_ADDRESS=$('pwd')
    echo "先用git clone git@github.com:chenyingcai/$GITPAGE.git 到本地"
    git clone git@github.com:chenyingcai/$GITPAGE.git $('pwd')/$PROJECTNAME/
    cd $('pwd')/$PROJECTNAME/$GITPAGE
    ls | grep -v ".git" | xargs rm -rf
    cd $MAIN_ADDRESS
    copyresume
    docker run --it --rm -p $BLOG_PORT:1313 -v $('pwd')/$PROJECTNAME/:/hugo/ $HUGO_DEMO hugo -d $GITPAGE/
    cd $('pwd')/$PROJECTNAME/$GITPAGE
    git add .
    git commit -m "update"
    git push origin master
    cd $MAIN_ADDRESS
    echo "" > PROJECT3
else
    echo "WARNING!: 未完成项目第二阶段"
    echo "请先执行项目第二阶段: sudo curl https://raw.githubusercontent.com/chenyingcai/blog_project/master/project_pre.sh | bash"
fi

