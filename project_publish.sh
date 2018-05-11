#!/bin/sh
# 由于存在需要权限建立新的文件夹,以及移动文件等 建议使用sudo /bin/bash project_publish.sh 启动此命令
PROJECTNAME="demo"
GITPAGE="chenyingcai.github.io"
HUGO_DEMO='chenyingcai/hugo_demo:v1'
SEPR="=========================================="
BLOG_PORT=8000

if [ -f $("pwd")/PROJECT2 ]; then
    MAIN_ADDRESS=$('pwd')
    if [ -r $('pwd')/$PROJECTNAME/static/resume ]; then
        read -r -p "是否有对resume进行了修改,[y/n]" response
        case $response in 
            [yY][eE][sS]|[yY])
                if [ ! "$(docker ps -q -f name=resume)" ]; then
                    if [ "$(docker ps -aq -f status=exited -f name=resume)" ]; then
                        # cleanup
                        docker rm resume
                    fi
                    # run your container
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
                    sudo rm -rf $('pwd')/resume/static/* && docker exec -it resume generate && cp -rf $('pwd')/resume/static/* $('pwd')/$PROJECTNAME/static/resume/
                    echo "Done"
                else
                    sudo rm -rf $('pwd')/resume/static/* && docker exec -it resume generate && cp -rf $('pwd')/resume/static/* $('pwd')/$PROJECTNAME/static/resume/
                fi
                read -r -p "是否建立PROJECT3文档,[y/n]" response
                case $response in 
                    [yY][eE][sS]|[yY])
                        echo "正在建立PROJECT3初始文档"
                        echo $SEPR
                        echo "" > PROJECT3
                        echo "done"
                        ;;
                    [nN][oO]|[nN])
                        echo $SEPR
                        ;;
                    *)
                        echo "无效输入.."
                        ;;
                esac
                ;;
            [nN][oO]|[nN])
                echo $SEPR
                ;;
            *)
                echo "无效输入.."
                ;;
        esac
    else
        echo $SEPR
        read -r -p "警告: 没有发现resume文档, yes: 继续, No: 退出" response
        case $response in 
            [yY][eE][sS]|[yY])
                echo "继续"
                echo $SEPR
                ;;
            [nN][oO]|[nN])
                echo $SEPR
                return
                ;;
            *)
                echo "无效输入.."
                return
                ;;
        esac
    fi
    cd $('pwd')/$PROJECTNAME/
    if [ ! -r $('pwd')/$GITPAGE ];then
        echo "没有发现$GITPAGE项目"
        echo $SEPR
        git clone https://github.com/chenyingcai/$GITPAGE.git
    fi
    cd $GITPAGE
    ls | grep -v ".git" | xargs sudo rm -rf
    cd $MAIN_ADDRESS
    read -p "按任意键继续..."
    docker run -it --rm -v $('pwd')/$PROJECTNAME/:/hugo/ $HUGO_DEMO hugo -d $GITPAGE/
    read -p "按任意键继续..."
    cd $('pwd')/$PROJECTNAME/$GITPAGE/
    git status
    echo $SEPR
    sudo git add .
    sudo git commit -m "update"
    sudo git push -u origin master
    echo "done"
    echo $SEPR
    cd $MAIN_ADDRESS
else
    echo "WARNING!: 未完成项目第二阶段"
    echo "请先执行项目第二阶段: sudo curl https://raw.githubusercontent.com/chenyingcai/blog_project/master/project_pre.sh | bash"
fi