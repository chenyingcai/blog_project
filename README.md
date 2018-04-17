我们使用本项目中的`sh` 脚本命令来构建系统, 需要用到的有 [git]() [hugo_demo]() [Resume]()

## 注意:
   这个项目是在 **Linux Ubuntu 16 以及以上** 实现的

## 初次构建
使用以下命令获取`sh`文件

```sh
curl -o project_build.sh https://raw.githubusercontent.com/chenyingcai/blog_project/master/project_build.sh
sudo /bin/sh project_build.sh
```
或者 **(推荐)**

```sh
sudo curl https://raw.githubusercontent.com/chenyingcai/blog_project/master/project_build.sh | bash
```

或者

```sh
git clone https://github.com/chenyingcai/blog_project.git
cd blog_project | sudo /bin/sh project_build.sh
```
自动构建

## 所有都齐全了后, 使用 `project_pre.sh` 预览修稿成果

```sh
curl -o project_pre.sh https://raw.githubusercontent.com/chenyingcai/blog_project/master/project_pre.sh
sudo /bin/sh project_pre.sh
```
或者

```sh
sudo curl https://raw.githubusercontent.com/chenyingcai/blog_project/master/project_pre.sh | bash
```

或者

```sh
git clone https://github.com/chenyingcai/blog_project.git
cd blog_project | sudo /bin/sh project_pre.sh
```

## 构建静态文件, 并发布到自己的github page 中去

```sh
curl -o project_publish.sh https://raw.githubusercontent.com/chenyingcai/blog_project/master/project_publish.sh
sudo /bin/sh project_publish.sh
```
或者

```sh
sudo curl https://raw.githubusercontent.com/chenyingcai/blog_project/master/project_publish.sh | bash
```

或者

```sh
git clone https://github.com/chenyingcai/blog_project.git
cd blog_project | sudo /bin/sh project_publish.sh
```
