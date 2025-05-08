# Jim Jing's Academic Website

这是我的个人学术网站，基于Jekyll构建，专注于展示神经科学研究和人工智能相关工作。

## 网站内容

- **关于(About)**: 个人简介、教育背景、研究经历
- **研究(Research)**: 当前研究项目、研究方法、合作机会
- **出版物(Publications)**: 期刊文章、会议论文、预印本
- **项目(Projects)**: 开源项目、技术贡献
- **博客(Blog)**: 技术教程、研究笔记、学术思考

## 本地开发

1. 安装依赖:
```bash
# 安装 Ruby 和 Bundler
gem install bundler
# 安装项目依赖
bundle install
```

2. 本地运行:
```bash
bundle exec jekyll serve
```

3. 访问 `http://localhost:4000/jjing-neuro/`

## 部署

网站通过 GitHub Pages 自动部署，访问地址：[https://junlinjing.github.io/jjing-neuro/](https://junlinjing.github.io/jjing-neuro/)

## 技术栈

- [Jekyll](https://jekyllrb.com/) - 静态网站生成器
- [SASS](https://sass-lang.com/) - CSS预处理器
- Responsive Design - 适配移动端和桌面端
- Dark/Light Theme - 支持暗色/亮色主题切换

## 文件结构

```
.
├── _includes/          # 可重用的模板组件
├── _layouts/          # 页面布局模板
├── _pages/           # 主要页面内容
├── _posts/           # 博客文章
├── assets/           # 静态资源（图片、样式等）
└── _config.yml       # 网站配置文件
```

## 配置说明

主要配置在 `_config.yml` 文件中：

```yaml
# 基本信息
title: Jim Jing
bio: Neuroscience Researcher | AI Enthusiast | Exploring Brain and Cognition

# 社交媒体链接
twitter: JimJing1997
linkedin: jjunlin
github: JunlinJing

# 学术链接
researchgate: Junlin-Jing
orcid: 0009-0006-1290-7445
```

## 许可

本网站基于 [MIT License](LICENSE) 开源。

## 致谢

网站主题基于 [Indigo](https://github.com/sergiokopplin/indigo) 修改，感谢原作者的工作。
