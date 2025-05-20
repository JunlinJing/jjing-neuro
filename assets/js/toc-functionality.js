// 目录导航功能 - 优化版
document.addEventListener('DOMContentLoaded', function() {
    // 获取目录元素
    const toc = document.getElementById('toc');
    if (!toc) return;
    
    const tocToggle = document.getElementById('toc-toggle');
    const tocContent = document.getElementById('toc-content');
    
    // 创建阅读进度指示器
    const progressIndicator = document.createElement('div');
    progressIndicator.className = 'toc-progress-indicator';
    progressIndicator.innerHTML = '<div class="toc-progress-bar"></div>';
    toc.appendChild(progressIndicator);
    
    const progressBar = progressIndicator.querySelector('.toc-progress-bar');
    
    // 更新阅读进度
    function updateReadingProgress() {
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight - windowHeight;
        const scrollTop = window.scrollY;
        const progress = (scrollTop / documentHeight) * 100;
        
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
    }
    
    // 初始更新进度
    updateReadingProgress();
    
    // 监听滚动事件更新进度
    window.addEventListener('scroll', throttle(updateReadingProgress, 50));
    
    // 设置适当的图标
    function setAppropriateIcon() {
        if (!tocToggle || !tocToggle.querySelector('i')) return;
        
        const icon = tocToggle.querySelector('i');
        const isCollapsed = toc.classList.contains('collapsed');
        const isMobile = window.innerWidth < 768;
        
        if (isCollapsed) {
            icon.className = isMobile ? 'fas fa-chevron-down' : 'fas fa-chevron-right';
        } else {
            icon.className = isMobile ? 'fas fa-chevron-up' : 'fas fa-chevron-left';
        }
    }
    
    // 目录折叠/展开功能
    function toggleTOC() {
        toc.classList.toggle('collapsed');
        setAppropriateIcon();
        
        // 添加动画效果
        if (toc.classList.contains('collapsed')) {
            toc.style.transition = 'all 0.3s ease-out';
        } else {
            toc.style.transition = 'all 0.3s ease-in';
            
            // 展开后等待短暂时间再显示内容，让动画更平滑
            setTimeout(() => {
                if (!toc.classList.contains('collapsed') && tocContent) {
                    tocContent.style.opacity = '1';
                }
            }, 150);
        }
        
        // 保存状态到localStorage
        localStorage.setItem('tocCollapsed', toc.classList.contains('collapsed'));
    }
    
    // 添加折叠/展开按钮点击事件
    if (tocToggle) {
        tocToggle.addEventListener('click', function(e) {
            e.preventDefault();
            
            // 如果是收起状态，先将内容透明度设为0
            if (!toc.classList.contains('collapsed') && tocContent) {
                tocContent.style.opacity = '0';
                tocContent.style.transition = 'opacity 0.15s ease-out';
            }
            
            // 短暂延迟后切换状态，让动画更平滑
            setTimeout(() => {
                toggleTOC();
            }, 50);
        });
    }
    
    // 从localStorage读取状态
    const savedCollapsedState = localStorage.getItem('tocCollapsed');
    
    // 设置初始状态
    if (savedCollapsedState === 'true') {
        toc.classList.add('collapsed');
        if (tocContent) {
            tocContent.style.opacity = '0';
        }
    } else if (window.innerWidth < 768) {
        // 在移动设备上默认折叠
        toc.classList.add('collapsed');
        if (tocContent) {
            tocContent.style.opacity = '0';
        }
    } else if (tocContent) {
        tocContent.style.opacity = '1';
    }
    
    // 设置初始图标
    setAppropriateIcon();
    
    // 监听窗口大小变化
    window.addEventListener('resize', function() {
        setAppropriateIcon();
        
        // 在移动设备上切换时自动折叠目录
        if (window.innerWidth < 768 && !toc.classList.contains('collapsed')) {
            tocContent.style.opacity = '0';
            setTimeout(() => {
                toc.classList.add('collapsed');
                setAppropriateIcon();
            }, 100);
        }
    });
    
    // 生成目录内容
    const content = document.querySelector('.post-content');
    if (!content) return;
    
    const headings = content.querySelectorAll('h2, h3, h4, h5');
    
    if (headings.length > 0) {
        const tocList = document.createElement('ul');
        tocList.className = 'toc-list';
        
        let currentList = tocList;
        let previousLevel = 2;
        let subList = null;
        
        headings.forEach((heading, index) => {
            // 为标题添加ID，如果不存在
            if (!heading.id) {
                // 使用标题文本生成ID
                const headingText = heading.textContent.trim();
                const id = headingText.toLowerCase()
                    .replace(/\s+/g, '-')
                    .replace(/[^\w\-]+/g, '')
                    .replace(/\-\-+/g, '-')
                    .replace(/^-+/, '')
                    .replace(/-+$/, '');
                
                heading.id = id || `heading-${index}`;
            }
            
            const level = parseInt(heading.tagName.charAt(1));
            const li = document.createElement('li');
            li.className = `toc-item toc-level-${level}`;
            
            const a = document.createElement('a');
            a.href = `#${heading.id}`;
            a.textContent = heading.textContent;
            a.className = `toc-link toc-link-${level}`;
            
            // 创建层级结构
            if (level > previousLevel) {
                subList = document.createElement('ul');
                subList.className = 'toc-sublist';
                if (currentList.lastChild) {
                    currentList.lastChild.appendChild(subList);
                    currentList = subList;
                }
            } else if (level < previousLevel) {
                const diff = previousLevel - level;
                for (let i = 0; i < diff; i++) {
                    if (currentList.parentNode && currentList.parentNode.parentNode) {
                        currentList = currentList.parentNode.parentNode;
                    }
                }
            }
            
            li.appendChild(a);
            currentList.appendChild(li);
            previousLevel = level;
            
            // 点击目录项时平滑滚动
            a.addEventListener('click', (e) => {
                e.preventDefault();
                
                // 移除所有活动状态
                document.querySelectorAll('.toc-link').forEach(link => {
                    link.classList.remove('active');
                });
                
                // 添加当前活动状态
                a.classList.add('active');
                
                // 平滑滚动到标题位置，稍微偏移以避免被固定头部覆盖
                const yOffset = -80; 
                const targetElement = document.getElementById(heading.id);
                
                if (targetElement) {
                    const y = targetElement.getBoundingClientRect().top + window.pageYOffset + yOffset;
                    
                    // 使用更平滑的滚动动画
                    window.scrollTo({
                        top: y,
                        behavior: 'smooth'
                    });
                    
                    // 在滚动时突出显示目标标题
                    targetElement.classList.add('highlight-target');
                    setTimeout(() => {
                        targetElement.classList.remove('highlight-target');
                    }, 2000);
                    
                    // 更新URL，不触发跳转
                    history.pushState(null, null, `#${heading.id}`);
                    
                    // 在移动设备上点击后折叠目录
                    if (window.innerWidth < 768) {
                        if (tocContent) {
                            tocContent.style.opacity = '0';
                            tocContent.style.transition = 'opacity 0.15s ease-out';
                        }
                        
                        setTimeout(() => {
                            toc.classList.add('collapsed');
                            setAppropriateIcon();
                        }, 150);
                    }
                }
            });
        });
        
        if (tocContent) {
            // 清空现有内容
            while (tocContent.firstChild) {
                tocContent.removeChild(tocContent.firstChild);
            }
            
            tocContent.appendChild(tocList);
        }
    } else {
        // 如果没有标题，隐藏目录
        toc.style.display = 'none';
    }
    
    // 滚动监听，高亮当前阅读部分 - 使用节流函数优化性能
    function throttle(callback, delay) {
        let lastCall = 0;
        return function() {
            const now = new Date().getTime();
            if (now - lastCall >= delay) {
                callback();
                lastCall = now;
            }
        };
    }
    
    const highlightCurrentSection = throttle(function() {
        if (!toc || !tocContent) return;
        
        const scrollPosition = window.scrollY + 100;
        const headings = document.querySelectorAll('.post-content h2, .post-content h3, .post-content h4, .post-content h5');
        if (headings.length === 0) return;
        
        let currentHeading = null;
        
        for (let i = 0; i < headings.length; i++) {
            const heading = headings[i];
            const headingPosition = heading.offsetTop;
            
            if (scrollPosition >= headingPosition) {
                currentHeading = heading;
            } else {
                break;
            }
        }
        
        if (currentHeading) {
            // 移除所有活动状态
            document.querySelectorAll('.toc-link').forEach(link => {
                link.classList.remove('active');
            });
            
            // 添加当前活动状态
            const currentLink = document.querySelector(`.toc-link[href="#${currentHeading.id}"]`);
            if (currentLink) {
                currentLink.classList.add('active');
                
                // 确保活动链接可见 - 改进滚动行为
                if (tocContent.scrollHeight > tocContent.clientHeight && !toc.classList.contains('collapsed')) {
                    const linkTop = currentLink.offsetTop;
                    const tocTop = tocContent.scrollTop;
                    const tocBottom = tocTop + tocContent.clientHeight;
                    
                    // 如果当前链接不在可视区域内，平滑滚动到合适位置
                    if (linkTop < tocTop || linkTop > tocBottom - 40) {
                        tocContent.scrollTo({
                            top: linkTop - tocContent.clientHeight / 2,
                            behavior: 'smooth'
                        });
                    }
                }
            }
        }
    }, 100);
    
    window.addEventListener('scroll', highlightCurrentSection);
    
    // 初次加载时高亮当前位置
    highlightCurrentSection();
    
    // 处理页面加载时的hash定位
    if (window.location.hash) {
        const targetId = window.location.hash.substring(1);
        const targetElement = document.getElementById(targetId);
        const targetLink = document.querySelector(`.toc-link[href="#${targetId}"]`);
        
        if (targetElement && targetLink) {
            // 延迟执行以确保页面完全加载
            setTimeout(() => {
                // 平滑滚动到目标位置
                const yOffset = -80;
                const y = targetElement.getBoundingClientRect().top + window.pageYOffset + yOffset;
                
                window.scrollTo({
                    top: y,
                    behavior: 'smooth'
                });
                
                // 高亮目标链接
                document.querySelectorAll('.toc-link').forEach(link => {
                    link.classList.remove('active');
                });
                targetLink.classList.add('active');
                
                // 突出显示目标标题
                targetElement.classList.add('highlight-target');
                setTimeout(() => {
                    targetElement.classList.remove('highlight-target');
                }, 2000);
            }, 300);
        }
    }
}); 