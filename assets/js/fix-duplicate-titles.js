// 修复重复标题问题的脚本
document.addEventListener('DOMContentLoaded', function() {
    // 文章标题
    const postTitle = document.querySelector('.post-title');
    if (!postTitle) return; // 如果不是文章页面则退出
    
    const titleText = postTitle.textContent.trim();
    
    // 查找所有h1标题
    const allH1s = document.querySelectorAll('h1');
    
    // 遍历所有h1标题，隐藏与文章标题相同的
    allH1s.forEach(h1 => {
        // 跳过主标题本身
        if (h1 === postTitle) return;
        
        // 检查标题文本是否相同或相似
        if (h1.textContent.trim() === titleText || 
            h1.textContent.trim().includes(titleText) || 
            titleText.includes(h1.textContent.trim())) {
            
            // 隐藏重复标题
            h1.style.display = 'none';
            h1.style.visibility = 'hidden';
            h1.style.height = '0';
            h1.style.margin = '0';
            h1.style.padding = '0';
            h1.style.overflow = 'hidden';
            
            console.log('隐藏了重复标题:', h1.textContent);
        }
    });
    
    // 特殊情况：找到导航栏下面的直接文本标题
    const mainElement = document.querySelector('main');
    if (mainElement) {
        for (let i = 0; i < mainElement.childNodes.length; i++) {
            const node = mainElement.childNodes[i];
            // 检查是否为文本节点并且包含标题文本
            if (node.nodeType === Node.TEXT_NODE && node.textContent.includes(titleText)) {
                // 替换为空字符串
                node.textContent = '';
                console.log('移除了文本节点中的标题');
            }
        }
    }
}); 