---
layout: page
title: News
permalink: /news/
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>

<style>
body .page-content {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 2rem 1rem !important;
}
.news-controls {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
    background: var(--bg-color-secondary);
    padding: 1.2rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}
.control-item {
    flex: 1;
    min-width: 180px;
}
.control-item select, .control-item input {
    width: 100%;
    padding: 0.7rem;
    border: 1px solid var(--border-color);
    border-radius: 7px;
    background: var(--bg-color);
    color: var(--text-color);
    font-size: 0.97rem;
    transition: all 0.3s;
}
.control-item select:hover, .control-item input:hover {
    border-color: #999;
}
.control-item select:focus, .control-item input:focus {
    outline: none;
    border-color: var(--accent-color);
    box-shadow: 0 0 0 2px rgba(var(--accent-color-rgb), 0.1);
}
.news-list {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    margin: 2rem 0;
}
.news-card {
    background: var(--bg-color-secondary);
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    margin-bottom: 1.5rem;
    padding: 1.5em 1.5em 1.1em 1.5em;
    min-width: 0;
    position: relative;
    transition: box-shadow 0.2s;
    width: 700px;
    max-width: 100%;
    margin-left: auto;
    margin-right: auto;
}
.news-card:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.13);
}
.news-tag {
    display: block;
    width: 100%;
    border: 1.5px solid #bbb;
    border-radius: 14px 14px 0 0;
    font-size: 1.08em;
    font-weight: bold;
    text-align: left;
    line-height: 2.1em;
    letter-spacing: 1px;
    color: #333;
    background: transparent;
    margin-bottom: 0.7em;
    padding-left: 1.2em;
    box-sizing: border-box;
}
.news-tag.personal { border-color: #6c63ff; color: #6c63ff; }
.news-tag.project { border-color: #00b894; color: #00b894; }
.news-tag.site { border-color: #0984e3; color: #0984e3; }
.news-tag.blog { border-color: #fdcb6e; color: #b8860b; }
.news-tag.research { border-color: #e84393; color: #e84393; }
.news-info {
    font-size: 0.98em;
    color: #888;
    margin-bottom: 0.5em;
    display: flex;
    flex-wrap: wrap;
    gap: 1.5em;
}
.news-info span { display: block; min-width: 120px; }
.news-title {
    font-size: 1.25em;
    font-weight: 600;
    margin: 0.8em 0;
    color: var(--heading-color) !important;
}
html.dark .news-title,
:root.dark .news-title {
    color: #fff !important;
}
html:not(.dark) .news-title,
:root:not(.dark) .news-title {
    color: #222 !important;
}
.news-content {
    margin: 1.2em 0;
    font-size: 1.05em;
    line-height: 1.7;
    color: var(--text-color);
    padding: 0.8em 1.2em;
    background: var(--bg-color);
    border-radius: 8px;
    border-left: 4px solid var(--accent-color);
}

.news-card[data-category="personal"] .news-content {
    border-left-color: #6c63ff;
}

.news-card[data-category="site"] .news-content {
    border-left-color: #0984e3;
}

.news-card[data-category="research"] .news-content {
    border-left-color: #e84393;
}

.news-card[data-category="project"] .news-content {
    border-left-color: #00b894;
}

html:not(.dark) .news-content,
:root:not(.dark) .news-content {
    color: #333;
    background: #f8f9fa;
}

html.dark .news-content,
:root.dark .news-content {
    color: #eee;
    background: rgba(255, 255, 255, 0.05);
}

.news-points {
    margin: 1em 0;
    padding-left: 1.5em;
    list-style-type: disc;
}

.news-points li {
    margin-bottom: 0.5em;
    line-height: 1.6;
    color: var(--text-color);
}
.news-social-share {
    display: flex;
    gap: 0.7em;
    margin-top: 0.7em;
    justify-content: flex-end;
}
.news-social-share a {
    display: inline-block;
    width: 1em;
    height: 1em;
    vertical-align: middle;
    text-align: center;
    line-height: 1;
    box-sizing: content-box;
    color: #888;
    transition: color 0.2s;
}
.news-social-share .share-xiaohongshu svg {
    width: 1em;
    height: 1em;
    vertical-align: middle;
    display: inline-block;
}
.news-social-share a:hover {
    color: var(--accent-color, #6c63ff);
}
.news-social-share a.share-twitter:hover { color: #1da1f2; }
.news-social-share a.share-linkedin:hover { color: #0077b5; }
.news-social-share a.share-facebook:hover { color: #1877f3; }
.news-social-share a.share-wechat:hover { color: #09bb07; }
.news-social-share a.share-xiaohongshu:hover { color: #ff2442; }
.news-tools {
    position: absolute;
    right: 1.2em;
    bottom: 1.1em;
    margin-top: 0;
}
.tool-button {
    padding: 0.32rem 0.8rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: transparent;
    color: #666;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.97rem;
}
.tool-button:hover {
    background: var(--accent-color);
    color: #fff;
    border-color: var(--accent-color);
}
.pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;
    margin: 2rem 0;
    padding: 1rem;
}
.pagination-button {
    padding: 0.5rem 1rem;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    background: var(--bg-color);
    color: var(--text-color);
    cursor: pointer;
    transition: all 0.3s;
    font-size: 0.9rem;
    min-width: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.pagination-button:hover:not(:disabled) {
    background: var(--accent-color);
    color: white;
    border-color: var(--accent-color);
    transform: translateY(-1px);
}
.pagination-button.active {
    background: var(--accent-color);
    color: white;
    border-color: var(--accent-color);
    font-weight: bold;
}
.pagination-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    background: var(--bg-color-secondary);
}
@media (max-width: 768px) {
    .news-controls { flex-direction: column; }
    .news-list { gap: 1rem; }
    .news-card { padding: 1.1em 0.7em 1em 0.7em; width: 100%; max-width: 100%; }
    .news-tag { font-size: 1em; padding-left: 0.7em; }
    .news-info { gap: 0.7em; font-size: 0.95em; }
    .news-title { font-size: 1em; }
    .news-points { font-size: 0.97em; }
}
body.dark .news-card .news-title,
html[data-theme="dark"] .news-card .news-title {
    color: #fff !important;
}
.news-social-share .share-xiaohongshu svg {
    color: #888;
    transition: color 0.2s;
}
.news-social-share .share-xiaohongshu:hover svg {
    color: #ff2442;
}
</style>

<div class="news-controls">
    <div class="control-item">
        <select id="newsFilter" onchange="filterNews()">
            <option value="all">All Categories</option>
            {% assign categories = site.data.news | map: "category" | uniq %}
            {% for category in categories %}
            <option value="{{ category }}">{{ category | capitalize }}</option>
            {% endfor %}
        </select>
    </div>
    <div class="control-item">
        <select id="newsSort" onchange="sortNews()">
            <option value="date-desc" selected>Latest First</option>
            <option value="date-asc">Oldest First</option>
        </select>
    </div>
    <div class="control-item">
        <input type="text" id="newsSearch" placeholder="Search news..." onkeyup="searchNews()">
    </div>
</div>

<div class="news-list" id="newsList">
    {% assign sorted_news = site.data.news | sort: "date" | reverse %}
    {% for item in sorted_news %}
    <div class="news-card" data-category="{{ item.category }}" id="{{ item.title | slugify }}">
        <span class="news-tag {{ item.category }}">{{ item.category | capitalize }}</span>
        <div class="news-info">
            <span><b>Date:</b> {{ item.date | date: "%B %-d, %Y" }}</span>
            {% if item.location %}
            <span><b>Location:</b> {{ item.location }}</span>
            {% endif %}
        </div>
        {% if item.title %}
        <div class="news-title">
            {{ item.title }}
        </div>
        {% endif %}
        {% if item.content %}
        <div class="news-content">
            {{ item.content }}
        </div>
        {% endif %}
        {% if item.description %}
        <ul class="news-points">
            {% assign description_lines = item.description | newline_to_br | split: '<br />' %}
            {% for line in description_lines %}
            {% assign trimmed_line = line | strip %}
            {% if trimmed_line != '' %}
            <li>{{ trimmed_line | remove: '- ' }}</li>
            {% endif %}
            {% endfor %}
        </ul>
        {% endif %}
        <div class="news-social-share">
            <a href="https://twitter.com/intent/tweet?text={{ item.title | url_encode }}&url={{ site.url }}{{ page.url }}%23{{ item.title | slugify }}" class="share-twitter" title="Share on Twitter" target="_blank" onclick="window.open(this.href, 'twitter-share', 'width=550,height=235');return false;"><i class="fab fa-twitter"></i></a>
            <a href="https://www.linkedin.com/shareArticle?mini=true&url={{ site.url }}{{ page.url }}%23{{ item.title | slugify }}&title={{ item.title | url_encode }}" class="share-linkedin" title="Share on LinkedIn" target="_blank" onclick="window.open(this.href, 'linkedin-share', 'width=550,height=435');return false;"><i class="fab fa-linkedin"></i></a>
            <a href="https://www.facebook.com/sharer/sharer.php?u={{ site.url }}{{ page.url }}%23{{ item.title | slugify }}" class="share-facebook" title="Share on Facebook" target="_blank" onclick="window.open(this.href, 'facebook-share', 'width=550,height=435');return false;"><i class="fab fa-facebook"></i></a>
            <a href="javascript:void(0);" class="share-wechat" title="Share on WeChat" onclick="showWeChatQR('{{ site.url }}{{ page.url }}%23{{ item.title | slugify }}');"><i class="fab fa-weixin"></i></a>
            <a href="javascript:void(0);" class="share-xiaohongshu" title="Share on RED" onclick="shareToXiaohongshu('{{ site.url }}{{ page.url }}%23{{ item.title | slugify }}', '{{ item.title }}');">
                <svg viewBox="0 0 40 40" width="1em" height="1em" fill="currentColor">
                    <rect x="0" y="0" width="40" height="40" rx="8" fill="currentColor"/>
                    <text x="50%" y="56%" text-anchor="middle" fill="#fff" font-size="16" font-family="Arial" dy=".3em" font-weight="bold" letter-spacing="1">RED</text>
                </svg>
            </a>
        </div>
    </div>
    {% endfor %}
</div>

<div class="pagination"></div>

<script>
const ITEMS_PER_PAGE = 5;
let currentPage = 1;
let allNews = Array.from(document.querySelectorAll('.news-card'));

function renderNews() {
    const list = document.getElementById('newsList');
    list.innerHTML = '';
    let filtered = allNews.filter(item => item.style.display !== 'none');
    const start = (currentPage - 1) * ITEMS_PER_PAGE;
    const end = start + ITEMS_PER_PAGE;
    filtered.slice(start, end).forEach(item => list.appendChild(item));
}

function filterNews() {
    const filter = document.getElementById('newsFilter').value;
    const search = document.getElementById('newsSearch').value.toLowerCase();
    allNews.forEach(item => {
        const category = item.getAttribute('data-category');
        const content = item.querySelector('.news-title').textContent.toLowerCase();
        if ((filter === 'all' || category === filter) && content.includes(search)) {
            item.style.display = '';
        } else {
            item.style.display = 'none';
        }
    });
    currentPage = 1;
    updatePagination();
    renderNews();
}

function sortNews() {
    const sort = document.getElementById('newsSort').value;
    allNews.sort((a, b) => {
        const dateA = new Date(a.querySelector('.news-info span').textContent.split(': ')[1]);
        const dateB = new Date(b.querySelector('.news-info span').textContent.split(': ')[1]);
        return sort === 'date-desc' ? dateB - dateA : dateA - dateB;
    });
    currentPage = 1;
    renderNews();
}

function searchNews() {
    filterNews();
}

function updatePagination() {
    const visible = allNews.filter(item => item.style.display !== 'none');
    const totalPages = Math.ceil(visible.length / ITEMS_PER_PAGE);
    const pagination = document.querySelector('.pagination');
    pagination.innerHTML = '';
    if (totalPages > 1) {
        const prev = document.createElement('button');
        prev.className = 'pagination-button';
        prev.innerHTML = '<i class="fas fa-chevron-left"></i> Previous';
        prev.disabled = currentPage === 1;
        prev.onclick = () => { if(currentPage>1){currentPage--; renderNews(); updatePagination();} };
        pagination.appendChild(prev);
        for (let i = 1; i <= totalPages; i++) {
            const btn = document.createElement('button');
            btn.className = 'pagination-button' + (i === currentPage ? ' active' : '');
            btn.textContent = i;
            btn.onclick = () => { currentPage = i; renderNews(); updatePagination(); };
            pagination.appendChild(btn);
        }
        const next = document.createElement('button');
        next.className = 'pagination-button';
        next.innerHTML = 'Next <i class="fas fa-chevron-right"></i>';
        next.disabled = currentPage === totalPages;
        next.onclick = () => { if(currentPage<totalPages){currentPage++; renderNews(); updatePagination();} };
        pagination.appendChild(next);
    }
}

function updateEnhancedCalendarIcons() {
    document.querySelectorAll('.news-card').forEach(card => {
        const dateText = card.querySelector('.news-info span').textContent.split(': ')[1];
        const cal = card.querySelector('.calendar-icon.enhanced');
        if (!cal) return;
        // 提取月份和日数字
        const match = dateText.match(/([A-Za-z]+)\s(\d{1,2}),\s(\d{4})/);
        let month = '', day = '';
        if (match) {
            month = match[1].toUpperCase();
            day = match[2];
        }
        cal.querySelector('.calendar-month').textContent = month;
        cal.querySelector('.calendar-day').textContent = day;
        // 彩条颜色与类别呼应
        const cat = card.getAttribute('data-category');
        let color = '#6c63ff';
        if (cat === 'site') color = '#0984e3';
        if (cat === 'project') color = '#00b894';
        if (cat === 'blog') color = '#fdcb6e';
        cal.querySelector('.calendar-month').style.background = color;
        // Pill标签颜色
        const pill = card.querySelector('.news-tag');
        if (pill) pill.style.borderColor = color;
        if (pill && cat === 'blog') pill.style.color = '#333';
    });
}

function shareToXiaohongshu(url, title) {
    // 由于小红书不提供直接分享链接，我们可以复制内容到剪贴板
    const text = `${title}\n${url}`;
    navigator.clipboard.writeText(text).then(() => {
        alert('Content copied! You can now paste it to Xiaohongshu.');
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}

function showWeChatQR(url) {
    var qr = document.createElement('div');
    qr.style.position = 'fixed';
    qr.style.left = '0';
    qr.style.top = '0';
    qr.style.width = '100vw';
    qr.style.height = '100vh';
    qr.style.background = 'rgba(0,0,0,0.5)';
    qr.style.display = 'flex';
    qr.style.alignItems = 'center';
    qr.style.justifyContent = 'center';
    qr.style.zIndex = '9999';
    qr.innerHTML = `
        <div style='background:#fff;padding:2em 2em 1em 2em;border-radius:12px;text-align:center;position:relative;'>
            <div style='font-size:1.1em;margin-bottom:0.7em;'>Scan QR Code to Share</div>
            <div id="qrcode"></div>
            <div style='margin-top:0.7em;'>
                <button onclick='this.parentNode.parentNode.parentNode.remove()' style='padding:0.4em 1.2em;border-radius:6px;border:1px solid #bbb;background:#f5f5f5;cursor:pointer;'>Close</button>
            </div>
        </div>
    `;
    document.body.appendChild(qr);
    
    // 使用 QRCode.js 生成二维码
    new QRCode(qr.querySelector('#qrcode'), {
        text: url,
        width: 180,
        height: 180
    });
}

document.addEventListener('DOMContentLoaded', function() {
    allNews = Array.from(document.querySelectorAll('.news-card'));
    filterNews();
    sortNews();
    updateEnhancedCalendarIcons();
    document.querySelectorAll('.news-card').forEach(function(card) {
        const title = card.querySelector('.news-title')?.textContent || document.title;
        const url = window.location.href;
        
        // 更新分享链接
        card.querySelector('.share-twitter').href = `https://twitter.com/intent/tweet?text=${encodeURIComponent(title)}&url=${encodeURIComponent(url)}`;
        card.querySelector('.share-linkedin').href = `https://www.linkedin.com/shareArticle?mini=true&url=${encodeURIComponent(url)}&title=${encodeURIComponent(title)}`;
        card.querySelector('.share-facebook').href = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`;
    });
});
</script>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css"> 