---
layout: page
title: News
permalink: /news/
---

<style>
.news-controls {
    display: flex;
    gap: 1.2rem;
    margin-bottom: 2.5rem;
    flex-wrap: wrap;
    background: var(--bg-color-secondary);
    padding: 1.4rem;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.control-item {
    flex: 1;
    min-width: 180px;
}
.control-item select, .control-item input {
    width: 100%;
    padding: 0.8rem;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-color);
    color: var(--text-color);
    font-size: 1rem;
    transition: all 0.3s;
    font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
}
.control-item select:hover, .control-item input:hover {
    border-color: #999;
}
.control-item select:focus, .control-item input:focus {
    outline: none;
    border-color: var(--accent-color);
    box-shadow: 0 0 0 3px rgba(var(--accent-color-rgb), 0.1);
}
.news-list {
    display: grid;
    grid-template-columns: 1fr;
    gap: 2.5rem;
    margin: 2.5rem 0;
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}
.news-card {
    background: var(--bg-color-secondary);
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    position: relative;
    transition: all 0.3s ease;
    overflow: hidden;
    border-left: 6px solid transparent;
    padding: 0;
}
.news-card:hover {
    box-shadow: 0 5px 20px rgba(0,0,0,0.12);
    transform: translateY(-3px);
}
.news-card[data-category="personal"] {
    border-left-color: #6c63ff;
}
.news-card[data-category="site"] {
    border-left-color: #0984e3;
}
.news-card[data-category="project"] {
    border-left-color: #00b894;
}
.news-card[data-category="blog"] {
    border-left-color: #fdcb6e;
}
.news-card[data-category="research"] {
    border-left-color: #e84393;
}
.news-tag {
    display: inline-block;
    font-size: 0.85em;
    font-weight: 600;
    padding: 0.4em 1em;
    border-radius: 4px;
    color: #fff;
    font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
    letter-spacing: 0.6px;
    margin: 1.5rem 0 0 1.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.news-tag.personal { background-color: #6c63ff; }
.news-tag.project { background-color: #00b894; }
.news-tag.site { background-color: #0984e3; }
.news-tag.blog { background-color: #fdcb6e; color: #333; }
.news-tag.research { background-color: #e84393; }
.news-info {
    font-size: 0.95em;
    color: #666;
    margin: 1.2rem 1.5rem;
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    font-family: "Times New Roman", Times, serif;
    line-height: 1.5;
}
.news-info span { 
    display: inline-flex;
    align-items: center;
    min-width: 180px;
}
.news-info b {
    font-weight: 600;
    color: #333;
    margin-right: 0.5em;
}
.news-title {
    font-size: 1.5em;
    font-weight: 600;
    margin: 0.6rem 1.5rem 1.5rem 1.5rem;
    font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
    letter-spacing: 0.02em;
    line-height: 1.4;
    color: #333;
    border-bottom: 1px solid #eaeaea;
    padding-bottom: 0.8em;
}
html.dark .news-title {
    color: #e2e8f0 !important;
    border-bottom-color: #4a5568;
}
.news-points {
    margin: 0 1.5rem 1.8rem 1.5rem;
    padding-left: 2.5em;
    font-size: 1.08em;
    color: var(--text-color);
    text-align: justify;
    line-height: 1.75;
    font-family: "Times New Roman", Times, serif;
    letter-spacing: 0.01em;
    list-style-type: none;
}
.news-points li {
    margin-bottom: 1em;
    position: relative;
    text-align: justify;
    padding-right: 0.5em;
}
.news-points li:last-child {
    margin-bottom: 0;
}
.news-points li::before {
    content: "•";
    color: #8b0000;
    font-weight: bold;
    position: absolute;
    left: -1.5em;
    font-size: 1.2em;
}
.news-points li a {
    text-decoration: none;
    color: #4a6da7;
    border-bottom: 1px dotted #4a6da7;
    transition: all 0.2s ease;
}
.news-points li a:hover {
    color: #6c63ff;
    border-bottom-color: #6c63ff;
}
html.dark .news-points li a {
    color: #81a1d6;
    border-bottom-color: #81a1d6;
}
html.dark .news-points li a:hover {
    color: #a3beff;
    border-bottom-color: #a3beff;
}
.news-social-share {
    display: flex;
    gap: 1em;
    justify-content: flex-end;
    border-top: 1px solid rgba(0,0,0,0.05);
    margin: 0 1.5rem;
    padding: 1.5rem 0;
}
.news-social-share a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.2em;
    height: 2.2em;
    border-radius: 50%;
    background-color: rgba(0,0,0,0.05);
    color: #666;
    transition: all 0.2s;
}
html.dark .news-social-share a {
    background-color: rgba(255,255,255,0.1);
    color: #a0aec0;
}
.news-social-share a:hover {
    transform: translateY(-3px);
}
.news-social-share a.share-twitter:hover { 
    background-color: #1da1f2; 
    color: white;
}
.news-social-share a.share-linkedin:hover { 
    background-color: #0077b5; 
    color: white;
}
.news-social-share a.share-facebook:hover { 
    background-color: #1877f3; 
    color: white;
}
.news-social-share a.share-wechat:hover { 
    background-color: #09bb07; 
    color: white;
}
.news-social-share a.share-xiaohongshu:hover { 
    background-color: #ff2442; 
    color: white;
}
.news-social-share a.share-xiaohongshu svg {
    width: 1em;
    height: 1em;
}
@media (max-width: 768px) {
    .news-controls { flex-direction: column; padding: 1.2rem; }
    .news-card { border-radius: 6px; }
    .news-tag { margin: 1.2rem 0 0 1.2rem; font-size: 0.8em; }
    .news-info { margin: 1rem 1.2rem; gap: 1rem; font-size: 0.9em; }
    .news-title { margin: 0.5rem 1.2rem 1.2rem 1.2rem; font-size: 1.3em; }
    .news-points { margin: 0 1.2rem 1.5rem 1.2rem; font-size: 1.02em; line-height: 1.7; padding-left: 2em; }
    .news-points li::before { left: -1.3em; }
    .news-social-share { margin: 0 1.2rem; padding: 1.2rem 0; }
}

/* 翻页控件样式 */
.pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 15px;
    margin: 30px 0;
}

.pagination-btn {
    padding: 8px 15px;
    border: none;
    background-color: var(--accent-color, #6c63ff);
    color: white;
    border-radius: 5px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 5px;
    transition: all 0.3s ease;
}

.pagination-btn:hover:not(:disabled) {
    background-color: var(--accent-color-dark, #5a52d5);
    transform: translateY(-2px);
}

.pagination-btn:disabled {
    background-color: #ccc;
    cursor: not-allowed;
}

.page-numbers {
    font-size: 1em;
    color: var(--text-color, #333);
}

@media (max-width: 768px) {
    .pagination {
        gap: 10px;
    }
    
    .pagination-btn {
        padding: 6px 10px;
        font-size: 0.9em;
    }
}
</style>

<div class="news-controls">
    <div class="control-item">
        <select id="newsFilter" onchange="filterNews()">
            <option value="all">All Categories</option>
            <option value="personal">Personal</option>
            <option value="project">Project</option>
            <option value="site">Site</option>
            <option value="blog">Blog</option>
            <option value="research">Research</option>
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
    <div class="news-card" data-category="personal" data-timestamp="20250523">
        <span class="news-tag personal">Personal</span>
        <div class="news-info">
            <span><b>Date:</b> May 23, 2025</span>
            <span><b>Location:</b> Ground floor, Leopoldstraße 30, 80802 München</span>
        </div>
        <div class="news-title">
            PhD Basics for International Doctoral Researchers
        </div>
  
        <ul class="news-points">
    
            <li>I will attend the event <a href="https://www.portal.graduatecenter.lmu.de/gc/de/phd_basics_internationals_2025" target="_blank">PhD Basics for International Doctoral Researchers</a> at LMU Munich.</li>
      
            <li>This peer-to-peer event addresses the particular challenges most international doctoral candidates must overcome, including communication with supervisors and adapting to academic culture in Germany.</li>
      
        </ul>
    
        <div class="news-social-share">
            <a href="#" class="share-twitter" title="Share on Twitter" target="_blank"><i class="fab fa-twitter"></i></a>
            <a href="#" class="share-linkedin" title="Share on LinkedIn" target="_blank"><i class="fab fa-linkedin"></i></a>
            <a href="#" class="share-facebook" title="Share on Facebook" target="_blank"><i class="fab fa-facebook"></i></a>
            <a href="#" class="share-wechat" title="Share on WeChat"><i class="fab fa-weixin"></i></a>
            <a href="#" class="share-xiaohongshu" title="Share on RED" target="_blank">
                <svg viewBox="0 0 40 40" width="1em" height="1em" fill="currentColor">
                    <rect x="0" y="0" width="40" height="40" rx="8" fill="currentColor"/>
                    <text x="50%" y="56%" text-anchor="middle" fill="#fff" font-size="16" font-family="Arial" dy=".3em" font-weight="bold" letter-spacing="1">RED</text>
                </svg>
            </a>
        </div>
    </div>
  
    <div class="news-card" data-category="site" data-timestamp="20250508">
        <span class="news-tag site">Site</span>
        <div class="news-info">
            <span><b>Date:</b> May 8, 2025</span>
            
        </div>
        <div class="news-title">
            Website Created
        </div>
  
        <ul class="news-points">
    
            <li>This academic website was created on May 8, 2025.</li>
      
            <li>Source code and updates are available on <a href="https://github.com/JunlinJing/jjing-neuro" target="_blank">GitHub</a>.</li>
      
        </ul>
    
        <div class="news-social-share">
            <a href="#" class="share-twitter" title="Share on Twitter" target="_blank"><i class="fab fa-twitter"></i></a>
            <a href="#" class="share-linkedin" title="Share on LinkedIn" target="_blank"><i class="fab fa-linkedin"></i></a>
            <a href="#" class="share-facebook" title="Share on Facebook" target="_blank"><i class="fab fa-facebook"></i></a>
            <a href="#" class="share-wechat" title="Share on WeChat"><i class="fab fa-weixin"></i></a>
            <a href="#" class="share-xiaohongshu" title="Share on RED" target="_blank">
                <svg viewBox="0 0 40 40" width="1em" height="1em" fill="currentColor">
                    <rect x="0" y="0" width="40" height="40" rx="8" fill="currentColor"/>
                    <text x="50%" y="56%" text-anchor="middle" fill="#fff" font-size="16" font-family="Arial" dy=".3em" font-weight="bold" letter-spacing="1">RED</text>
                </svg>
            </a>
        </div>
    </div>
  
    <div class="news-card" data-category="personal" data-timestamp="20241001">
        <span class="news-tag personal">Personal</span>
        <div class="news-info">
            <span><b>Date:</b> October 1, 2024</span>
            <span><b>Location:</b> Munich, Germany</span>
        </div>
        <div class="news-title">
            Joined Functional Neuroimaging Lab at LMU Munich
        </div>
  
        <ul class="news-points">
    
            <li>Started Ph.D. research in Neuroimaging and Machine Learning under the supervision of Prof. Sophia Stöcklein</li>
      
            <li>Will focus on developing advanced computational methods for neuroimaging data analysis</li>
      
        </ul>
    
        <div class="news-social-share">
            <a href="#" class="share-twitter" title="Share on Twitter" target="_blank"><i class="fab fa-twitter"></i></a>
            <a href="#" class="share-linkedin" title="Share on LinkedIn" target="_blank"><i class="fab fa-linkedin"></i></a>
            <a href="#" class="share-facebook" title="Share on Facebook" target="_blank"><i class="fab fa-facebook"></i></a>
            <a href="#" class="share-wechat" title="Share on WeChat"><i class="fab fa-weixin"></i></a>
            <a href="#" class="share-xiaohongshu" title="Share on RED" target="_blank">
                <svg viewBox="0 0 40 40" width="1em" height="1em" fill="currentColor">
                    <rect x="0" y="0" width="40" height="40" rx="8" fill="currentColor"/>
                    <text x="50%" y="56%" text-anchor="middle" fill="#fff" font-size="16" font-family="Arial" dy=".3em" font-weight="bold" letter-spacing="1">RED</text>
                </svg>
            </a>
        </div>
    </div>
  </div>


<script>
    document.addEventListener('DOMContentLoaded', function() {
        // Initialize
        updateNewsCount();
    });

    function filterNews() {
        const category = document.getElementById('newsFilter').value;
        const cards = document.querySelectorAll('.news-card');
        
        cards.forEach(card => {
            if (category === 'all' || card.dataset.category === category) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
        
        updateNewsCount();
    }

    function sortNews() {
        const sortOrder = document.getElementById('newsSort').value;
        const newsContainer = document.getElementById('newsList');
        const cards = Array.from(document.querySelectorAll('.news-card'));
        
        cards.sort((a, b) => {
            const aTime = parseInt(a.dataset.timestamp || '0');
            const bTime = parseInt(b.dataset.timestamp || '0');
            
            if (sortOrder === 'date-desc') {
                return bTime - aTime; // Newest first
            } else {
                return aTime - bTime; // Oldest first
            }
        });
        
        // Re-append cards in the sorted order
        cards.forEach(card => newsContainer.appendChild(card));
    }

    function searchNews() {
        const searchText = document.getElementById('newsSearch').value.toLowerCase();
        const cards = document.querySelectorAll('.news-card');
        
        cards.forEach(card => {
            const cardText = card.textContent.toLowerCase();
            if (cardText.includes(searchText)) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
        
        updateNewsCount();
    }

    function updateNewsCount() {
        const visibleCards = document.querySelectorAll('.news-card[style="display: block;"], .news-card:not([style*="display"])').length;
        const countElem = document.getElementById('newsCount');
        if (countElem) {
            countElem.textContent = visibleCards;
        }
    }
</script>
