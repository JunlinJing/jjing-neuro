---
layout: page
title: News
permalink: /news/
---

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
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    padding: 1.5rem 1.5rem 1rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    border-left: 4px solid var(--accent-color);
    transition: box-shadow 0.2s;
}
.news-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.10);
}
.news-date {
    font-size: 1rem;
    color: var(--meta-color);
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.news-date:before {
    content: "\1F4C5";
    font-size: 1rem;
}
.news-content {
    font-size: 1.05rem;
    line-height: 1.7;
    color: var(--text-color);
}
.news-tag {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 4px;
    font-size: 0.92rem;
    font-weight: 600;
    margin-right: 0.7rem;
    background: var(--accent-color);
    color: #fff;
    letter-spacing: 0.5px;
}
.news-tag.personal { background: #6c63ff; }
.news-tag.project { background: #00b894; }
.news-tag.site { background: #0984e3; }
.news-tag.blog { background: #fdcb6e; color: #333; }
.news-tools {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.tool-button {
    padding: 0.35rem 0.8rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: transparent;
    color: #666;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.92rem;
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
    .news-card { padding: 1rem; }
}
.calendar-icon {
  display: inline-block;
  width: 1.8em;
  height: 1.8em;
  background: #fff;
  color: #0984e3;
  border-radius: 0.3em;
  font-weight: bold;
  text-align: center;
  line-height: 1.8em;
  margin-right: 0.5em;
  font-family: system-ui, sans-serif;
  border: 1.5px solid #bbb;
  box-shadow: 0 1px 2px rgba(0,0,0,0.07);
  font-size: 1.1em;
  vertical-align: middle;
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
    <div class="news-card" data-category="site">
        <span class="news-tag site">Site</span>
        <div class="news-date"><span class="calendar-icon"></span>May 8, 2025</div>
        <div class="news-content">
            This academic website was created on May 8, 2025.<br>
            Source code and updates are available on <a href="https://github.com/JunlinJing/jjing-neuro" target="_blank">GitHub</a>.
        </div>
        <div class="news-tools">
            <button class="tool-button" onclick="shareNews(this)" title="Share"><i class="fas fa-share-alt"></i>Share</button>
        </div>
    </div>
    <div class="news-card" data-category="personal">
        <span class="news-tag personal">Personal</span>
        <div class="news-date"><span class="calendar-icon"></span>May 2025</div>
        <div class="news-content">
            I will attend the event <a href="https://www.portal.graduatecenter.lmu.de/gc/de/phd_basics_internationals_2025" target="_blank">PhD Basics for International Doctoral Researchers</a> at LMU Munich.<br>
            This peer-to-peer event addresses the particular challenges most international doctoral candidates must overcome, including communication with supervisors and adapting to academic culture in Germany.<br>
            <b>Location:</b> Ground floor, Leopoldstraße 30, 80802 München.
        </div>
        <div class="news-tools">
            <button class="tool-button" onclick="shareNews(this)" title="Share"><i class="fas fa-share-alt"></i>Share</button>
        </div>
    </div>
    <div class="news-card" data-category="project">
        <span class="news-tag project">Project</span>
        <div class="news-date">March 2024</div>
        <div class="news-content">Started a new research project on deep learning applications in EEG signal processing.</div>
        <div class="news-tools">
            <button class="tool-button" onclick="shareNews(this)" title="Share"><i class="fas fa-share-alt"></i>Share</button>
        </div>
    </div>
    <div class="news-card" data-category="blog">
        <span class="news-tag blog">Blog</span>
        <div class="news-date">March 2024</div>
        <div class="news-content">Published new article: "Understanding Brain Connectivity Through Graph Neural Networks"</div>
        <div class="news-tools">
            <button class="tool-button" onclick="shareNews(this)" title="Share"><i class="fas fa-share-alt"></i>Share</button>
        </div>
    </div>
    <div class="news-card" data-category="personal">
        <span class="news-tag personal">Personal</span>
        <div class="news-date">February 2024</div>
        <div class="news-content">Presented research findings at the International Conference on Neural Engineering.</div>
        <div class="news-tools">
            <button class="tool-button" onclick="shareNews(this)" title="Share"><i class="fas fa-share-alt"></i>Share</button>
        </div>
    </div>
    <div class="news-card" data-category="project">
        <span class="news-tag project">Project</span>
        <div class="news-date">February 2024</div>
        <div class="news-content">Released beta version of BrainConnect toolkit for neural data analysis.</div>
        <div class="news-tools">
            <button class="tool-button" onclick="shareNews(this)" title="Share"><i class="fas fa-share-alt"></i>Share</button>
        </div>
    </div>
    <div class="news-card" data-category="blog">
        <span class="news-tag blog">Blog</span>
        <div class="news-date">January 2024</div>
        <div class="news-content">Published tutorial series: "Getting Started with Neural Data Analysis"</div>
        <div class="news-tools">
            <button class="tool-button" onclick="shareNews(this)" title="Share"><i class="fas fa-share-alt"></i>Share</button>
        </div>
    </div>
    <div class="news-card" data-category="personal">
        <span class="news-tag personal">Personal</span>
        <div class="news-date">January 2024</div>
        <div class="news-content">Joined the Computational Neuroscience Research Group as a visiting researcher.</div>
        <div class="news-tools">
            <button class="tool-button" onclick="shareNews(this)" title="Share"><i class="fas fa-share-alt"></i>Share</button>
        </div>
    </div>
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
        const content = item.querySelector('.news-content').textContent.toLowerCase();
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
        const dateA = new Date(a.querySelector('.news-date').textContent);
        const dateB = new Date(b.querySelector('.news-date').textContent);
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

function updateCalendarIcons() {
    document.querySelectorAll('.news-date').forEach(dateEl => {
        const cal = dateEl.querySelector('.calendar-icon');
        if (!cal) return;
        const text = dateEl.textContent;
        let day = '';
        const match = text.match(/\b(\d{1,2})[\,\s]/);
        if (match) {
            day = match[1];
        } else {
            const monthMatch = text.match(/([A-Za-z]+)/);
            day = monthMatch ? monthMatch[1][0] : '?';
        }
        cal.textContent = day;
    });
}

document.addEventListener('DOMContentLoaded', function() {
    allNews = Array.from(document.querySelectorAll('.news-card'));
    filterNews();
    sortNews();
    updateCalendarIcons();
});
</script>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"> 