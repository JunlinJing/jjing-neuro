---
layout: page
title: News
permalink: /news/
---

<style>
/* Base Styles */
body .page-content {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

/* Controls Section */
.news-controls {
    display: flex !important;
    gap: 1rem !important;
    margin-bottom: 2rem !important;
    flex-wrap: wrap !important;
    background: var(--bg-color-secondary) !important;
    padding: 1rem !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

.control-item {
    flex: 1 !important;
    min-width: 200px !important;
}

.control-item select, .control-item input {
    width: 100% !important;
    padding: 0.75rem !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    background: var(--bg-color) !important;
    color: var(--text-color) !important;
    font-size: 0.95rem !important;
}

/* News Grid */
.news-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 2rem !important;
    margin: 2rem 0 !important;
}

.news-section {
    background: var(--bg-color-secondary) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    padding: 2rem !important;
    position: relative !important;
}

.section-title {
    font-size: 1.8rem !important;
    color: var(--heading-color) !important;
    margin-bottom: 2rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 3px solid var(--accent-color) !important;
    text-align: center !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.5rem !important;
}

/* News Items */
.news-item {
    background: var(--bg-color) !important;
    border-radius: 8px !important;
    border-left: 4px solid var(--accent-color) !important;
    padding: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    transition: all 0.3s ease !important;
    position: relative !important;
}

.news-item:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
}

.news-date {
    font-size: 1.2rem !important;
    color: var(--meta-color) !important;
    font-weight: 600 !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.news-content {
    font-size: 1.1rem !important;
    line-height: 1.8 !important;
    color: var(--text-color) !important;
}

.news-tag {
    display: inline-block !important;
    padding: 0.3rem 0.8rem !important;
    background: var(--accent-color) !important;
    color: white !important;
    border-radius: 4px !important;
    font-size: 0.9rem !important;
    margin-top: 1rem !important;
}

/* Tools Section */
.news-tools {
    display: flex !important;
    gap: 0.5rem !important;
    margin-top: 1rem !important;
    padding-top: 1rem !important;
    border-top: 1px solid var(--border-color) !important;
    justify-content: flex-end !important;
}

.tool-button {
    padding: 0.5rem 1rem !important;
    border: none !important;
    border-radius: 6px !important;
    background: transparent !important;
    color: var(--text-color) !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.25rem !important;
}

.tool-button:hover {
    background: var(--accent-color) !important;
    color: white !important;
}

/* Expand/Collapse Button */
.expand-button {
    background: var(--accent-color) !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    margin: 1rem auto !important;
    transition: all 0.3s ease !important;
}

.expand-button:hover {
    background: var(--accent-color-dark) !important;
    transform: translateY(-1px) !important;
}

.expand-button i {
    transition: transform 0.3s ease !important;
}

.expand-button.expanded i {
    transform: rotate(180deg) !important;
}

.hidden-items {
    display: none !important;
    opacity: 0 !important;
    transition: all 0.3s ease !important;
}

.hidden-items.visible {
    display: block !important;
    opacity: 1 !important;
}

/* Responsive Design */
@media (max-width: 768px) {
    .news-grid {
        grid-template-columns: 1fr !important;
    }
    
    .news-controls {
        flex-direction: column !important;
    }
    
    .control-item {
        width: 100% !important;
    }
    
    .section-title {
        font-size: 1.5rem !important;
    }
    
    .news-date {
        font-size: 1.1rem !important;
    }
    
    .news-content {
        font-size: 1rem !important;
    }
    
    .news-tools {
        flex-wrap: wrap !important;
    }
}

/* Dark Theme Support */
[data-theme="dark"] .news-controls select,
[data-theme="dark"] .news-controls input {
    background: var(--bg-color-dark) !important;
    border-color: var(--border-color-dark) !important;
}

[data-theme="dark"] .news-section {
    background: var(--bg-color-dark) !important;
}

[data-theme="dark"] .news-item {
    background: var(--bg-color-darker) !important;
}
</style>

<div class="news-controls">
    <div class="control-item">
        <select id="newsFilter" onchange="filterNews()">
            <option value="all">All Categories</option>
            <option value="website">Website Updates</option>
            <option value="personal">Personal Updates</option>
            <option value="research">Research Progress</option>
            <option value="publication">Published Papers</option>
            <option value="conference">Academic Conferences</option>
            <option value="award">Awards & Honors</option>
            <option value="collaboration">Collaborations</option>
            <option value="media">Media Coverage</option>
            <option value="teaching">Teaching Activities</option>
            <option value="outreach">Outreach</option>
        </select>
    </div>
    <div class="control-item">
        <select id="newsSort" onchange="sortNews()">
            <option value="date-desc">Latest First</option>
            <option value="date-asc">Oldest First</option>
            <option value="title">By Title</option>
            <option value="views">Most Viewed</option>
            <option value="likes">Most Liked</option>
        </select>
    </div>
    <div class="control-item">
        <input type="text" id="newsSearch" placeholder="Search news..." onkeyup="searchNews()">
    </div>
</div>

<div class="news-grid">
    <div class="news-section">
        <h2 class="section-title">
            Website Updates
            <span class="update-count">(4)</span>
        </h2>
        <div class="news-item" data-category="website">
            <div class="news-date">March 2024</div>
            <div class="news-content">
                Launched academic website, featuring research interests in neuroimaging, machine learning, and brain connectivity analysis.
            </div>
            <div class="news-tag">Update</div>
            <div class="news-tools">
                <button class="tool-button" onclick="shareNews(this)" title="Share">
                    <i class="fas fa-share-alt"></i>
                    Share
                </button>
                <button class="tool-button" onclick="downloadNews(this)" title="Download">
                    <i class="fas fa-download"></i>
                    Download
                </button>
            </div>
        </div>
        <div class="news-item" data-category="website">
            <div class="news-date">February 2024</div>
            <div class="news-content">
                Updated website with new research focus areas and improved navigation.
            </div>
            <div class="news-tag">Update</div>
            <div class="news-tools">
                <button class="tool-button" onclick="shareNews(this)" title="Share">
                    <i class="fas fa-share-alt"></i>
                    Share
                </button>
                <button class="tool-button" onclick="downloadNews(this)" title="Download">
                    <i class="fas fa-download"></i>
                    Download
                </button>
            </div>
        </div>
        <button class="expand-button" onclick="toggleSection(this)">
            Show More <i class="fas fa-chevron-down"></i>
        </button>
        <div class="hidden-items">
            <!-- More news items -->
        </div>
    </div>
    
    <div class="news-section">
        <h2 class="section-title">
            Research Progress
            <span class="update-count">(3)</span>
        </h2>
        <div class="news-item" data-category="research">
            <div class="news-date">March 2024</div>
            <div class="news-content">
                Started working on a new research project focusing on deep learning applications in EEG signal processing.
            </div>
            <div class="news-tag">Research</div>
            <div class="news-tools">
                <button class="tool-button" onclick="shareNews(this)" title="Share">
                    <i class="fas fa-share-alt"></i>
                    Share
                </button>
                <button class="tool-button" onclick="downloadNews(this)" title="Download">
                    <i class="fas fa-download"></i>
                    Download
                </button>
            </div>
        </div>
    </div>
</div>

<script>
function filterNews() {
    const filter = document.getElementById('newsFilter').value;
    const items = document.querySelectorAll('.news-item');
    
    items.forEach(item => {
        if (filter === 'all' || item.dataset.category === filter) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });
}

function sortNews() {
    const sort = document.getElementById('newsSort').value;
    const sections = document.querySelectorAll('.news-section');
    
    sections.forEach(section => {
        const items = Array.from(section.querySelectorAll('.news-item'));
        
        items.sort((a, b) => {
            const dateA = new Date(a.querySelector('.news-date').textContent);
            const dateB = new Date(b.querySelector('.news-date').textContent);
            
            if (sort === 'date-desc') {
                return dateB - dateA;
            } else if (sort === 'date-asc') {
                return dateA - dateB;
            }
            // Additional sorting logic can be added as needed
        });
        
        const container = section.querySelector('.news-item').parentNode;
        items.forEach(item => container.appendChild(item));
    });
}

function searchNews() {
    const searchText = document.getElementById('newsSearch').value.toLowerCase();
    const items = document.querySelectorAll('.news-item');
    
    items.forEach(item => {
        const content = item.querySelector('.news-content').textContent.toLowerCase();
        const date = item.querySelector('.news-date').textContent.toLowerCase();
        
        if (content.includes(searchText) || date.includes(searchText)) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });
}

function toggleSection(button) {
    const section = button.closest('.news-section');
    const hiddenItems = section.querySelector('.hidden-items');
    const isExpanded = button.classList.contains('expanded');
    
    if (isExpanded) {
        hiddenItems.classList.remove('visible');
        button.classList.remove('expanded');
        button.innerHTML = 'Show More <i class="fas fa-chevron-down"></i>';
    } else {
        hiddenItems.classList.add('visible');
        button.classList.add('expanded');
        button.innerHTML = 'Show Less <i class="fas fa-chevron-up"></i>';
    }
}

function shareNews(button) {
    const newsItem = button.closest('.news-item');
    const newsText = newsItem.querySelector('.news-content').textContent;
    const newsDate = newsItem.querySelector('.news-date').textContent;
    
    if (navigator.share) {
        navigator.share({
            title: 'Academic Website News',
            text: `${newsDate}: ${newsText}`,
            url: window.location.href
        }).catch(console.error);
    } else {
        const textToCopy = `${newsDate}: ${newsText}\n${window.location.href}`;
        navigator.clipboard.writeText(textToCopy)
            .then(() => alert('Copied to clipboard'))
            .catch(console.error);
    }
}

function downloadNews(button) {
    const newsItem = button.closest('.news-item');
    const newsText = newsItem.querySelector('.news-content').textContent;
    const newsDate = newsItem.querySelector('.news-date').textContent;
    const newsTag = newsItem.querySelector('.news-tag').textContent;
    
    const content = `Date: ${newsDate}\nContent: ${newsText}\nTag: ${newsTag}\n\nSource: ${window.location.href}`;
    const blob = new Blob([content], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `news-${newsDate}.txt`;
    a.click();
    window.URL.revokeObjectURL(url);
}
</script>

<!-- Font Awesome -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"> 