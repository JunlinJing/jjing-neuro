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
}

.control-item {
    flex: 1 !important;
    min-width: 200px !important;
}

.control-item select, .control-item input {
    width: 100% !important;
    padding: 0.75rem !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    background: white !important;
}

/* News Grid */
.news-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 2rem !important;
    margin: 2rem 0 !important;
}

.news-section {
    background: #f8fafc !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    padding: 2rem !important;
    position: relative !important;
}

.section-title {
    font-size: 1.8rem !important;
    color: #1a365d !important;
    margin-bottom: 2rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 3px solid #4299e1 !important;
    text-align: center !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.5rem !important;
}

/* News Items */
.news-item {
    background: white !important;
    border-radius: 8px !important;
    border-left: 4px solid #4299e1 !important;
    padding: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    transition: all 0.3s ease !important;
}

.news-item:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
}

.news-date {
    font-size: 1.2rem !important;
    color: #2c5282 !important;
    font-weight: 600 !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

.news-content {
    font-size: 1.1rem !important;
    line-height: 1.8 !important;
    color: #2d3748 !important;
}

/* Expand/Collapse Button */
.expand-button {
    background: #4299e1 !important;
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
    background: #2b6cb0 !important;
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

/* Tools Section */
.news-tools {
    display: flex !important;
    gap: 0.5rem !important;
    margin-top: 1rem !important;
    padding-top: 1rem !important;
    border-top: 1px solid #e2e8f0 !important;
}

.tool-button {
    padding: 0.5rem 1rem !important;
    border: none !important;
    border-radius: 6px !important;
    background: #4299e1 !important;
    color: white !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.25rem !important;
}

.tool-button:hover {
    background: #2b6cb0 !important;
    transform: translateY(-1px) !important;
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
                Launched my academic website, featuring research interests in neuroimaging, machine learning, and brain connectivity analysis.
            </div>
            <div class="news-tools">
                <button class="tool-button" onclick="shareNews(this)">
                    <i class="fas fa-share-alt"></i>
                    Share
                </button>
                <button class="tool-button" onclick="downloadNews(this)">
                    <i class="fas fa-download"></i>
                    Download
                </button>
            </div>
        </div>
        <div class="news-item" data-category="website">
            <div class="news-date">February 2024</div>
            <div class="news-content">
                Updated the website with new research focus areas and improved navigation.
            </div>
            <div class="news-tools">
                <button class="tool-button" onclick="shareNews(this)">
                    <i class="fas fa-share-alt"></i>
                    Share
                </button>
                <button class="tool-button" onclick="downloadNews(this)">
                    <i class="fas fa-download"></i>
                    Download
                </button>
            </div>
        </div>
        <div class="hidden-items">
            <div class="news-item" data-category="website">
                <div class="news-date">January 2024</div>
                <div class="news-content">
                    Added new sections for publications and research projects.
                </div>
                <div class="news-tools">
                    <button class="tool-button" onclick="shareNews(this)">
                        <i class="fas fa-share-alt"></i>
                        Share
                    </button>
                    <button class="tool-button" onclick="downloadNews(this)">
                        <i class="fas fa-download"></i>
                        Download
                    </button>
                </div>
            </div>
            <div class="news-item" data-category="website">
                <div class="news-date">December 2023</div>
                <div class="news-content">
                    Initial website setup and design implementation.
                </div>
                <div class="news-tools">
                    <button class="tool-button" onclick="shareNews(this)">
                        <i class="fas fa-share-alt"></i>
                        Share
                    </button>
                    <button class="tool-button" onclick="downloadNews(this)">
                        <i class="fas fa-download"></i>
                        Download
                    </button>
                </div>
            </div>
        </div>
        <button class="expand-button" onclick="toggleSection(this, 'website')">
            <span class="button-text">Show More</span>
            <i class="fas fa-chevron-down"></i>
        </button>
    </div>
    
    <div class="news-section">
        <h2 class="section-title">
            Personal Updates
            <span class="update-count">(4)</span>
        </h2>
        <div class="news-item" data-category="personal">
            <div class="news-date">March 2024</div>
            <div class="news-content">
                Started working on a new research project focusing on deep learning applications in EEG signal processing.
            </div>
            <div class="news-tools">
                <button class="tool-button" onclick="shareNews(this)">
                    <i class="fas fa-share-alt"></i>
                    Share
                </button>
                <button class="tool-button" onclick="downloadNews(this)">
                    <i class="fas fa-download"></i>
                    Download
                </button>
            </div>
        </div>
        <div class="news-item" data-category="personal">
            <div class="news-date">February 2024</div>
            <div class="news-content">
                Published a new paper on neural signal processing methods.
            </div>
            <div class="news-tools">
                <button class="tool-button" onclick="shareNews(this)">
                    <i class="fas fa-share-alt"></i>
                    Share
                </button>
                <button class="tool-button" onclick="downloadNews(this)">
                    <i class="fas fa-download"></i>
                    Download
                </button>
            </div>
        </div>
        <div class="hidden-items">
            <div class="news-item" data-category="personal">
                <div class="news-date">January 2024</div>
                <div class="news-content">
                    Presented research findings at the International Neuroscience Conference.
                </div>
                <div class="news-tools">
                    <button class="tool-button" onclick="shareNews(this)">
                        <i class="fas fa-share-alt"></i>
                        Share
                    </button>
                    <button class="tool-button" onclick="downloadNews(this)">
                        <i class="fas fa-download"></i>
                        Download
                    </button>
                </div>
            </div>
            <div class="news-item" data-category="personal">
                <div class="news-date">December 2023</div>
                <div class="news-content">
                    Received grant funding for upcoming research project.
                </div>
                <div class="news-tools">
                    <button class="tool-button" onclick="shareNews(this)">
                        <i class="fas fa-share-alt"></i>
                        Share
                    </button>
                    <button class="tool-button" onclick="downloadNews(this)">
                        <i class="fas fa-download"></i>
                        Download
                    </button>
                </div>
            </div>
        </div>
        <button class="expand-button" onclick="toggleSection(this, 'personal')">
            <span class="button-text">Show More</span>
            <i class="fas fa-chevron-down"></i>
        </button>
    </div>
</div>

<script>
function filterNews() {
    const category = document.getElementById('newsFilter').value;
    const items = document.querySelectorAll('.news-item');
    
    items.forEach(item => {
        if (category === 'all' || item.dataset.category === category) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });
}

function sortNews() {
    const sortBy = document.getElementById('newsSort').value;
    const sections = document.querySelectorAll('.news-section');
    
    sections.forEach(section => {
        const items = Array.from(section.querySelectorAll('.news-item'));
        items.sort((a, b) => {
            const dateA = new Date(a.querySelector('.news-date').textContent);
            const dateB = new Date(b.querySelector('.news-date').textContent);
            return sortBy === 'date-desc' ? dateB - dateA : dateA - dateB;
        });
        
        items.forEach(item => section.appendChild(item));
    });
}

function searchNews() {
    const searchTerm = document.getElementById('newsSearch').value.toLowerCase();
    const items = document.querySelectorAll('.news-item');
    
    items.forEach(item => {
        const content = item.textContent.toLowerCase();
        if (content.includes(searchTerm)) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });
}

function toggleSection(button, section) {
    const newsSection = button.closest('.news-section');
    const hiddenItems = newsSection.querySelector('.hidden-items');
    const buttonText = button.querySelector('.button-text');
    const icon = button.querySelector('i');
    
    if (hiddenItems.classList.contains('visible')) {
        hiddenItems.classList.remove('visible');
        buttonText.textContent = 'Show More';
        button.classList.remove('expanded');
    } else {
        hiddenItems.classList.add('visible');
        buttonText.textContent = 'Show Less';
        button.classList.add('expanded');
    }
}

function shareNews(button) {
    const newsItem = button.closest('.news-item');
    const content = newsItem.querySelector('.news-content').textContent;
    const date = newsItem.querySelector('.news-date').textContent;
    
    if (navigator.share) {
        navigator.share({
            title: `News Update - ${date}`,
            text: content,
            url: window.location.href
        });
    } else {
        alert('Sharing is not supported on this device');
    }
}

function downloadNews(button) {
    const newsItem = button.closest('.news-item');
    const content = newsItem.querySelector('.news-content').textContent;
    const date = newsItem.querySelector('.news-date').textContent;
    
    const blob = new Blob([`${date}\n\n${content}`], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `news-${date}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}
</script>

<!-- Font Awesome -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"> 