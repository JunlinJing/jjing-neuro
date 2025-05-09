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
    padding: 1.5rem !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
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
    transition: all 0.3s ease !important;
}

.control-item select:hover, .control-item input:hover {
    border-color: #999 !important;
}

.control-item select:focus, .control-item input:focus {
    outline: none !important;
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 2px rgba(var(--accent-color-rgb), 0.1) !important;
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
    box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    padding: 2rem !important;
    position: relative !important;
}

.section-title {
    font-size: 1.5rem !important;
    color: var(--heading-color) !important;
    margin-bottom: 2rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid var(--border-color) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
}

.update-count {
    font-size: 1rem !important;
    color: var(--meta-color) !important;
    font-weight: normal !important;
}

/* News Items */
.news-item {
    background: var(--bg-color) !important;
    border-radius: 8px !important;
    border-left: 3px solid var(--accent-color) !important;
    padding: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    transition: all 0.3s ease !important;
}

.news-item:hover {
    transform: translateX(5px) !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
}

.news-date {
    font-size: 1rem !important;
    color: var(--meta-color) !important;
    font-weight: 500 !important;
    margin-bottom: 0.75rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

.news-date:before {
    content: "📅" !important;
    font-size: 1rem !important;
}

.news-content {
    font-size: 1rem !important;
    line-height: 1.6 !important;
    color: var(--text-color) !important;
    margin-bottom: 1rem !important;
}

.news-footer {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-top: 1rem !important;
}

.news-tag {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.25rem !important;
    padding: 0.4rem 0.8rem !important;
    background: #f0f0f0 !important;
    color: #666 !important;
    border-radius: 4px !important;
    font-size: 0.9rem !important;
    border: 1px solid #ddd !important;
}

.news-tools {
    display: flex !important;
    gap: 0.5rem !important;
}

.tool-button {
    padding: 0.4rem 0.8rem !important;
    border: 1px solid #ddd !important;
    border-radius: 4px !important;
    background: transparent !important;
    color: #666 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.25rem !important;
    font-size: 0.9rem !important;
}

/* Expand/Collapse Button */
.expand-button {
    display: block;
    width: 100%;
    padding: 10px;
    margin-top: 15px;
    background-color: #f0f0f0;
    color: #333;
    border: 1px solid #ddd;
    border-radius: 5px;
    cursor: pointer;
    text-align: center;
    transition: all 0.3s ease;
    font-weight: 500;
}

.expand-button:hover {
    background-color: #e0e0e0;
    border-color: #ccc;
    transform: translateY(-2px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Dark theme support */
[data-theme="dark"] .expand-button {
    background-color: var(--accent-color);
    color: white;
    border: none;
}

[data-theme="dark"] .expand-button:hover {
    background-color: var(--accent-color-dark);
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
        font-size: 1.3rem !important;
    }
    
    .news-date {
        font-size: 0.95rem !important;
    }
    
    .news-content {
        font-size: 0.95rem !important;
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

/* Pagination Styles */
.pagination {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 0.5rem !important;
    margin: 2rem 0 !important;
    padding: 1rem !important;
}

.pagination-button {
    padding: 0.5rem 1rem !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    background: var(--bg-color) !important;
    color: var(--text-color) !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    font-size: 0.9rem !important;
    min-width: 40px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.pagination-button:hover:not(:disabled) {
    background: var(--accent-color) !important;
    color: white !important;
    border-color: var(--accent-color) !important;
    transform: translateY(-1px) !important;
}

.pagination-button.active {
    background: var(--accent-color) !important;
    color: white !important;
    border-color: var(--accent-color) !important;
    font-weight: bold !important;
}

.pagination-button:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    background: var(--bg-color-secondary) !important;
}

/* Dark theme support */
[data-theme="dark"] .pagination-button {
    background: var(--bg-color-dark) !important;
    border-color: var(--border-color-dark) !important;
}

[data-theme="dark"] .pagination-button:hover:not(:disabled),
[data-theme="dark"] .pagination-button.active {
    background: var(--accent-color) !important;
    border-color: var(--accent-color) !important;
    color: white !important;
}

.news-items {
    margin-bottom: 1rem;
}

.news-item.hidden {
    display: none;
}

.expand-button {
    display: block;
    width: 100%;
    padding: 10px;
    margin-top: 15px;
    background-color: #f0f0f0;
    color: #333;
    border: 1px solid #ddd;
    border-radius: 5px;
    cursor: pointer;
    text-align: center;
    transition: all 0.3s ease;
    font-weight: 500;
}

.expand-button:hover {
    background-color: #e0e0e0;
    border-color: #ccc;
    transform: translateY(-2px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.expand-button i {
    margin-left: 5px;
    transition: transform 0.3s ease;
}

.expand-button.expanded i {
    transform: rotate(180deg);
}

/* Dark theme support for tool buttons */
[data-theme="dark"] .tool-button {
    border-color: var(--border-color-dark);
    color: var(--text-color);
}

[data-theme="dark"] .tool-button:hover {
    background: var(--accent-color);
    color: white;
    border-color: var(--accent-color);
}

/* News tag styles */
.news-tag {
    display: inline-block;
    padding: 4px 10px;
    background: #f0f0f0;
    color: #666;
    border-radius: 4px;
    font-size: 0.9rem;
    margin-top: 1rem;
    border: 1px solid #ddd;
}

[data-theme="dark"] .news-tag {
    background: var(--accent-color);
    color: white;
    border: none;
}
</style>

<div class="news-controls">
    <div class="control-item">
        <select id="newsFilter" onchange="filterNews()">
            <option value="all">All Categories</option>
            <option value="personal">Personal</option>
            <option value="site">Site</option>
            <option value="project">Project</option>
            <option value="blog">Blog</option>
        </select>
    </div>
    <div class="control-item">
        <select id="newsSort" onchange="sortNews()">
            <option value="date-desc" selected>Latest First</option>
            <option value="date-asc">Oldest First</option>
            <option value="category">By Category</option>
        </select>
    </div>
    <div class="control-item">
        <input type="text" id="newsSearch" placeholder="Search news..." onkeyup="searchNews()">
    </div>
</div>

<div class="news-grid">
    <!-- Personal Updates Section -->
    <div class="news-section">
        <h2 class="section-title">
            Personal Updates
            <span class="update-count">(3)</span>
        </h2>
        <div class="news-items">
            <div class="news-item" data-category="personal">
                <div class="news-date">March 2024</div>
                <div class="news-content">
                    Received certification in Advanced Neural Networks and Deep Learning from Stanford Online.
                </div>
                <div class="news-tag">Personal</div>
                <div class="news-tools">
                    <button class="tool-button" onclick="shareNews(this)" title="Share">
                        <i class="fas fa-share-alt"></i>
                        Share
                    </button>
                </div>
            </div>

            <div class="news-item" data-category="personal">
                <div class="news-date">February 2024</div>
                <div class="news-content">
                    Presented research findings at the International Conference on Neural Engineering.
                </div>
                <div class="news-tag">Personal</div>
                <div class="news-tools">
                    <button class="tool-button" onclick="shareNews(this)" title="Share">
                        <i class="fas fa-share-alt"></i>
                        Share
                    </button>
                </div>
            </div>

            <div class="news-item hidden" data-category="personal">
                <div class="news-date">January 2024</div>
                <div class="news-content">
                    Joined the Computational Neuroscience Research Group as a visiting researcher.
                </div>
                <div class="news-tag">Personal</div>
                <div class="news-tools">
                    <button class="tool-button" onclick="shareNews(this)" title="Share">
                        <i class="fas fa-share-alt"></i>
                        Share
                    </button>
                </div>
            </div>
        </div>
        <button class="expand-button" onclick="toggleSection(this)">
            Show More <i class="fas fa-chevron-down"></i>
        </button>
    </div>

    <!-- Project Updates Section -->
    <div class="news-section">
        <h2 class="section-title">
            Project Updates
            <span class="update-count">(2)</span>
        </h2>
        <div class="news-items">
            <div class="news-item" data-category="project">
                <div class="news-date">March 2024</div>
                <div class="news-content">
                    Started a new research project on deep learning applications in EEG signal processing.
                </div>
                <div class="news-tag">Project</div>
                <div class="news-tools">
                    <button class="tool-button" onclick="shareNews(this)" title="Share">
                        <i class="fas fa-share-alt"></i>
                        Share
                    </button>
                </div>
            </div>

            <div class="news-item" data-category="project">
                <div class="news-date">February 2024</div>
                <div class="news-content">
                    Released beta version of BrainConnect toolkit for neural data analysis.
                </div>
                <div class="news-tag">Project</div>
                <div class="news-tools">
                    <button class="tool-button" onclick="shareNews(this)" title="Share">
                        <i class="fas fa-share-alt"></i>
                        Share
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Site Updates Section -->
    <div class="news-section">
        <h2 class="section-title">
            Site Updates
            <span class="update-count">(1)</span>
        </h2>
        <div class="news-items">
            <div class="news-item" data-category="site">
                <div class="news-date">March 2024</div>
                <div class="news-content">
                    Launched academic website, featuring research interests in neuroimaging, machine learning, and brain connectivity analysis.
                </div>
                <div class="news-tag">Site</div>
                <div class="news-tools">
                    <button class="tool-button" onclick="shareNews(this)" title="Share">
                        <i class="fas fa-share-alt"></i>
                        Share
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Blog Updates Section -->
    <div class="news-section">
        <h2 class="section-title">
            Blog Updates
            <span class="update-count">(2)</span>
        </h2>
        <div class="news-items">
            <div class="news-item" data-category="blog">
                <div class="news-date">March 2024</div>
                <div class="news-content">
                    Published new article: "Understanding Brain Connectivity Through Graph Neural Networks"
                </div>
                <div class="news-tag">Blog</div>
                <div class="news-tools">
                    <button class="tool-button" onclick="shareNews(this)" title="Share">
                        <i class="fas fa-share-alt"></i>
                        Share
                    </button>
                </div>
            </div>

            <div class="news-item hidden" data-category="blog">
                <div class="news-date">January 2024</div>
                <div class="news-content">
                    Published tutorial series: "Getting Started with Neural Data Analysis"
                </div>
                <div class="news-tag">Blog</div>
                <div class="news-tools">
                    <button class="tool-button" onclick="shareNews(this)" title="Share">
                        <i class="fas fa-share-alt"></i>
                        Share
                    </button>
                </div>
            </div>
        </div>
        <button class="expand-button" onclick="toggleSection(this)">
            Show More <i class="fas fa-chevron-down"></i>
        </button>
    </div>
</div>

<script>
const ITEMS_PER_PAGE = 5;
let currentPage = 1;

function filterNews() {
    const filter = document.getElementById('newsFilter').value;
    const items = document.querySelectorAll('.news-item');
    let visibleCount = 0;
    
    items.forEach(item => {
        if (filter === 'all' || item.dataset.category === filter) {
            item.style.display = 'block';
            visibleCount++;
        } else {
            item.style.display = 'none';
        }
    });
    
    currentPage = 1;
    updatePagination(visibleCount);
    applyPagination();
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
            } else if (sort === 'category') {
                const categoryA = a.dataset.category;
                const categoryB = b.dataset.category;
                return categoryA.localeCompare(categoryB);
            }
        });
        
        const container = section.querySelector('.news-item').parentNode;
        items.forEach(item => container.appendChild(item));
    });
    
    currentPage = 1;
    const visibleItems = document.querySelectorAll('.news-item[style*="display: block"]').length;
    updatePagination(visibleItems);
    applyPagination();
}

function searchNews() {
    const searchText = document.getElementById('newsSearch').value.toLowerCase();
    const items = document.querySelectorAll('.news-item');
    let visibleCount = 0;
    
    items.forEach(item => {
        const content = item.querySelector('.news-content').textContent.toLowerCase();
        const date = item.querySelector('.news-date').textContent.toLowerCase();
        const category = item.dataset.category.toLowerCase();
        
        if (content.includes(searchText) || 
            date.includes(searchText) || 
            category.includes(searchText)) {
            item.style.display = 'block';
            visibleCount++;
        } else {
            item.style.display = 'none';
        }
    });
    
    currentPage = 1;
    updatePagination(visibleCount);
    applyPagination();
}

function updatePagination(totalItems) {
    const totalPages = Math.ceil(totalItems / ITEMS_PER_PAGE);
    let paginationContainer = document.querySelector('.pagination');
    
    if (!paginationContainer) {
        paginationContainer = document.createElement('div');
        paginationContainer.className = 'pagination';
        document.querySelector('.news-grid').after(paginationContainer);
    }
    
    paginationContainer.innerHTML = '';
    
    if (totalPages > 1) {
        // Previous button
        const prevButton = document.createElement('button');
        prevButton.innerHTML = '<i class="fas fa-chevron-left"></i> Previous';
        prevButton.className = 'pagination-button';
        prevButton.onclick = () => changePage(currentPage - 1, totalPages);
        prevButton.disabled = currentPage === 1;
        paginationContainer.appendChild(prevButton);
        
        // Page numbers
        for (let i = 1; i <= totalPages; i++) {
            const pageButton = document.createElement('button');
            pageButton.textContent = i;
            pageButton.className = `pagination-button ${i === currentPage ? 'active' : ''}`;
            pageButton.onclick = () => changePage(i, totalPages);
            paginationContainer.appendChild(pageButton);
        }
        
        // Next button
        const nextButton = document.createElement('button');
        nextButton.innerHTML = 'Next <i class="fas fa-chevron-right"></i>';
        nextButton.className = 'pagination-button';
        nextButton.onclick = () => changePage(currentPage + 1, totalPages);
        nextButton.disabled = currentPage === totalPages;
        paginationContainer.appendChild(nextButton);
    }
}

function changePage(newPage, totalPages) {
    if (newPage >= 1 && newPage <= totalPages) {
        currentPage = newPage;
        applyPagination();
        
        // Update pagination buttons
        const paginationContainer = document.querySelector('.pagination');
        if (paginationContainer) {
            const buttons = paginationContainer.querySelectorAll('.pagination-button');
            buttons.forEach(button => {
                const pageNum = parseInt(button.textContent);
                if (!isNaN(pageNum)) {
                    if (pageNum === currentPage) {
                        button.classList.add('active');
                    } else {
                        button.classList.remove('active');
                    }
                }
            });

            // Update Previous/Next buttons
            const prevButton = paginationContainer.querySelector('.pagination-button:first-child');
            const nextButton = paginationContainer.querySelector('.pagination-button:last-child');
            
            if (prevButton) prevButton.disabled = currentPage === 1;
            if (nextButton) nextButton.disabled = currentPage === totalPages;
        }
        
        // Scroll to top of news section smoothly
        document.querySelector('.news-grid').scrollIntoView({ behavior: 'smooth' });
    }
}

function applyPagination() {
    const sections = document.querySelectorAll('.news-section');
    
    sections.forEach(section => {
        const visibleItems = Array.from(section.querySelectorAll('.news-item'));
        const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
        const endIndex = startIndex + ITEMS_PER_PAGE;
        
        visibleItems.forEach((item, index) => {
            if (index >= startIndex && index < endIndex) {
                item.style.display = 'block';
                item.classList.remove('hidden');
            } else {
                item.style.display = 'none';
                item.classList.add('hidden');
            }
        });

        // Update section count
        const totalVisible = visibleItems.length;
        const countSpan = section.querySelector('.update-count');
        if (countSpan) {
            countSpan.textContent = `(${totalVisible})`;
        }
    });
}

// Initialize pagination when page loads
document.addEventListener('DOMContentLoaded', function() {
    const totalItems = document.querySelectorAll('.news-item').length;
    currentPage = 1;
    updatePagination(totalItems);
    applyPagination();
});

function toggleSection(button) {
    const section = button.closest('.news-section');
    const items = section.querySelectorAll('.news-item.hidden');
    const isExpanded = button.classList.contains('expanded');
    
    items.forEach(item => {
        item.style.display = isExpanded ? 'none' : 'block';
    });
    
    button.classList.toggle('expanded');
    button.innerHTML = isExpanded ? 
        'Show More <i class="fas fa-chevron-down"></i>' : 
        'Show Less <i class="fas fa-chevron-up"></i>';
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