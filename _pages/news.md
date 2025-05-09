---
layout: page
title: News
permalink: /news/
---

<style>
.news-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 40px;
}

.news-section {
    padding: 20px;
}

.news-section-title {
    font-size: 1.8rem;
    color: #2c5282;
    margin-bottom: 2rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
}

.news-item {
    margin-bottom: 2.5rem;
    padding: 1.5rem;
    background: linear-gradient(to right, rgba(66, 153, 225, 0.05), transparent);
    border-radius: 8px;
    border-left: 4px solid #4299e1;
}

.news-date {
    font-size: 1.2rem;
    color: #2c5282;
    font-weight: 600;
    margin-bottom: 1rem;
}

.news-content {
    font-size: 1.1rem;
    line-height: 1.8;
    color: #2d3748;
}

.news-link {
    color: #3182ce;
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: all 0.3s ease;
}

.news-link:hover {
    border-bottom-color: currentColor;
}

@media (max-width: 768px) {
    .news-container {
        grid-template-columns: 1fr;
        gap: 20px;
        padding: 15px;
    }
    
    .news-section {
        padding: 15px;
    }
    
    .news-section-title {
        font-size: 1.5rem;
    }
    
    .news-item {
        padding: 1rem;
    }
    
    .news-date {
        font-size: 1.1rem;
    }
    
    .news-content {
        font-size: 1rem;
    }
}
</style>

<div class="news-container">
    <div class="news-section">
        <h2 class="news-section-title">Website Updates</h2>
        <div class="news-item">
            <div class="news-date">March 2024</div>
            <div class="news-content">
                Launched my academic website, featuring research interests in neuroimaging, machine learning, and brain connectivity analysis. The website showcases my academic background, research experience, and technical expertise.
            </div>
        </div>
    </div>
    
    <div class="news-section">
        <h2 class="news-section-title">Personal Updates</h2>
        <div class="news-item">
            <div class="news-date">March 2024</div>
            <div class="news-content">
                Started working on a new research project focusing on deep learning applications in EEG signal processing, exploring novel approaches for brain signal analysis and interpretation.
            </div>
        </div>
    </div>
</div> 