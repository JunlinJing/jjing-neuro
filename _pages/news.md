---
layout: page
title: News
permalink: /news/
---

<style>
.page-content .news-wrapper {
    display: flex;
    flex-wrap: wrap;
    gap: 2rem;
    margin: 0 -1rem;
}

.page-content .news-column {
    flex: 1;
    min-width: 300px;
    padding: 1.5rem;
    background: #f8fafc;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.page-content .news-column-title {
    font-size: 1.8rem;
    color: #1a365d;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #4299e1;
    text-align: center;
}

.page-content .news-item {
    margin-bottom: 1.5rem;
    padding: 1.5rem;
    background: white;
    border-radius: 8px;
    border-left: 4px solid #4299e1;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.page-content .news-date {
    font-size: 1.2rem;
    color: #2c5282;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
}

.page-content .news-content {
    font-size: 1.1rem;
    line-height: 1.8;
    color: #2d3748;
}

@media (max-width: 768px) {
    .page-content .news-column {
        flex: 100%;
    }
    
    .page-content .news-column-title {
        font-size: 1.5rem;
    }
    
    .page-content .news-item {
        padding: 1rem;
    }
    
    .page-content .news-date {
        font-size: 1.1rem;
    }
    
    .page-content .news-content {
        font-size: 1rem;
    }
}
</style>

<div class="news-wrapper">
    <div class="news-column">
        <h2 class="news-column-title">Website Updates</h2>
        <div class="news-item">
            <div class="news-date">March 2024</div>
            <div class="news-content">
                Launched my academic website, featuring research interests in neuroimaging, machine learning, and brain connectivity analysis. The website showcases my academic background, research experience, and technical expertise.
            </div>
        </div>
    </div>
    
    <div class="news-column">
        <h2 class="news-column-title">Personal Updates</h2>
        <div class="news-item">
            <div class="news-date">March 2024</div>
            <div class="news-content">
                Started working on a new research project focusing on deep learning applications in EEG signal processing, exploring novel approaches for brain signal analysis and interpretation.
            </div>
        </div>
    </div>
</div> 