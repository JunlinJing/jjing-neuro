---
layout: page
title: News
permalink: /news/
---

<style>
.news-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 20px;
}

.news-cell {
    width: 50%;
    padding: 20px;
    vertical-align: top;
    background: #f8fafc;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.news-title {
    font-size: 1.8rem;
    color: #1a365d;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #4299e1;
    text-align: center;
}

.news-item {
    margin-bottom: 1.5rem;
    padding: 1.5rem;
    background: white;
    border-radius: 8px;
    border-left: 4px solid #4299e1;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.news-date {
    font-size: 1.2rem;
    color: #2c5282;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
}

.news-content {
    font-size: 1.1rem;
    line-height: 1.8;
    color: #2d3748;
}

@media (max-width: 768px) {
    .news-table {
        display: block;
    }
    
    .news-cell {
        display: block;
        width: 100%;
        margin-bottom: 20px;
    }
    
    .news-title {
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

<table class="news-table">
    <tr>
        <td class="news-cell">
            <h2 class="news-title">Website Updates</h2>
            <div class="news-item">
                <div class="news-date">March 2024</div>
                <div class="news-content">
                    Launched my academic website, featuring research interests in neuroimaging, machine learning, and brain connectivity analysis. The website showcases my academic background, research experience, and technical expertise.
                </div>
            </div>
        </td>
        <td class="news-cell">
            <h2 class="news-title">Personal Updates</h2>
            <div class="news-item">
                <div class="news-date">March 2024</div>
                <div class="news-content">
                    Started working on a new research project focusing on deep learning applications in EEG signal processing, exploring novel approaches for brain signal analysis and interpretation.
                </div>
            </div>
        </td>
    </tr>
</table> 