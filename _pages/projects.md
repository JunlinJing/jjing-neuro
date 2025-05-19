---
layout: page
title: Projects
permalink: /projects/
---

# Open Source Projects for Neuroscience

<div class="quote-container">
  <blockquote>
    "Open source exists because idealism exists."
    <span class="quote-author">— Jim Jing</span>
  </blockquote>
</div>

<div class="project-filters">
  <div class="filter-group">
    <label>Status:</label>
    <button class="filter-btn active" data-filter="all">All</button>
    <button class="filter-btn" data-filter="active">Active</button>
    <button class="filter-btn" data-filter="in-progress">In Progress</button>
    <button class="filter-btn" data-filter="planning">Planning</button>
  </div>
  <div class="filter-group">
    <label>Category:</label>
    <button class="filter-btn active" data-category="all">All</button>
    <button class="filter-btn" data-category="applications">Applications</button>
    <button class="filter-btn" data-category="resources">Resource Integration</button>
    <button class="filter-btn" data-category="tutorials">Tutorial Documentation</button>
    <button class="filter-btn" data-category="books">Books</button>
  </div>
</div>

<div id="projects-container">
  <div class="project-card" data-status="active" data-category="applications">
    <div class="project-header">
      <div class="project-title">
        <h3>ChatPSY: Mental Health Assessment System</h3>
      </div>
      <span class="project-type">Open Source AI Tool</span>
      <span class="project-status status-active">Active</span>
      <a href="https://github.com/JunlinJing/ChatPSY_demo" target="_blank" class="github-link">
        <i class="fab fa-github"></i> GitHub
      </a>
    </div>

    <div class="project-tags">
      <span class="tag">#MentalHealth</span>
      <span class="tag">#AI</span>
      <span class="tag">#NLP</span>
      <span class="tag">#PsychiatricScales</span>
      <span class="tag">#HealthTech</span>
    </div>

    <div class="project-description">
      An open-source AI system designed to assist mental health research and preliminary assessments, implementing standardized psychiatric scales through natural language processing.
    </div>

    <div class="project-section">
      <h4>Key Features</h4>
      <ul>
        <li><strong>Assessment:</strong> Digital implementation of psychiatric scales</li>
        <li><strong>Analysis:</strong> Automated scoring and interpretation</li>
        <li><strong>Education:</strong> Research-based mental health information</li>
        <li><strong>Guidance:</strong> Evidence-based recommendation system</li>
      </ul>
    </div>

    <div class="project-section">
      <h4>Technical Stack</h4>
      <ul>
        <li><strong>AI Integration:</strong> Open source LLM implementation</li>
        <li><strong>Framework:</strong> Langchain for context handling</li>
        <li><strong>Interface:</strong> Open source Gradio UI</li>
        <li><strong>Method:</strong> Reproducible prompt engineering</li>
      </ul>
    </div>
  </div>

  <div class="project-card" data-status="in-progress" data-category="resources">
    <div class="project-header">
      <div class="project-title">
        <h3>BrainDT (Brain-Database-Toolkit)</h3>
      </div>
      <span class="project-type">Open Source Integration Platform</span>
      <span class="project-status status-in-progress">In Progress</span>
      <a href="https://github.com/JunlinJing/BrainDT" class="github-link" target="_blank">
        <i class="fab fa-github"></i> GitHub
      </a>
    </div>

    <div class="project-tags">
      <span class="tag">#Neuroimaging</span>
      <span class="tag">#DataIntegration</span>
      <span class="tag">#Toolkits</span>
      <span class="tag">#BrainAtlas</span>
      <span class="tag">#OpenScience</span>
    </div>

    <div class="project-description">
      An open-source platform that integrates and organizes neuroscience research resources, making databases, toolkits, and brain atlases more accessible to researchers.
    </div>

    <div class="project-section">
      <h4>Key Features</h4>
      <ul>
        <li><strong>Data Integration:</strong> Organized collection of neuroimaging datasets (OASIS, ABIDE, etc.)</li>
        <li><strong>Analysis Tools:</strong> Curated list of analysis toolkits (GIFT, PAGANI, GRETNA)</li>
        <li><strong>Reference Resources:</strong> Centralized access to brain atlas resources</li>
        <li><strong>Documentation:</strong> User guides and implementation examples</li>
      </ul>
    </div>

    <div class="project-section">
      <h4>Resource Categories</h4>
      <ul>
        <li><strong>Datasets:</strong> Open access neuroimaging databases</li>
        <li><strong>Toolkits:</strong> Open source analysis software and frameworks</li>
        <li><strong>Atlas:</strong> Public brain mapping resources</li>
        <li><strong>Community:</strong> Open for contributions and suggestions</li>
      </ul>
    </div>
  </div>

  <div class="project-card" data-status="in-progress" data-category="applications">
    <div class="project-header">
      <div class="project-title">
        <h3>Journal Classification System for Neuroscience and Psychiatry</h3>
      </div>
      <span class="project-type">Open Source Web Tool</span>
      <span class="project-status status-in-progress">In Progress</span>
      <div class="project-links">
        <a href="https://github.com/JunlinJing/neuroscience_psychiatry_journal_classification" target="_blank" class="github-link">
          <i class="fab fa-github"></i> GitHub
        </a>
        <a href="https://neuroscience-psychiatry-journal-classification.vercel.app" target="_blank" class="demo-link">
          <i class="fas fa-external-link-alt"></i> Live Demo
        </a>
      </div>
    </div>

    <div class="project-tags">
      <span class="tag">#JournalMetrics</span>
      <span class="tag">#AcademicPublishing</span>
      <span class="tag">#SearchTools</span>
      <span class="tag">#Neuroscience</span>
      <span class="tag">#Psychiatry</span>
    </div>

    <div class="project-description">
      An open-source web application that helps researchers navigate and search academic journals in neuroscience and psychiatry, based on the Chinese Academy of Sciences journal classification system.
    </div>

    <div class="project-section">
      <h4>Features</h4>
      <ul>
        <li><strong>Data:</strong> Open access journal classification data</li>
        <li><strong>Search:</strong> Efficient journal discovery tools</li>
        <li><strong>Metrics:</strong> Multiple journal impact measures</li>
        <li><strong>Interface:</strong> Researcher-friendly search experience</li>
      </ul>
    </div>
  </div>

  <div class="project-card" data-status="planning" data-category="tutorials">
    <div class="project-header">
      <div class="project-title">
        <h3>Neuro Cookbook</h3>
      </div>
      <span class="project-type">Open Source Documentation</span>
      <span class="project-status status-planning">Planning</span>
      <a href="https://github.com/JunlinJing/Neuro_cookbook" target="_blank" class="github-link">
        <i class="fab fa-github"></i> GitHub
      </a>
    </div>

    <div class="project-tags">
      <span class="tag">#Tutorials</span>
      <span class="tag">#Neuroimaging</span>
      <span class="tag">#DataAnalysis</span>
      <span class="tag">#Documentation</span>
      <span class="tag">#OpenEducation</span>
    </div>

    <div class="project-description">
      An open-source knowledge base providing practical tutorials and reproducible workflows for neuroscience research tools and methodologies.
    </div>

    <div class="project-section">
      <h4>Planned Contents</h4>
      <ul>
        <li><strong>Tutorials:</strong> Step-by-step guides for open source tools (FSL, SPM, FreeSurfer)</li>
        <li><strong>Workflows:</strong> Reproducible neuroimaging pipelines</li>
        <li><strong>Methods:</strong> Open science practices in neuroscience</li>
        <li><strong>Code:</strong> Shared scripts and implementations</li>
      </ul>
    </div>
  </div>

  <div class="project-card" data-status="planning" data-category="books">
    <div class="project-header">
      <div class="project-title">
        <h3>Brain Connectivity: A Popular Science Guide</h3>
      </div>
      <span class="project-type">Open Science Book</span>
      <span class="project-status status-planning">Planning</span>
    </div>

    <div class="project-tags">
      <span class="tag">#BrainConnectivity</span>
      <span class="tag">#Neuroscience</span>
      <span class="tag">#PopularScience</span>
      <span class="tag">#Education</span>
      <span class="tag">#OpenAccess</span>
    </div>

    <div class="project-description">
      An accessible guide to understanding brain connectivity, written for both general readers and students interested in neuroscience. The book aims to bridge the gap between scientific research and public understanding of how brain networks function.
    </div>

    <div class="project-section">
      <h4>Book Contents</h4>
      <ul>
        <li><strong>Introduction:</strong> Purpose and target audience</li>
        <li><strong>Basic Concepts:</strong> Definition and development of brain connectivity</li>
        <li><strong>Significance:</strong> Impact on individuals and society</li>
        <li><strong>Fundamentals:</strong> Basic theories and key discoveries</li>
        <li><strong>Current Research:</strong> Latest findings and developments</li>
        <li><strong>Practical Applications:</strong> Relevance to daily life</li>
        <li><strong>Future Directions:</strong> Enhancing brain connectivity through daily activities</li>
      </ul>
    </div>

    <div class="project-section">
      <h4>Key Features</h4>
      <ul>
        <li><strong>Accessibility:</strong> Written in clear, engaging language</li>
        <li><strong>Visual Elements:</strong> Illustrations and diagrams</li>
        <li><strong>Case Studies:</strong> Real-world examples and applications</li>
        <li><strong>Interactive:</strong> Practical exercises and activities</li>
      </ul>
    </div>
  </div>
</div>

<div class="pagination">
  <button id="prev-page" class="page-btn" disabled>Previous</button>
  <span id="page-info">Page <span id="current-page">1</span> of <span id="total-pages">1</span></span>
  <button id="next-page" class="page-btn">Next</button>
</div>

<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Lora:wght@400;700&family=Roboto+Slab:wght@400;700&display=swap');

.page-content {
    max-width: 1100px;
    margin: 0 auto;
    padding: 60px 24px;
    font-family: 'Merriweather', 'Lora', 'Georgia', serif;
    background: #fafbfc;
    line-height: 1.8;
}

h1 {
    font-family: 'Roboto Slab', 'Lora', serif;
    font-size: 2.7em;
    color: #1a2340;
    margin-bottom: 1.2em;
    border-bottom: 2px solid #2d3a5a;
    padding-bottom: 0.25em;
    letter-spacing: 0.5px;
}

h2, h3, h4 {
    font-family: 'Roboto Slab', 'Lora', serif;
    color: #23305a;
}

h3 {
    font-size: 1.45em;
    margin: 1.2em 0 0.7em;
    border-bottom: 1px solid #e0e4ea;
    padding-bottom: 0.3em;
}

h4 {
    font-size: 1.15em;
    margin: 1em 0 0.5em;
    color: #2d3a5a;
}

.quote-container {
    max-width: 700px;
    margin: 2.5em auto 2.5em;
    padding: 0 16px;
    text-align: left;
}
blockquote {
    border-left: 4px solid #2d3a5a;
    margin: 0;
    padding: 0.5em 1.5em;
    font-style: italic;
    color: #23305a;
    background: #f5f7fa;
    font-size: 1.25em;
    line-height: 1.7;
    position: relative;
}
.quote-author {
    display: block;
    font-size: 0.9em;
    color: #6c7a99;
    margin-top: 0.8em;
    font-style: normal;
    text-align: right;
}

.project-filters {
    background: #f3f4f7;
    padding: 22px 18px;
    border-radius: 12px;
    margin: 36px 0 32px;
    box-shadow: none;
    border: 1px solid #e0e4ea;
}
.filter-group {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    margin: 14px 0;
}
.filter-group label {
    font-weight: 700;
    color: #23305a;
    min-width: 90px;
    font-size: 1.05em;
    font-family: 'Roboto Slab', 'Lora', serif;
}
.filter-btn {
    background: #fff;
    border: 1px solid #bfc7d1;
    padding: 7px 18px;
    border-radius: 20px;
    font-size: 1em;
    cursor: pointer;
    color: #23305a;
    font-family: 'Lora', serif;
    transition: all 0.18s;
    font-weight: 500;
}
.filter-btn.active {
    background: #23305a;
    color: #fff;
    border-color: #23305a;
    box-shadow: 0 2px 6px rgba(35,48,90,0.08);
}
.filter-btn:hover {
    background: #e6eaf2;
    border-color: #23305a;
}

#projects-container {
    margin-top: 10px;
}
.project-card {
    background: #fff;
    border-radius: 10px;
    border: 1px solid #e0e4ea;
    box-shadow: none;
    padding: 28px 24px 22px 24px;
    margin-bottom: 28px;
    transition: box-shadow 0.2s, border 0.2s;
    text-align: left;
}
.project-card:hover {
    border: 1.5px solid #23305a;
    box-shadow: 0 4px 18px rgba(35,48,90,0.07);
}
.project-header {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: 18px;
    margin-bottom: 18px;
    border-bottom: 1px solid #e0e4ea;
    padding-bottom: 10px;
}
.project-title {
    flex: 1;
    min-width: 220px;
}
.project-title h3 {
    font-size: 1.25em;
    margin: 0;
    color: #1a2340;
    font-family: 'Roboto Slab', 'Lora', serif;
    font-weight: 700;
    line-height: 1.4;
    border: none;
}
.project-type {
    background: #e6eaf2;
    color: #23305a;
    padding: 5px 14px;
    border-radius: 16px;
    font-size: 0.98em;
    font-weight: 500;
    margin-right: 6px;
}
.project-status {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 14px;
    font-size: 0.95em;
    font-weight: 600;
    margin-right: 6px;
}
.status-active {
    background: #e2f3e2;
    color: #1a5928;
    border: 1px solid #b6e3c3;
}
.status-in-progress {
    background: #fdf6e3;
    color: #b35900;
    border: 1px solid #ffe7b8;
}
.status-planning {
    background: #e6f2fa;
    color: #0969da;
    border: 1px solid #b6d8ff;
}
.github-link, .demo-link {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 13px;
    border-radius: 16px;
    font-weight: 500;
    font-size: 0.98em;
    text-decoration: none;
    background: #23305a;
    color: #fff;
    border: none;
    transition: background 0.18s;
}
.github-link:hover, .demo-link:hover {
    background: #1a2340;
    color: #fff;
}
.project-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 18px 0 10px 0;
    padding: 0 2px;
}
.tag {
    background: #f0f1f4;
    color: #4b5563;
    padding: 5px 13px;
    border-radius: 14px;
    font-size: 0.97em;
    font-weight: 500;
    border: 1px solid #e0e4ea;
    font-family: 'Lora', serif;
}
.tag:hover {
    background: #e6eaf2;
}
.project-section {
    margin: 22px 0 10px 0;
    padding: 0 2px;
}
.project-section h4 {
    font-size: 1.08em;
    color: #23305a;
    margin-bottom: 10px;
    font-family: 'Roboto Slab', 'Lora', serif;
}
.project-section ul {
    list-style: none;
    padding: 0;
    margin: 0;
}
.project-section li {
    margin: 10px 0;
    padding-left: 22px;
    position: relative;
    line-height: 1.7;
    font-size: 1.04em;
}
.project-section li:before {
    content: "•";
    color: #23305a;
    font-weight: bold;
    position: absolute;
    left: 0;
}
.project-section li strong {
    color: #1a2340;
    font-weight: 600;
}
.project-description {
    font-size: 1.08em;
    line-height: 1.7;
    color: #23305a;
    margin: 18px 0 10px 0;
    padding: 0 2px;
    text-align: justify;
}

.pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 2em 0 1em 0;
    gap: 1em;
}
.page-btn {
    background: #fff;
    border: 1px solid #bfc7d1;
    padding: 0.45em 1.1em;
    border-radius: 4px;
    cursor: pointer;
    color: #23305a;
    font-family: 'Lora', serif;
    font-size: 1em;
    transition: background 0.18s, border 0.18s;
}
.page-btn:disabled {
    background: #f3f4f7;
    cursor: not-allowed;
    opacity: 0.6;
}
.page-btn:not(:disabled):hover {
    background: #e6eaf2;
    border-color: #23305a;
}
#page-info {
    color: #23305a;
    font-weight: 500;
}

/* Responsive */
@media (max-width: 768px) {
    .page-content {
        padding: 32px 6px;
    }
    .project-card {
        padding: 18px 8px 16px 8px;
    }
    .project-title h3 {
        font-size: 1.08em;
    }
    .project-description {
        font-size: 0.98em;
        line-height: 1.6;
    }
    .filter-group {
        flex-direction: column;
        align-items: stretch;
    }
    .filter-group label {
        margin-bottom: 8px;
    }
    .filter-btn {
        width: 100%;
        text-align: center;
    }
}

html.dark .page-content {
    background: #181c24;
}
html.dark h1, html.dark h2, html.dark h3, html.dark h4 {
    color: #e0e4ea;
}
html.dark .project-card {
    background: #232a3a;
    border: 1px solid #2d3a5a;
}
html.dark .project-card:hover {
    border: 1.5px solid #4a9eff;
    box-shadow: 0 4px 18px rgba(74,158,255,0.07);
}
html.dark .project-title h3, html.dark .project-section li strong {
    color: #e0e4ea;
}
html.dark .project-type {
    background: #2d3a5a;
    color: #b6d8ff;
}
html.dark .project-status.status-active {
    background: #1a5928;
    color: #e2f3e2;
    border-color: #2a7038;
}
html.dark .project-status.status-in-progress {
    background: #804000;
    color: #fdf6e3;
    border-color: #b35900;
}
html.dark .project-status.status-planning {
    background: #054a91;
    color: #e6f2fa;
    border-color: #0969da;
}
html.dark .github-link, html.dark .demo-link {
    background: #4a9eff;
    color: #232a3a;
}
html.dark .github-link:hover, html.dark .demo-link:hover {
    background: #1976d2;
    color: #fff;
}
html.dark .tag {
    background: #2d3a5a;
    color: #b6d8ff;
    border-color: #23305a;
}
html.dark .tag:hover {
    background: #23305a;
}
html.dark .project-section h4 {
    color: #b6d8ff;
}
html.dark .project-section li {
    color: #b6d8ff;
}
html.dark .project-description {
    color: #b6d8ff;
}
html.dark .page-btn {
    background: #232a3a;
    border-color: #2d3a5a;
    color: #b6d8ff;
}
html.dark .page-btn:not(:disabled):hover {
    background: #2d3a5a;
}
html.dark #page-info {
    color: #b6d8ff;
}
html.dark .quote-container blockquote {
    color: #b6d8ff;
    background: #232a3a;
    border-left: 4px solid #4a9eff;
}
html.dark .quote-author {
    color: #b6d8ff;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const projectsContainer = document.getElementById('projects-container');
    const projectCards = document.querySelectorAll('.project-card');
    const statusFilters = document.querySelectorAll('[data-filter]');
    const categoryFilters = document.querySelectorAll('[data-category]');
    
    // Pagination variables
    const cardsPerPage = 3;
    let currentPage = 1;
    let filteredCards = [...projectCards];
    
    // Filter functionality
    function filterProjects() {
        const activeStatusFilter = document.querySelector('[data-filter].active').dataset.filter;
        const activeCategoryFilter = document.querySelector('[data-category].active').dataset.category;
        
        filteredCards = [...projectCards].filter(card => {
            const matchesStatus = activeStatusFilter === 'all' || card.dataset.status === activeStatusFilter;
            const matchesCategory = activeCategoryFilter === 'all' || card.dataset.category === activeCategoryFilter;
            return matchesStatus && matchesCategory;
        });
        
        projectCards.forEach(card => card.style.display = 'none');
        filteredCards.forEach(card => card.style.display = 'block');
        
        currentPage = 1;
        updatePagination();
        showPage(1);
    }
    
    // Pagination functionality
    function updatePagination() {
        const totalPages = Math.ceil(filteredCards.length / cardsPerPage);
        document.getElementById('total-pages').textContent = totalPages;
        document.getElementById('current-page').textContent = currentPage;
        document.getElementById('prev-page').disabled = currentPage === 1;
        document.getElementById('next-page').disabled = currentPage === totalPages;
    }
    
    function showPage(page) {
        const start = (page - 1) * cardsPerPage;
        const end = start + cardsPerPage;
        
        projectCards.forEach(card => card.style.display = 'none');
        filteredCards.slice(start, end).forEach(card => card.style.display = 'block');
    }
    
    // Event Listeners
    statusFilters.forEach(btn => {
        btn.addEventListener('click', (e) => {
            statusFilters.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            filterProjects();
        });
    });
    
    categoryFilters.forEach(btn => {
        btn.addEventListener('click', (e) => {
            categoryFilters.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            filterProjects();
        });
    });
    
    document.getElementById('prev-page').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            showPage(currentPage);
            updatePagination();
        }
    });
    
    document.getElementById('next-page').addEventListener('click', () => {
        const totalPages = Math.ceil(filteredCards.length / cardsPerPage);
        if (currentPage < totalPages) {
            currentPage++;
            showPage(currentPage);
            updatePagination();
        }
    });
    
    // Initial setup
    filterProjects();

    // Update status text display
    const statusElements = document.querySelectorAll('.project-status');
    statusElements.forEach(element => {
        if (element.classList.contains('status-in-development')) {
            element.textContent = 'Planning';
        }
    });
});
</script> 