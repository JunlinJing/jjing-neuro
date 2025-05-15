---
layout: page
title: Projects
permalink: /projects/
---

# Research Projects

## Featured Projects

### BrainDT (Brain-Database-Toolkit)

<div class="project-card">
  <div class="project-header">
    <span class="project-type">Integration Platform</span>
    <a href="https://github.com/JunlinJing/BrainDT" class="project-link" target="_blank">GitHub Repository</a>
  </div>

  <div class="project-description">
    A comprehensive integration platform that combines the latest neuroscience databases, toolkits, and brain atlases to support neuroscience research.
  </div>

  <div class="project-section">
    <h4>Key Features</h4>
    <ul>
      <li><strong>Data Integration:</strong> Curated collection of neuroimaging datasets (OASIS, ABIDE, etc.)</li>
      <li><strong>Analysis Tools:</strong> Integration of popular analysis toolkits (GIFT, PAGANI, GRETNA)</li>
      <li><strong>Reference Resources:</strong> Brain atlas resources for anatomical reference</li>
      <li><strong>Documentation:</strong> Detailed documentation and usage guides</li>
    </ul>
  </div>

  <div class="project-section">
    <h4>Technical Components</h4>
    <ul>
      <li><strong>Datasets:</strong> Comprehensive coverage of major neuroimaging databases</li>
      <li><strong>Toolkits:</strong> Analysis software and frameworks for neuroimaging</li>
      <li><strong>Atlas:</strong> Anatomical reference resources for brain mapping</li>
      <li><strong>Updates:</strong> Regular maintenance with community contributions</li>
    </ul>
  </div>
</div>

### Journal Classification System for Neuroscience and Psychiatry

<div class="project-card">
  <div class="project-header">
    <span class="project-type">Web Application</span>
    <div class="project-links">
      <a href="https://github.com/JunlinJing/neuroscience_psychiatry_journal_classification" target="_blank">GitHub</a>
      <a href="https://neuroscience-psychiatry-journal-classification.vercel.app" target="_blank">Live Demo</a>
    </div>
  </div>

  <div class="project-description">
    A web-based system for classifying and searching academic journals in neuroscience and psychiatry fields based on the latest Chinese Academy of Sciences journal division data.
  </div>

  <div class="project-section">
    <h4>Features</h4>
    <ul>
      <li><strong>Data:</strong> Comprehensive journal classification data</li>
      <li><strong>Search:</strong> Real-time search functionality</li>
      <li><strong>Metrics:</strong> Multiple classification criteria (Impact Factor, CiteScore)</li>
      <li><strong>Interface:</strong> User-friendly data exploration tools</li>
    </ul>
  </div>
</div>

### ChatPSY: Mental Health Assessment System

<div class="project-card">
  <div class="project-header">
    <span class="project-type">AI Application</span>
    <a href="https://github.com/JunlinJing/ChatPSY_demo" target="_blank">GitHub Repository</a>
  </div>

  <div class="project-description">
    An AI-powered mental health consultation system that provides preliminary assessments and guidance using standardized psychiatric scales.
  </div>

  <div class="project-section">
    <h4>Key Features</h4>
    <ul>
      <li><strong>Assessment:</strong> Standardized psychiatric scale evaluations</li>
      <li><strong>Guidance:</strong> Personalized recommendations based on severity</li>
      <li><strong>Education:</strong> Scientific background for mental health concerns</li>
      <li><strong>Support:</strong> Professional referral guidance when needed</li>
    </ul>
  </div>

  <div class="project-section">
    <h4>Technical Stack</h4>
    <ul>
      <li><strong>AI Integration:</strong> LLM API for natural language understanding</li>
      <li><strong>Framework:</strong> Langchain for context management</li>
      <li><strong>Interface:</strong> Gradio-based interactive consultation</li>
      <li><strong>Method:</strong> Few-shot prompting for medical responses</li>
    </ul>
  </div>
</div>

### Neuro Cookbook

<div class="project-card">
  <div class="project-header">
    <span class="project-type">Educational Resource</span>
    <span class="project-status">In Development</span>
    <a href="https://github.com/JunlinJing/Neuro_cookbook" target="_blank">GitHub Repository</a>
  </div>

  <div class="project-description">
    A comprehensive tutorial collection for neuroscience research tools and methodologies, providing practical guides and hands-on tutorials.
  </div>

  <div class="project-section">
    <h4>Planned Contents</h4>
    <ul>
      <li><strong>Software:</strong> Tutorials for FSL, SPM, FreeSurfer</li>
      <li><strong>Workflows:</strong> fMRI data preprocessing and analysis</li>
      <li><strong>Methods:</strong> Connectivity analysis and statistical tools</li>
      <li><strong>Code:</strong> Python/MATLAB scripts for neuroscience</li>
    </ul>
  </div>
</div>

<style>
.page-content {
    max-width: 1000px;
    margin: 0 auto;
    padding: 40px 20px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

h1 {
    font-size: 2.5em;
    color: #2c3e50;
    margin-bottom: 1.5em;
    border-bottom: 3px solid #3498db;
    padding-bottom: 0.3em;
}

h2 {
    font-size: 2em;
    color: #34495e;
    margin: 1.5em 0 1em;
}

h3 {
    font-size: 1.5em;
    color: #2c3e50;
    margin: 1.5em 0 1em;
}

h4 {
    font-size: 1.2em;
    color: #34495e;
    margin: 1em 0 0.5em;
}

.project-card {
    background: #ffffff;
    border: 1px solid #e1e4e8;
    border-radius: 8px;
    padding: 25px;
    margin: 20px 0 40px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.project-card:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transform: translateY(-2px);
    transition: all 0.3s ease;
}

.project-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.project-type {
    background: #e3f2fd;
    color: #1976d2;
    padding: 4px 12px;
    border-radius: 15px;
    font-size: 0.9em;
    font-weight: 500;
}

.project-status {
    background: #fff3e0;
    color: #f57c00;
    padding: 4px 12px;
    border-radius: 15px;
    font-size: 0.9em;
    font-weight: 500;
    margin-left: 10px;
}

.project-links a {
    color: #3498db;
    text-decoration: none;
    margin-left: 15px;
    font-weight: 500;
}

.project-links a:hover {
    text-decoration: underline;
}

.project-description {
    font-size: 1.1em;
    line-height: 1.6;
    color: #2c3e50;
    margin-bottom: 20px;
}

.project-section {
    margin: 20px 0;
}

.project-section ul {
    list-style-type: none;
    padding-left: 0;
}

.project-section li {
    margin: 10px 0;
    line-height: 1.6;
    color: #34495e;
}

.project-section li strong {
    color: #2c3e50;
    margin-right: 8px;
}

a {
    color: #3498db;
    text-decoration: none;
    transition: color 0.2s ease;
}

a:hover {
    color: #2980b9;
    text-decoration: underline;
}

@media (max-width: 768px) {
    .project-header {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .project-links {
        margin-top: 10px;
    }
    
    .project-links a {
        margin: 5px 15px 5px 0;
        display: inline-block;
    }
}
</style> 