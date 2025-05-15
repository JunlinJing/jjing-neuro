---
layout: page
title: Projects
permalink: /projects/
---

# Research Projects

<div class="projects-container">

<div class="project-card">
<div class="project-header">
### Journal Classification System
<div class="project-tags">
<span class="tag">Web Development</span>
<span class="tag">Data Analysis</span>
</div>
</div>

A web-based system for classifying and searching academic journals in neuroscience and psychiatry fields based on the latest Chinese Academy of Sciences journal division data.

**Key Features:**
- Comprehensive journal classification data
- Real-time search functionality
- Multiple classification criteria (Impact Factor, CiteScore, etc.)
- User-friendly interface for data exploration

<div class="project-links">
<a href="https://github.com/JunlinJing/neuroscience_psychiatry_journal_classification" class="button" target="_blank">View on GitHub</a>
<a href="https://neuroscience-psychiatry-journal-classification.vercel.app" class="button primary" target="_blank">Live Demo</a>
</div>
</div>

<div class="project-card">
<div class="project-header">
### ChatPSY
<div class="project-tags">
<span class="tag">AI</span>
<span class="tag">Healthcare</span>
<span class="tag">Python</span>
</div>
</div>

An AI-powered mental health consultation system that provides preliminary assessments and guidance using standardized psychiatric scales.

**Key Features:**
- Detailed introduction and scientific background for mental health concerns
- Standardized psychiatric scale assessments with result interpretation
- Personalized recommendations based on assessment severity
- Professional referral guidance system

**Technical Stack:**
- LLM API Integration
- Langchain Framework
- Few-shot Learning
- Gradio UI

<div class="project-links">
<a href="https://github.com/JunlinJing/ChatPSY_demo" class="button" target="_blank">View on GitHub</a>
</div>
</div>

<div class="project-card">
<div class="project-header">
### Neuro Cookbook
<div class="project-tags">
<span class="tag">Documentation</span>
<span class="tag">Tutorials</span>
<span class="tag">Research</span>
</div>
</div>

A comprehensive tutorial collection for neuroscience research tools and methodologies, providing practical guides and hands-on tutorials.

**Current Focus:**
- Neuroimaging software (FSL, SPM, FreeSurfer)
- fMRI data analysis workflows
- Connectivity analysis methods
- Statistical analysis techniques
- Python/MATLAB implementations

**Development Status:** In Progress

<div class="project-links">
<a href="https://github.com/JunlinJing/Neuro_cookbook" class="button" target="_blank">View on GitHub</a>
</div>
</div>

</div>

<style>
.projects-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    padding: 20px 0;
}

.project-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border: 1px solid #eaeaea;
}

.project-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 15px rgba(0,0,0,0.1);
}

.project-header {
    margin-bottom: 20px;
}

.project-header h3 {
    margin: 0 0 10px 0;
    color: #2c3e50;
    font-size: 1.5em;
}

.project-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

.tag {
    background: #f7f9fc;
    color: #4a5568;
    padding: 4px 12px;
    border-radius: 15px;
    font-size: 0.85em;
    font-weight: 500;
}

.project-links {
    margin-top: 20px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.button {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: 500;
    text-decoration: none;
    transition: all 0.2s ease;
    background: #f7f9fc;
    color: #4a5568;
    border: 1px solid #eaeaea;
}

.button.primary {
    background: #3498db;
    color: white;
    border: none;
}

.button:hover {
    transform: translateY(-2px);
    text-decoration: none;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

h1 {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 2rem;
    font-size: 2.5em;
    position: relative;
}

h1::after {
    content: '';
    display: block;
    width: 60px;
    height: 4px;
    background: #3498db;
    margin: 20px auto;
    border-radius: 2px;
}

ul {
    padding-left: 20px;
    margin: 15px 0;
}

li {
    margin-bottom: 8px;
    line-height: 1.6;
    color: #4a5568;
}

strong {
    color: #2c3e50;
    font-weight: 600;
}

@media (max-width: 768px) {
    .projects-container {
        grid-template-columns: 1fr;
        gap: 1.5rem;
    }
    
    .project-card {
        padding: 20px;
    }
    
    h1 {
        font-size: 2em;
    }
}
</style> 