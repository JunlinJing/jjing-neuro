---
layout: page
title: Research
permalink: /research/
---

<style>
.page-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 60px 30px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

h1.page-title {
    font-size: 3em;
    color: #2c3e50;
    margin-bottom: 1.2em;
    line-height: 1.4;
    border-bottom: 3px solid #3498db;
    padding-bottom: 0.3em;
}

.research-intro {
    font-size: 1.25em;
    line-height: 1.8;
    color: #2c3e50;
    margin-bottom: 3em;
    padding: 0 10px;
}

h2.section-title {
    font-size: 2.2em;
    color: #34495e;
    margin: 2em 0 1em;
    padding-bottom: 0.5em;
    border-bottom: 2px solid #3498db;
}

h3.project-title {
    font-size: 1.8em;
    color: #2c3e50;
    margin: 1.5em 0 1em;
    line-height: 1.4;
}

.project-description {
    font-size: 1.15em;
    line-height: 1.8;
    color: #34495e;
    margin: 1.5em 0;
    padding: 0 10px;
}

.project-list {
    list-style: none;
    padding: 0;
    margin: 1.5em 0;
}

.project-list li {
    font-size: 1.15em;
    line-height: 1.8;
    margin: 1em 0;
    padding-left: 25px;
    position: relative;
    color: #34495e;
}

.project-list li:before {
    content: "•";
    color: #3498db;
    font-weight: bold;
    position: absolute;
    left: 0;
}

.methods-section {
    margin: 3em 0;
    padding: 0 10px;
}

.methods-category {
    margin: 2em 0;
}

.methods-title {
    font-size: 1.4em;
    color: #2c3e50;
    font-weight: 600;
    margin-bottom: 1em;
}

.methods-list {
    list-style: none;
    padding: 0;
    margin: 1em 0;
}

.methods-list li {
    font-size: 1.15em;
    line-height: 1.8;
    margin: 0.8em 0;
    padding-left: 25px;
    position: relative;
    color: #34495e;
}

.methods-list li:before {
    content: "•";
    color: #3498db;
    font-weight: bold;
    position: absolute;
    left: 0;
}

/* 夜间模式样式 */
html.dark h1.page-title {
    color: #ffffff;
    border-bottom-color: #4a9eff;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

html.dark .research-intro {
    color: #e6e6e6;
    background-color: rgba(45, 45, 45, 0.6);
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    border-left: 3px solid #4a9eff;
}

html.dark h2.section-title {
    color: #ffffff;
    border-bottom-color: #4a9eff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

html.dark h3.project-title {
    color: #ffffff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

html.dark .project-description {
    color: #e6e6e6;
    background-color: rgba(45, 45, 45, 0.5);
    padding: 15px;
    border-radius: 6px;
}

html.dark .project-list li,
html.dark .methods-list li {
    color: #e0e0e0;
}

html.dark .project-list li:before,
html.dark .methods-list li:before {
    color: #4a9eff;
}

html.dark .methods-title {
    color: #ffffff;
}

@media (max-width: 768px) {
    .page-content {
        padding: 40px 20px;
    }
    
    h1.page-title {
        font-size: 2.5em;
    }
    
    h2.section-title {
        font-size: 2em;
    }
    
    h3.project-title {
        font-size: 1.6em;
    }
    
    .research-intro,
    .project-description,
    .project-list li,
    .methods-list li {
        font-size: 1.1em;
        line-height: 1.7;
    }
    
    .methods-title {
        font-size: 1.3em;
    }
}
</style>

<h1 class="page-title">Research</h1>

<div class="research-intro">
My research focuses on machine learning applications in neuroimaging analysis, particularly in the investigation of functional brain connectivity patterns in neurological and psychiatric disorders. I combine advanced computational methods with resting-state functional MRI data to develop novel approaches for personalized diagnosis and treatment evaluation.
</div>

<h2 class="section-title">Current Projects</h2>

<h3 class="project-title">Use of Machine Learning for Investigation of Deviant Functional Connectome</h3>

<div class="project-description">
In recent years, Intrinsic Functional Connectivity Magnetic Resonance Imaging (fcMRI) has become a valuable tool for understanding how different parts of the human brain communicate with each other. This project investigates:
</div>

<ul class="project-list">
    <li>Analysis of resting-state functional MRI scans for brain connectivity mapping</li>
    <li>Development of machine learning approaches for detecting abnormal connectivity patterns</li>
    <li>Investigation of disease-specific network alterations in neurological and psychiatric disorders</li>
    <li>Implementation of personalized diagnostic tools based on connectivity biomarkers</li>
</ul>

<h3 class="project-title">Clinical Applications of Brain Connectivity Analysis</h3>

<div class="project-description">
Translating neuroimaging research into clinical practice through:
</div>

<ul class="project-list">
    <li>Development of automated analysis pipelines for clinical fcMRI data</li>
    <li>Investigation of tumor-related changes in brain connectivity</li>
    <li>Assessment of whole-brain connectivity alterations in various pathological conditions</li>
</ul>

<h2 class="section-title">Research Methods</h2>

<div class="methods-section">
    <div class="methods-category">
        <div class="methods-title">Neuroimaging Analysis</div>
        <ul class="methods-list">
            <li>Resting-state fMRI processing</li>
            <li>Functional connectivity analysis</li>
            <li>Advanced machine learning applications</li>
        </ul>
    </div>

    <div class="methods-category">
        <div class="methods-title">Data Analysis</div>
        <ul class="methods-list">
            <li>Large-scale neuroimaging data processing</li>
            <li>Statistical modeling and inference</li>
            <li>Pattern recognition in clinical data</li>
        </ul>
    </div>

    <div class="methods-category">
        <div class="methods-title">Computational Approaches</div>
        <ul class="methods-list">
            <li>Deep learning for neuroimaging</li>
            <li>Network analysis and graph theory</li>
            <li>Multivariate pattern analysis</li>
        </ul>
    </div>
</div>

<style>
.page-content {
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
}

h1, h2, h3 {
    color: #2c3e50;
    margin-top: 30px;
}

html.dark h1, 
html.dark h2, 
html.dark h3 {
    color: #ffffff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

h2 {
    border-bottom: 2px solid #eee;
    padding-bottom: 10px;
}

html.dark h2 {
    border-bottom: 2px solid #444;
}

ul {
    padding-left: 20px;
}

li {
    margin-bottom: 10px;
    line-height: 1.6;
}

html.dark li {
    color: #e0e0e0;
}

strong {
    color: #34495e;
}

html.dark strong {
    color: #ffffff;
}

a {
    color: #3498db;
    text-decoration: none;
}

html.dark a {
    color: #6bb9ff;
}

a:hover {
    text-decoration: underline;
}

html.dark a:hover {
    color: #ffffff;
}
</style> 