document.addEventListener('DOMContentLoaded', function() {
    // Generate table of contents
    generateTableOfContents();
    // Create reading progress bar
    createProgressBar();
    // Add code copy buttons
    addCodeCopyButtons();
    // Add image zoom functionality
    addImageZoom();
    // Add back to top button
    addBackToTop();
    // Add code line numbers
    addLineNumbers();
});

// Generate table of contents
function generateTableOfContents() {
    const content = document.querySelector('.post-content');
    const toc = document.getElementById('toc-content');
    if (!content || !toc) return;

    const headings = content.querySelectorAll('h2, h3, h4');
    if (headings.length === 0) {
        document.getElementById('toc')?.classList.add('hidden');
        return;
    }

    const tocList = document.createElement('ul');
    headings.forEach((heading, index) => {
        const id = `heading-${index}`;
        heading.id = id;

        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = `#${id}`;
        a.textContent = heading.textContent;
        a.className = `toc-${heading.tagName.toLowerCase()}`;

        li.appendChild(a);
        tocList.appendChild(li);

        // Smooth scroll when clicking TOC items
        a.addEventListener('click', (e) => {
            e.preventDefault();
            heading.scrollIntoView({ behavior: 'smooth' });
        });
    });

    toc.appendChild(tocList);
}

// Create reading progress bar
function createProgressBar() {
    const progressBar = document.createElement('div');
    progressBar.className = 'reading-progress';
    document.body.appendChild(progressBar);

    window.addEventListener('scroll', () => {
        const docElement = document.documentElement;
        const docBody = document.body;
        const scrollTop = docElement.scrollTop || docBody.scrollTop;
        const scrollHeight = docElement.scrollHeight || docBody.scrollHeight;
        const clientHeight = docElement.clientHeight;
        const scrollPercentage = (scrollTop / (scrollHeight - clientHeight)) * 100;
        progressBar.style.width = `${scrollPercentage}%`;
    });
}

// Add code copy buttons
function addCodeCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(code => {
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.innerHTML = '<i class="far fa-copy"></i>';
        
        copyButton.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(code.textContent);
                copyButton.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    copyButton.innerHTML = '<i class="far fa-copy"></i>';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy code:', err);
            }
        });

        code.parentNode.style.position = 'relative';
        code.parentNode.appendChild(copyButton);
    });
}

// Add image zoom functionality
function addImageZoom() {
    const images = document.querySelectorAll('.post-content img');
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    document.body.appendChild(modal);

    images.forEach(img => {
        img.style.cursor = 'zoom-in';
        img.addEventListener('click', () => {
            modal.innerHTML = `<img src="${img.src}" alt="${img.alt}">`;
            modal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        });
    });

    modal.addEventListener('click', () => {
        modal.style.display = 'none';
        document.body.style.overflow = '';
    });
}

// Add back to top button
function addBackToTop() {
    const button = document.createElement('button');
    button.className = 'back-to-top';
    button.innerHTML = '<i class="fas fa-arrow-up"></i>';
    document.body.appendChild(button);

    window.addEventListener('scroll', () => {
        if (window.scrollY > 300) {
            button.classList.add('show');
        } else {
            button.classList.remove('show');
        }
    });

    button.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// Add code line numbers
function addLineNumbers() {
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(code => {
        const lines = code.innerHTML.split('\n');
        const numberedLines = lines.map((line, index) => 
            `<span class="line-number">${index + 1}</span>${line}`
        ).join('\n');
        code.innerHTML = numberedLines;
    });
}

// Update TOC highlight
function updateTocHighlight() {
    const headings = document.querySelectorAll('.post-content h2, .post-content h3, .post-content h4');
    const tocLinks = document.querySelectorAll('#toc-content a');
    
    let currentHeading = null;
    headings.forEach(heading => {
        const rect = heading.getBoundingClientRect();
        if (rect.top <= 100) {
            currentHeading = heading;
        }
    });

    tocLinks.forEach(link => {
        link.classList.remove('active');
        if (currentHeading && link.getAttribute('href') === `#${currentHeading.id}`) {
            link.classList.add('active');
        }
    });
}

// Listen for scroll events to update TOC highlight
window.addEventListener('scroll', updateTocHighlight); 